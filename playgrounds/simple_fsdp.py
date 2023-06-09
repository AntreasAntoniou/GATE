# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import always_wrap_policy, enable_wrap, wrap
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from gate.data.image_text.zero_shot.flickr30k import (
    build_dataset,
    build_gate_dataset,
)
from gate.models.task_specific_models.zero_shot_classification.clip import (
    build_gate_model,
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(model, rank, dataloader, optimizer, epoch):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)["image_text"]["image_text"]
        output["loss"].backward()
        optimizer.step()
        ddp_loss[0] += output["loss"].item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            "Train Epoch: {} \tLoss: {:.6f}".format(
                epoch, ddp_loss[0] / ddp_loss[1]
            )
        )


def test(model, rank, dataloader):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
            )
        )


def main(
    model,
    rank,
    world_size,
    batch_size,
    dataset,
    epochs,
    lr,
    gamma,
    save_model,
    seed,
):
    setup(rank, world_size)

    sampler = DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
    )

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = model.to(rank)

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload.offload_params,
        auto_wrap_policy=always_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, cast_forward_inputs=True
        ),
        device_id=torch.cuda.current_device(),
    )

    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    init_start_event.record()
    for epoch in range(1, epochs + 1):
        train(
            model=model,
            rank=rank,
            dataloader=dataloader,
            optimizer=optimizer,
        )
        test(model, rank, world_size, dataloader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")

    if save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        # state_dict for FSDP model is only available on Nightlies for now
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()


if __name__ == "__main__":
    model_and_transform = build_gate_model()
    transform = model_and_transform.transform
    model = model_and_transform.model
    rank = 0

    batch_size = 64
    dataset = build_gate_dataset(data_dir="/data1/", transforms=transform)[
        "train"
    ]
    epochs = 1
    lr = 1e-5
    gamma = 0.7
    save_model = True
    seed = 42

    torch.manual_seed(seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(
        main,
        (
            model,
            rank,
            WORLD_SIZE,
            batch_size,
            dataset,
            epochs,
            lr,
            gamma,
            save_model,
            seed,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
