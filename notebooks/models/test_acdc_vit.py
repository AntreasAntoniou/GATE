import multiprocessing as mp

import accelerate
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from rich import print
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import gate.data.medical.segmentation.automated_cardiac_diagnosis as acd
import gate.data.medical.segmentation.medical_decathlon as md
from gate.models.task_adapters.medical_semantic_segmentation import logger
from gate.models.task_specific_models.semantic_segmentation.timm import (
    ModelAndTransform,
    build_gate_model,
    build_model,
)

logger.setLevel("DEBUG")


def build_dataloader(
    dataset_name,
    data_dir,
    image_size,
    target_image_size,
    transforms,
    batch_size=1,
    num_workers=12,
):
    if dataset_name == "md":
        data = md.build_gate_dataset(
            data_dir=data_dir,
            image_size=image_size,
            target_image_size=target_image_size,
            transforms=transforms,
        )
    elif dataset_name == "acd":
        data = acd.build_gate_dataset(
            data_dir=data_dir,
            image_size=image_size,
            target_image_size=target_image_size,
            transforms=transforms,
        )
    else:
        raise ValueError("Invalid dataset name")

    return DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def sub_batch_generator(batch_dict, sub_batch_size):
    """
    Generator function to yield sub-batches from a given batch dictionary.

    Parameters:
    - batch_dict (dict): Dictionary containing original batch data. Each key maps to a tensor or list.
    - sub_batch_size (int): Size of each sub-batch to be returned.

    Yields:
    - dict: Dictionary containing sub-batch data.
    """
    batch_size = None

    # Validate input and get original batch size
    for key, value in batch_dict.items():
        if batch_size is None:
            batch_size = value.shape[0] * value.shape[1]
        elif batch_size != value.shape[0] * value.shape[1]:
            raise ValueError(
                f"Batch sizes for different keys in batch_dict must be the same. Mismatch at key: {key}"
            )

    if sub_batch_size > batch_size:
        raise ValueError(
            "Sub-batch size cannot be greater than the original batch size."
        )

    # Generate and yield sub-batches
    for start_idx in range(0, batch_size, sub_batch_size):
        end_idx = min(start_idx + sub_batch_size, batch_size)
        sub_batch = {}

        for key, value in batch_dict.items():
            sub_batch[key] = value.reshape(-1, *value.shape[2:])[
                start_idx:end_idx
            ]

        yield sub_batch


def main(
    dataset_name: str = "md",
    data_dir: str = "/data/",
    image_size: int = 512,
    target_image_size: int = 256,
    batch_size: int = 1,
    sub_batch_size: int = 1,
    num_workers: int = 12,
    eval_mode: bool = False,
):
    model_and_transform = build_gate_model(
        timm_model_name="vit_base_patch16_clip_224.laion2b",
        num_classes=100,
        pretrained=True,
        use_temporal_model=True,
        num_channels=3,
    )

    model = model_and_transform.model
    transforms = model_and_transform.transform
    dataloader = build_dataloader(
        dataset_name,
        data_dir,
        image_size,
        target_image_size,
        transforms,
        batch_size,
        num_workers,
    )

    accelerator = accelerate.Accelerator(mixed_precision="fp16")
    model = accelerator.prepare(model)
    dataloader = accelerator.prepare(dataloader)
    optimizer = transformers.AdamW(
        model.parameters(), lr=1e-5, weight_decay=0.0
    )
    optimizer = accelerator.prepare(optimizer)

    input_dict = next(iter(dataloader))

    for key, value in input_dict.items():
        print(key, value.shape)

    with tqdm(total=100) as pbar:
        for i in range(100):
            for batch in sub_batch_generator(input_dict, sub_batch_size):
                if eval_mode:
                    with torch.no_grad():
                        output = model.forward(batch)
                        loss = output["image"]["image"]["loss"]
                else:
                    optimizer.zero_grad()
                    output = model.forward(batch)
                    loss = output["image"]["image"]["loss"]
                    accelerator.backward(loss)
                    optimizer.step()
            pbar.update(1)


if __name__ == "__main__":
    fire.Fire(main)
