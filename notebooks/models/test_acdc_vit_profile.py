import accelerate
import fire
import torch
import transformers
from rich import print
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

import gate.data.medical.segmentation.automated_cardiac_diagnosis as acd
import gate.data.medical.segmentation.medical_decathlon as md
from gate.models.adapters.medical_semantic_segmentation import logger
from gate.models.task_specific_models.semantic_segmentation.timm import (
    build_gate_model,
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
            label_image_size=target_image_size,
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


def main(
    dataset_name: str = "md",
    data_dir: str = "/data/",
    image_size: int = 512,
    target_image_size: int = 256,
    batch_size: int = 1,
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

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            for i in range(100):
                if eval_mode:
                    with torch.no_grad():
                        output = model.forward(input_dict)
                        loss = output["image"]["image"]["loss"]
                else:
                    optimizer.zero_grad()
                    output = model.forward(input_dict)
                    loss = output["image"]["image"]["loss"]
                    accelerator.backward(loss)
                    optimizer.step()

    # Show the results sorted by CUDA memory usage in descending order
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_memory_usage",
            row_limit=10,
            top_level_events_only=False,
        )
    )

    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    fire.Fire(main)
