import multiprocessing as mp

import accelerate
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
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


def main(
    dataset_name="md",
    num_samples=100,
    data_dir="/data/",
    image_size=512,
    target_image_size=256,
    batch_size=1,
    num_workers=12,
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
        input_dict[key] = value[:, :num_samples]

    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():
            with tqdm(total=100) as pbar:
                for i in range(100):
                    optimizer.zero_grad()
                    output = model.forward(input_dict)
                    loss = output["image"]["image"]["loss"]
                    accelerator.backward(loss)
                    optimizer.step()
                    pbar.update(1)
                    pbar.set_description(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    fire.Fire(main)
