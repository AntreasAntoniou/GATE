import os
import pytest
from torch.utils.data import DataLoader
import torch.nn.functional as F
from gate.data.core import collate_fn_with_token_pad
from gate.data.image.classification.imagenet1k import build_gate_dataset
from gate.models import ModelAndTransform
from gate.metrics import accuracy_top_k

from gate.models.task_specific_models.zero_shot_classification.clip import (
    build_gate_model_with_presets,
)
from accelerate import Accelerator
from rich import print
from rich.traceback import install

install()

accelerator = Accelerator()


def test_build_model():
    model = build_gate_model_with_presets()


def test_build_dataset():
    model: ModelAndTransform = build_gate_model_with_presets()
    dataset = build_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"), transforms=model.transform
    )


def test_forward():
    model_and_transform: ModelAndTransform = build_gate_model_with_presets()
    model = model_and_transform.model
    model = accelerator.prepare(model)
    dataset = build_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
        transforms=model_and_transform.transform,
    )
    dataloader = DataLoader(
        dataset["train"],
        batch_size=2,
        collate_fn=collate_fn_with_token_pad,
        shuffle=True,
        num_workers=4,
    )
    dataloader = accelerator.prepare(dataloader)
    sample = next(iter(dataloader))

    model.forward(sample)


def test_forward_backward():
    model_and_transform: ModelAndTransform = build_gate_model_with_presets()
    model = model_and_transform.model
    model = accelerator.prepare(model)
    dataset = build_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
        transforms=model_and_transform.transform,
    )
    dataloader = DataLoader(
        dataset["train"],
        batch_size=2,
        collate_fn=collate_fn_with_token_pad,
        shuffle=True,
        num_workers=4,
    )
    dataloader = accelerator.prepare(dataloader)
    sample = next(iter(dataloader))

    outputs = model.forward(sample)["image"]
    loss = F.cross_entropy(outputs["logits"], sample["label"])
    accuracy = accuracy_top_k(outputs["logits"], sample["label"], k=1)

    print(f"loss: {loss}, accuracy: {accuracy}")


if __name__ == "__main__":
    test_forward()
    test_forward_backward()
