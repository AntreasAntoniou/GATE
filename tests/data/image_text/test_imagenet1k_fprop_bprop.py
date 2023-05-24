import os
import pytest
from torch.utils.data import DataLoader
from gate.data.core import collate_fn_with_token_pad
from gate.data.image.classification.imagenet1k import build_gate_dataset
from gate.models import ModelAndTransform

from gate.models.task_specific_models.zero_shot_classification.clip import (
    build_gate_model_with_presets,
)


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
    sample = next(iter(dataloader))

    model.forward(sample)


def test_forward_backward():
    model: ModelAndTransform = build_gate_model_with_presets()
    dataset = build_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"), transforms=model.transform
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    sample = next(iter(dataloader))

    outputs = model.forward(image=sample["image"])
    loss = F.cross_entropy(outputs["logits"], sample["label"])
