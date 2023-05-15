# stl10.py
import os
from typing import Any, Dict, Optional

import torch
import torchvision
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import (
    ClassificationTask,
)
from gate.data.transforms.tiny_image_transforms import pad_image


def build_stl10_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a Food-101 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    # Create a generator with the specified seed
    rng = torch.Generator().manual_seed(42)
    try:
        data = torchvision.datasets.STL10(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/stl10-train/"),
            split="train",
            download=True,
        )
    except RuntimeError:
        data = torchvision.datasets.STL10(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/stl10-train/"),
            split="train",
            download=False,
        )

    dataset_length = len(data)
    val_split = 0.1  # Fraction for the validation set (e.g., 10%)

    # Calculate the number of samples for train and validation sets
    val_length = int(dataset_length * val_split)
    train_length = dataset_length - val_length

    # Split the dataset into train and validation sets using the generator
    train_data, val_data = random_split(
        data, [train_length, val_length], generator=rng
    )
    try:
        test_data = torchvision.datasets.STL10(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/stl10-test/"),
            split="test",
            download=True,
        )
    except RuntimeError:
        test_data = torchvision.datasets.STL10(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/stl10-test/"),
            split="test",
            download=False,
        )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


def transform_wrapper(inputs: Dict, target_size=224):
    return {
        "image": pad_image(inputs["image"], target_size=target_size),
        "labels": inputs["labels"],
    }


@configurable(
    group="dataset", name="stl10", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_stl10_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=10,
) -> dict:
    train_set = GATEDataset(
        dataset=build_stl10_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        item_keys=["image", "labels"],
        transforms=[transform_wrapper, transforms],
    )

    val_set = GATEDataset(
        dataset=build_stl10_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        item_keys=["image", "labels"],
        transforms=[transform_wrapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_stl10_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        item_keys=["image", "labels"],
        transforms=[transform_wrapper, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_stl10_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass
