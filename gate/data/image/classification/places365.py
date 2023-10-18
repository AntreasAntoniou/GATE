# places365.py
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torchvision
from timm.data import rand_augment_transform
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations


def build_places365_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a Food-101 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    # Create a generator with the specified seed
    rng = torch.Generator().manual_seed(42)

    try:
        data = torchvision.datasets.Places365(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser(
                "~/.cache/torch/datasets/places365-train/"
            ),
            split="train-standard",
            small=True,
            download=True,
        )
    except RuntimeError:
        data = torchvision.datasets.Places365(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser(
                "~/.cache/torch/datasets/places365-train/"
            ),
            split="train-standard",
            small=True,
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
        test_data = torchvision.datasets.Places365(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/places365-val/"),
            split="val",
            small=True,
            download=True,
        )
    except RuntimeError:
        test_data = torchvision.datasets.Places365(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/places365-val/"),
            split="val",
            small=True,
            download=False,
        )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


def data_format_transform(inputs):
    return {"image": inputs[0], "labels": inputs[1]}


@configurable(
    group="dataset", name="places365", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_places365_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=365,
) -> dict:
    train_set = GATEDataset(
        dataset=build_places365_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            data_format_transform,
            StandardAugmentations(image_key="image"),
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=build_places365_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[data_format_transform, transforms],
    )

    test_set = GATEDataset(
        dataset=build_places365_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[data_format_transform, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_places365_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 1000
