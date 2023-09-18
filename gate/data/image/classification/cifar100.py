# cifar100.py
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torchvision
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations
from gate.data.transforms.tiny_image_transforms import pad_image


def build_cifar100_dataset(
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
        data = torchvision.datasets.CIFAR100(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/cifar100-train/"),
            train=True,
            download=True,
        )
    except RuntimeError:
        data = torchvision.datasets.CIFAR100(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/cifar100-train/"),
            train=True,
            download=False,
        )

    dataset_length = len(data)
    val_split = 0.05  # Fraction for the validation set (e.g., 10%)

    # Calculate the number of samples for train and validation sets
    val_length = int(dataset_length * val_split)
    train_length = dataset_length - val_length

    # Split the dataset into train and validation sets using the generator
    train_data, val_data = random_split(
        data, [train_length, val_length], generator=rng
    )

    try:
        test_data = torchvision.datasets.CIFAR100(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/cifar100-test/"),
            train=False,
            download=True,
        )
    except RuntimeError:
        test_data = torchvision.datasets.CIFAR100(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/cifar100-test/"),
            train=False,
            download=False,
        )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


def transform_wrapper(inputs: Tuple, target_size=224):
    return {
        "image": pad_image(inputs[0], target_size=target_size),
        "labels": inputs[1],
    }


@configurable(
    group="dataset", name="cifar100", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_cifar100_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=100,
):
    augmentations = StandardAugmentations(image_key="image")
    train_set = GATEDataset(
        dataset=build_cifar100_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[transform_wrapper, augmentations, transforms],
    )

    val_set = GATEDataset(
        dataset=build_cifar100_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_cifar100_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_cifar100_dataset(transforms: Optional[Any] = None):
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 100
