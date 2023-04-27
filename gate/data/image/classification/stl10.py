# stl10.py
import os
from typing import Optional

import numpy as np
import torch
import torchvision
from datasets import load_dataset
from py import test
from torch.utils.data import Subset, random_split


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
