# cityscapes.py
import os
from typing import Optional

import torch
import torchvision


def build_cityscapes_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
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
        train_data = torchvision.datasets.Cityscapes(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/cifar100-train/"),
            split="train",
            mode="fine",
            target_type="semantic",
            download=True,
        )
    except RuntimeError:
        train_data = torchvision.datasets.Cityscapes(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser(
                "~/.cache/torch/datasets/cityscapes-train/"
            ),
            split="train",
            mode="fine",
            target_type="semantic",
            download=False,
        )

    try:
        val_data = torchvision.datasets.Cityscapes(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/cityscapes-val/"),
            split="val",
            mode="fine",
            target_type="semantic",
            download=True,
        )
    except RuntimeError:
        val_data = torchvision.datasets.Cityscapes(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser("~/.cache/torch/datasets/cityscapes-val/"),
            split="test",
            mode="fine",
            target_type="semantic",
            download=False,
        )

    try:
        test_data = torchvision.datasets.Cityscapes(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser(
                "~/.cache/torch/datasets/cityscapes-test/"
            ),
            split="test",
            mode="fine",
            target_type="semantic",
            download=True,
        )
    except RuntimeError:
        test_data = torchvision.datasets.Cityscapes(
            root=data_dir
            if data_dir is not None
            else os.path.expanduser(
                "~/.cache/torch/datasets/cityscapes-test/"
            ),
            split="test",
            mode="fine",
            target_type="semantic",
            download=False,
        )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]
