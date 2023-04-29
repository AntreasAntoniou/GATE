# pascal_context.py
import os
from typing import Optional

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset, random_split


def build_pascal_context_dataset(
    set_name: str, data_dir: Optional[str] = None, download: bool = False
) -> Dataset:
    """
    Build the Pascal Context dataset.

    Args:
        set_name (str): The name of the dataset split to return
        ("train", "val" or "test").
        data_dir (Optional[str]): The directory where the dataset is stored.
        Default: None.
        download (bool): Whether to download the dataset if
        not already present. Default: False.

    Returns:
        A Dataset object containing the specified dataset split.
    """
    if data_dir is None:
        data_dir = "data/pascal_context"

    if set_name not in ["train", "val", "test"]:
        raise ValueError("❌ Invalid set_name, choose 'train', 'val' or 'test'")

    # 🛠️ Create the Pascal Context dataset using the torchvision
    # VOCSegmentation class

    try:
        train_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="train",
            download=True,
        )

        test_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="val",
            download=True,
        )
    except RuntimeError:
        train_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="train",
            download=False,
        )

        test_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="val",
            download=False,
        )

    if set_name == "test":
        return test_dataset

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # 💥 Split the train set into training and validation sets
    train_len = int(0.9 * len(train_dataset))
    val_len = len(train_dataset) - train_len

    train_dataset, val_dataset = random_split(
        train_dataset, [train_len, val_len]
    )

    if set_name == "train":
        return train_dataset
    elif set_name == "val":
        return val_dataset
