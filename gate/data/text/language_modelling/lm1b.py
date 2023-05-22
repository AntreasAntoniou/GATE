# lm1b.py
from typing import Optional

import numpy as np

from datasets import load_dataset


def build_lm1b_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a lm1b dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/lm1b

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="lm1b",
        split="train",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="lm1b",
        split="test",
        cache_dir=data_dir,
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]
