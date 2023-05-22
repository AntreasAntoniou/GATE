# multi_fc.py
from typing import Optional

import numpy as np
from datasets import load_dataset


def build_multi_fc_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a multi_fc dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/multi_fc

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_data = load_dataset(
        path="pszemraj/multi_fc",
        split="train",
        cache_dir=data_dir,
    )

    val_data = load_dataset(
        path="pszemraj/multi_fc",
        split="validation",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="pszemraj/multi_fc",
        split="test",
        cache_dir=data_dir,
    )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]
