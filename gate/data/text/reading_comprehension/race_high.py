# race_high.py
from typing import Optional
import multiprocessing as mp

import numpy as np
from datasets import load_dataset


def build_race_high_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a race_high dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/race

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_data = load_dataset(
        "race",
        "high",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    val_data = load_dataset(
        "race",
        "high",
        split="validation",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_data = load_dataset(
        "race",
        "high",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]
