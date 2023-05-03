# librispeech.py
from typing import Optional

import numpy as np
from datasets import load_dataset


def build_librispeech_dataset(
    set_name: str, hours: str = 360, data_dir: Optional[str] = None
) -> dict:
    """
    Build LibriSpeech dataset using the Hugging Face datasets library.

    Args:
        set_name: The name of the dataset split to return
        ("train", "val", or "test").
        hours: Number of hours to load (available subsets: "100" and "360").
        data_dir: The directory where the dataset cache is stored.


    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="librispeech_asr",
        name="clean",
        split=f"train.{hours}",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="librispeech_asr",
        name="clean",
        split="test",
        cache_dir=data_dir,
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]