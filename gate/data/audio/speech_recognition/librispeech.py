# librispeech.py
from typing import Optional

import numpy as np
from datasets import load_dataset


def build_librispeech_dataset(
    set_name: str, hours: str = "360", data_dir: Optional[str] = None
) -> dict:
    """
    Build a LibriSpeech dataset using the Hugging Face datasets library.

    :param set_name: The name of the dataset split to return ("train", "val", or "test").
    :type set_name: str
    :param hours: The number of hours to load (available subsets: "100" and "360").
    :type hours: str, optional
    :param data_dir: The directory where the dataset cache is stored.
    :type data_dir: str, optional
    :return: A dictionary containing the dataset split.
    :rtype: dict
    """
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
