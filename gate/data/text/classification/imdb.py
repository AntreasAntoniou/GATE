# imdb.py
from typing import Dict, Optional

import numpy as np
from datasets import load_dataset


def build_imdb_dataset(data_dir: str, set_name: str) -> Dict:
    """
    Build an IMDB dataset using the Hugging Face datasets library.

    :param data_dir: The directory where the dataset cache is stored.
    :type data_dir: str
    :param set_name: The name of the dataset split to return ("train", "val", or "test").
    :type set_name: str
    :return: A dictionary containing the dataset split.
    :rtype: dict
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="imdb",
        split="train",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="imdb",
        split="test",
        cache_dir=data_dir,
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]
