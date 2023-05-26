# common_voice.py
from typing import Optional

import numpy as np
from datasets import load_dataset


def build_common_voice_dataset(
    set_name: str, language: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a Common Voice Corpus 12.0 dataset using the Hugging Face datasets
    library.

    :param set_name: The name of the dataset split to return
    ("train", "val", or "test").
    :type set_name: str
    :param language: The name of the language to load eg.
    ("hi", "en", "fr" etc)
    :type language: str
    :param data_dir: The directory where the dataset cache is stored.
    :type data_dir: str, optional
    :return: A dictionary containing the dataset split.
    :rtype: dict
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="mozilla-foundation/common_voice_12_0",
        name=language,
        split="train",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="mozilla-foundation/common_voice_12_0",
        name=language,
        split="test",
        cache_dir=data_dir,
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]
