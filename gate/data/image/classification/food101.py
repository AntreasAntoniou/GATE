# food101.py
from typing import Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask


@configurable
def build_food101_dataset(
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
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="food101",
        split="train",
        cache_dir=data_dir,
        task="image-classification",
    )

    test_set = load_dataset(
        path="food101",
        split="validation",
        cache_dir=data_dir,
        task="image-classification",
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    gate_dataset_config = GATEDataset.__config__(populate_full_signature=True)

    train_set = gate_dataset_config(
        dataset=train_set,
        infinite_sampling=True,
        task=ClassificationTask(),
        key_remapper_dict={{"pixel_values": "image"}},
        transforms=None,
    )

    val_set = gate_dataset_config(
        dataset=val_set,
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={{"pixel_values": "image"}},
        transforms=None,
    )

    test_set = gate_dataset_config(
        dataset=test_set,
        infinite_sampling=True,
        task=ClassificationTask(),
        key_remapper_dict={{"pixel_values": "image"}},
        transforms=None,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict
