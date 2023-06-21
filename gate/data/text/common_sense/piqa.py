# piqa.py
from typing import Any, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.common_sense import PIQATask


def build_piqa_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a piqa dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/piqa

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    train_data = load_dataset(
        path="piqa",
        split="train",
        cache_dir=data_dir,
    )

    val_data = load_dataset(
        path="piqa",
        split="validation",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="piqa",
        split="test",
        cache_dir=data_dir,
    )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="piqa", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_piqa_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_piqa_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=PIQATask(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_piqa_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=PIQATask(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_piqa_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=PIQATask(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_piqa_dataset("train")
    print(train_data[0])
    print("GATE DATASET")
    data = build_gate_piqa_dataset()
    print(data["train"][0])
    print(data["val"][0])
    print(data["test"][0])
