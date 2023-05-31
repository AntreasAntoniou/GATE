# winogrande.py
from typing import Any, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.common_sense import WinograndeTask


def build_winogrande_debiased_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a winogrande dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/winogrande

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_data = load_dataset(
        "winogrande",
        "winogrande_debiased",
        split="train",
        cache_dir=data_dir,
    )

    val_data = load_dataset(
        "winogrande",
        "winogrande_debiased",
        split="validation",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        "winogrande",
        "winogrande_debiased",
        split="test",
        cache_dir=data_dir,
    )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="winogrande", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_winogrande_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_winogrande_debiased_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=WinograndeTask(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_winogrande_debiased_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=WinograndeTask(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_winogrande_debiased_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=WinograndeTask(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_winogrande_debiased_dataset("train")
    print(train_data[0])
    print("GATE DATASET")
    data = build_gate_winogrande_dataset()
    print(data["train"][0])
    print(data["val"][0])
    print(data["test"][0])
