# lm1b.py
import multiprocessing as mp
from typing import Any, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.language_modelling import LanguageModellingTask


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
        num_proc=mp.cpu_count(),
    )

    test_data = load_dataset(
        path="lm1b",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="lm1b", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_lm1b_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_lm1b_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=LanguageModellingTask(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_lm1b_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=LanguageModellingTask(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_lm1b_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=LanguageModellingTask(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_lm1b_dataset("train")
    print(train_data[12])
    print("GATE DATASET")
    data = build_gate_lm1b_dataset()
    print(data["train"][12])
    print(data["val"][12])
    print(data["test"][12])
