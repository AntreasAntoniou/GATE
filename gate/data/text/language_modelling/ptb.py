# ptb.py
from typing import Any, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.language_modelling import LanguageModellingTask


def build_ptb_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a PTB dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/ptb_text_only

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_data = load_dataset(
        path="ptb_text_only",
        split="train",
        cache_dir=data_dir,
    )

    val_data = load_dataset(
        path="ptb_text_only",
        split="validation",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="ptb_text_only",
        split="test",
        cache_dir=data_dir,
    )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


@configurable(group="dataset", name="ptb", defaults=dict(data_dir=DATASET_DIR))
def build_gate_ptb_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_ptb_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=LanguageModellingTask(),
        key_remapper_dict={"sentence": "text"},
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_ptb_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=LanguageModellingTask(),
        key_remapper_dict={"sentence": "text"},
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_ptb_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=LanguageModellingTask(),
        key_remapper_dict={"sentence": "text"},
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_ptb_dataset("train")
    print(train_data[12])
    print("GATE DATASET")
    data = build_gate_ptb_dataset()
    print(data["train"][12])
    print(data["val"][12])
    print(data["test"][12])
