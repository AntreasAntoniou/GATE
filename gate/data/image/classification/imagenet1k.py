# imagenet1k.py
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask


def build_imagenet1k_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a SVHN dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        "imagenet-1k",
        split="train",
        cache_dir=data_dir,
        task="image-classification",
    )

    test_data = load_dataset(
        "imagenet-1k",
        split="validation",
        cache_dir=data_dir,
        task="image-classification",
    )

    train_val_data = train_val_data.train_test_split(test_size=0.05)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="imagenet1k", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_imagenet1k_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=1000,
) -> dict:
    train_set = GATEDataset(
        dataset=build_imagenet1k_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_imagenet1k_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_imagenet1k_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_imagenet1k_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
