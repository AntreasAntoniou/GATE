# svhn.py
import multiprocessing as mp
from typing import Any, Dict, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations
from gate.data.transforms.tiny_image_transforms import pad_image


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
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
        "svhn",
        "cropped_digits",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_data = load_dataset(
        "svhn",
        "cropped_digits",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]


def transform_wrapper(inputs: Dict, target_size=224):
    return {
        "image": pad_image(inputs["image"], target_size=target_size),
        "labels": inputs["label"],
    }


@configurable(
    group="dataset", name="svhn", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=10,
) -> dict:
    """
    Builds and returns a gate dataset for the SVHN (Street View House Numbers) dataset.

    Args:
        data_dir (Optional[str]): The directory path where the SVHN dataset is stored. If None, defaults to the value of DATASET_DIR.
        transforms (Optional[Any]): Any additional transforms to apply to the dataset.
        num_classes (int): The number of classes in the dataset. Defaults to 10.

    Returns:
        dict: A dictionary containing the gate dataset for training, validation, and testing. The keys are "train", "val", and "test", and the values are the respective datasets.
    """
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            transform_wrapper,
            StandardAugmentations(image_key="image"),
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_stl10_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass
