# food101.py
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torchvision.transforms as T
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import (
    KeyMapper,
    StandardAugmentations,
)

logger = get_logger(name=__name__, set_rich=True)


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

    logger.info(
        f"Loading Food-101 dataset, will download to {data_dir} if necessary."
    )

    train_val_data = load_dataset(
        path="food101",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_set = load_dataset(
        path="food101",
        split="validation",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="food101", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=101,
) -> dict:
    single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))
    augmentations = StandardAugmentations(image_key="image")

    def train_augment(input_dict):
        input_dict = augmentations(input_dict)

        x = input_dict["image"]
        x = T.ToTensor()(x)
        if x.shape[0] == 1:
            x = single_to_three_channel(x)
        x = T.ToPILImage()(x)
        input_dict["image"] = x

        return input_dict

    train_set = GATEDataset(
        dataset=build_food101_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            KeyMapper(),
            train_augment,
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=build_food101_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            KeyMapper(),
            transforms,
        ],
    )

    test_set = GATEDataset(
        dataset=build_food101_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            KeyMapper(),
            transforms,
        ],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_food101_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
