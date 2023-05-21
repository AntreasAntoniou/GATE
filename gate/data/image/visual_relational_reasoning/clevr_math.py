from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from datasets import load_dataset
import torch
from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask
from gate.data.transforms.tiny_image_transforms import pad_image

logger = get_logger(name=__name__, set_rich=True)


def transform_wrapper(inputs: Dict, target_size=224):
    # print(list(inputs.keys()))
    print(inputs["label"])
    return {
        "image": pad_image(inputs["image"], target_size=target_size),
        "text": inputs["question"],
        "labels": int(inputs["label"]),
    }


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a CLEVR Math dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    logger.info(
        f"Loading CLEVR Math dataset, will download to {data_dir} if necessary."
    )

    if set_name not in ["train", "val", "test"]:
        raise KeyError(f"Invalid set name {set_name}.")

    train_set = load_dataset(
        path="dali-does/clevr-math",
        split="train",
        cache_dir=data_dir,
    )

    validation_set = load_dataset(
        path="dali-does/clevr-math",
        split="validation",
        cache_dir=data_dir,
    )
    test_set = load_dataset(
        path="dali-does/clevr-math",
        split="test",
        cache_dir=data_dir,
    )

    dataset_dict = {
        "train": train_set,
        "val": validation_set,
        "test": test_set,
    }

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="clevr_math", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=10,
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=[transform_wrapper, transforms],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=[transform_wrapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=[transform_wrapper, transforms],
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
