from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from zstandard import train_dictionary

from datasets import load_dataset
from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask

logger = get_logger(name=__name__, set_rich=True)


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

    train_set = load_dataset(
        path="dali-does/clevr-math",
        split="train",
        subset="general",
        cache_dir=data_dir,
        task="image-classification",
    )

    validation_set = load_dataset(
        path="dali-does/clevr-math",
        split="validation",
        subset="general",
        cache_dir=data_dir,
        task="image-classification",
    )
    test_set = load_dataset(
        path="dali-does/clevr-math",
        split="testing",
        subset="general",
        cache_dir=data_dir,
        task="image-classification",
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
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"pixel_values": "image"},
        transforms=transforms,
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
