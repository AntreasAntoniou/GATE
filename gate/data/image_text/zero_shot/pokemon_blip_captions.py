# food101.py
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from datasets import load_dataset
from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask

logger = get_logger(name=__name__, set_rich=True)

HF_DATASET_PATH = "lambdalabs/pokemon-blip-captions"


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a Flickr30K dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    logger.info(
        f"Loading Flickr dataset, will download to {data_dir} if necessary."
    )

    dataset = load_dataset(
        path=HF_DATASET_PATH, cache_dir=data_dir, split="test"
    )

    train_val_test_data = dataset.train_test_split(test_size=0.20)
    train_set = train_val_test_data["train"]
    val_test_set = train_val_test_data["test"].train_test_split(test_size=0.75)
    val_set = val_test_set["train"]
    test_set = val_test_set["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="flickr30k", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=101,
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


def build_dummy_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
