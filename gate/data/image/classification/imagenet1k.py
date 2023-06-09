# imagenet1k.py
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torchvision.transforms as T
from datasets import load_dataset
from timm.data import rand_augment_transform

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build an ImageNet1K dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """

    data = load_dataset(
        "imagenet-1k",
        cache_dir=data_dir,
        task="image-classification",
        num_proc=mp.cpu_count(),
    )
    train_val_data = data["train"]
    test_data = data["validation"]

    train_val_data = train_val_data.train_test_split(test_size=0.05)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]


logger = get_logger(name=__name__)


@configurable(
    group="dataset",
    name="imagenet1k-classification",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=1000,
) -> dict:
    rand_augment = rand_augment_transform(
        "rand-m9-n3-mstd0.5-inc1", hparams={}
    )
    single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))

    def train_augment(input_dict):
        x = input_dict["image"]
        x = T.ToTensor()(x)
        if x.shape[0] == 1:
            x = single_to_three_channel(x)
        x = T.ToPILImage()(x)
        try:
            input_dict["image"] = rand_augment(x)
        except Exception as e:
            logger.warn(f"RandAugment failed with error: {e}")
            input_dict["image"] = x

        return input_dict

    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            train_augment,
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            transforms,
        ],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            transforms,
        ],
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
