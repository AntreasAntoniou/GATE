# cifar100.py
# imagenet1k.py
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from timm.data import rand_augment_transform
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.transforms.image import convert_to_rgb, pad_image

logger = logging.getLogger(__name__)


class StandardAugmentations:
    def __init__(self, image_key: Optional[str] = None) -> None:
        self.image_key = image_key

    def __call__(self, input_dict: Dict) -> Any:
        rand_augment = rand_augment_transform(
            "rand-m9-n3-mstd0.5-inc1", hparams={}
        )

        if self.image_key is None:
            x = input_dict
        else:
            x = input_dict[self.image_key]

        if isinstance(x, Image.Image):
            x = convert_to_rgb(x)

        try:
            if self.image_key is None:
                input_dict = rand_augment(x)
            else:
                input_dict[self.image_key] = rand_augment(x)
        except Exception as e:
            logger.warn(f"RandAugment failed with error: {e}")

        return input_dict


class KeyMapper:
    def __call__(self, input_dict: Dict) -> Any:
        return {"image": input_dict["image"], "labels": input_dict["label"]}


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a CIFAR100 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    # Create a generator with the specified seed
    rng = torch.Generator().manual_seed(42)
    try:
        data = torchvision.datasets.CIFAR100(
            root=(
                data_dir
                if data_dir is not None
                else os.path.expanduser(
                    "~/.cache/torch/datasets/cifar100-train/"
                )
            ),
            train=True,
            download=True,
        )
    except RuntimeError:
        data = torchvision.datasets.CIFAR100(
            root=(
                data_dir
                if data_dir is not None
                else os.path.expanduser(
                    "~/.cache/torch/datasets/cifar100-train/"
                )
            ),
            train=True,
            download=False,
        )

    dataset_length = len(data)
    val_split = 0.05  # Fraction for the validation set (e.g., 10%)

    # Calculate the number of samples for train and validation sets
    val_length = int(dataset_length * val_split)
    train_length = dataset_length - val_length

    # Split the dataset into train and validation sets using the generator
    train_data, val_data = random_split(
        data, [train_length, val_length], generator=rng
    )

    try:
        test_data = torchvision.datasets.CIFAR100(
            root=(
                data_dir
                if data_dir is not None
                else os.path.expanduser(
                    "~/.cache/torch/datasets/cifar100-test/"
                )
            ),
            train=False,
            download=True,
        )
    except RuntimeError:
        test_data = torchvision.datasets.CIFAR100(
            root=(
                data_dir
                if data_dir is not None
                else os.path.expanduser(
                    "~/.cache/torch/datasets/cifar100-test/"
                )
            ),
            train=False,
            download=False,
        )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


def transform_wrapper(inputs: Tuple, target_size=224):
    return {
        "image": pad_image(inputs[0], target_size=target_size),
        "labels": inputs[1],
    }


@configurable(
    group="dataset", name="cifar100", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_cifar100_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=100,
):
    augmentations = StandardAugmentations(image_key="image")
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[transform_wrapper, augmentations, transforms],
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


def build_dummy_cifar100_dataset(transforms: Optional[Any] = None):
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 100
