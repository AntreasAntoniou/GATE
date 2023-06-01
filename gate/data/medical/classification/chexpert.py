# chexpert.py
import multiprocessing as mp
from dataclasses import dataclass
import pathlib
from typing import Any, Dict, Optional

import numpy as np
from datasets import load_dataset
import torch

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a Chexpert dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    dataset = load_dataset(
        "alkzar90/NIH-Chest-X-ray-dataset",
        "image-classification",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_data = dataset["train"].train_test_split(test_size=0.10)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {
        "train": train_set,
        "val": val_set,
        "test": dataset["test"],
    }

    return dataset_dict[set_name]


class_descriptions = [
    "no-finding",
    "enlarged-cardiomediastinum",
    "cardiomegaly",
    "lung-opacity",
    "lung-lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural-effusion",
    "pleural-other",
    "fracture",
    "support-devices",
]


def dataset_format_transform(sample: Dict) -> Dict:
    # Example of sample:
    #

    input_dict = {}
    input_dict["image"] = sample["image"]
    input_dict["labels"] = torch.zeros(14)
    input_dict["labels"][sample["labels"]] = 1
    return input_dict


@configurable(
    group="dataset",
    name="chexpert-classification",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=14,
    label_idx_to_class_name: Optional[Dict[int, str]] = {
        i: name for i, name in enumerate(class_descriptions)
    },
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[dataset_format_transform, transforms],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
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


import torchvision.transforms as T

if __name__ == "__main__":
    data_dir = pathlib.Path("/data0/datasets/medical/chexpert")
    dataset = build_dataset(set_name="train", data_dir=data_dir)

    for item in dataset:
        image = T.ToTensor()(item["image"])

        print(image.shape)
