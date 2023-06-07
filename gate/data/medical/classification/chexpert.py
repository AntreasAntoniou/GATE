# chexpert.py
import multiprocessing as mp
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import datasets
import numpy as np
import torch
from datasets import load_dataset

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
    filtered_dataset_path = (
        pathlib.Path(data_dir) / "chexpert" / "filtered_dataset"
    )

    if filtered_dataset_path.exists():
        dataset = datasets.load_from_disk(filtered_dataset_path)
    else:
        dataset = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            "image-classification",
            cache_dir=data_dir,
            num_proc=mp.cpu_count(),
        )

        def process_labels(example):
            # Keep only the labels that are in class_description_main
            example["labels"] = [
                label
                for label in example["labels"]
                if label in class_description_main.keys()
            ]
            return example

        # Apply the function in parallel
        dataset = dataset.map(process_labels, num_proc=mp.cpu_count())

        # Now filter the examples that have at least one label
        dataset = dataset.filter(
            lambda example: len(example["labels"]) > 0, num_proc=mp.cpu_count()
        )

        dataset.save_to_disk(filtered_dataset_path)

    train_val_data = dataset["train"].train_test_split(test_size=0.10)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {
        "train": train_set,
        "val": val_set,
        "test": dataset["test"],
    }

    return dataset_dict[set_name]


class_descriptions = {
    0: "no-finding",
    1: "enlarged-cardiomediastinum",
    2: "cardiomegaly",
    3: "lung-opacity",
    4: "lung-lesion",
    5: "edema",
    6: "consolidation",
    7: "pneumonia",
    8: "atelectasis",
    9: "pneumothorax",
    10: "pleural-effusion",
    11: "pleural-other",
    12: "fracture",
    13: "support-devices",
    14: "unknown",
}

class_description_main = {
    2: "cardiomegaly",
    5: "edema",
    6: "consolidation",
    8: "atelectasis",
    10: "pleural-effusion",
}

class_description_post = {
    0: "cardiomegaly",
    1: "edema",
    2: "consolidation",
    3: "atelectasis",
    4: "pleural-effusion",
}

class_map = {2: 0, 5: 1, 6: 2, 8: 3, 10: 4}


def dataset_format_transform(sample: Dict) -> Dict:
    # Example of sample:
    #

    input_dict = {}
    input_dict["image"] = sample["image"]
    sample["labels"] = [class_map[label] for label in sample["labels"]]
    input_dict["labels"] = torch.zeros(len(class_description_post))
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
    num_classes=len(class_description_post),
    label_idx_to_class_name=class_description_post,
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
