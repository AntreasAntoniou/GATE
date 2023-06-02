# imdb.py
import os
from dataclasses import dataclass
import multiprocessing as mp
from typing import Any, Dict, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask
import torch

def dataset_format_transform(sample: Dict) -> Dict:
    input_dict = {}
    input_dict["text"] = sample["text"]
    input_dict["labels"] = torch.zeros(DefaultHyperparameters.num_classes)
    input_dict["labels"][sample["label"]] = 1
    return input_dict

def build_imdb_dataset(data_dir: str, set_name: str) -> Dict:
    """
    Build an IMDB dataset using the Hugging Face datasets library.

    :param data_dir: The directory where the dataset cache is stored.
    :type data_dir: str
    :param set_name: The name of the dataset split to return ("train", "val", or "test").
    :type set_name: str
    :return: A dictionary containing the dataset split.
    :rtype: dict
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="imdb",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_data = load_dataset(
        path="imdb",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="imdb", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_imdb_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=2
) -> dict:
    train_set = GATEDataset(
        dataset=build_imdb_dataset(data_dir, "train"),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    val_set = GATEDataset(
        dataset=build_imdb_dataset(data_dir, "val"),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    test_set = GATEDataset(
        dataset=build_imdb_dataset(data_dir, "test"),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 32
    eval_batch_size: int = 128
    num_classes: int = 2


# For debugging purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_imdb_dataset(os.environ["DATASET_DIR"], "train")
    print(train_data[0])
    print("GATE DATASET")
    data = build_gate_imdb_dataset(data_dir=os.environ["DATASET_DIR"])
    print(data["train"][0])
    print(data["val"][0])
    print(data["test"][0])
