# imdb.py
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask


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
) -> dict:
    train_set = GATEDataset(
        dataset=build_imdb_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=ClassificationTask(),
        key_remapper_dict={"label": "labels"},
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_imdb_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"label": "labels"},
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_imdb_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=ClassificationTask(),
        key_remapper_dict={"label": "label"},
        transforms=transforms,
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
    train_data = build_imdb_dataset("train")
    print(train_data[0])
    print("GATE DATASET")
    data = build_gate_imdb_dataset()
    print(data["train"][0])
    print(data["val"][0])
    print(data["test"][0])
