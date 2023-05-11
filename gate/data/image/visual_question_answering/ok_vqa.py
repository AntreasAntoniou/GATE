# food101.py
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from datasets import load_dataset
from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset

from gate.data.tasks.visual_question_answering import OKVQATask


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a OK VQA dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="Multimodal-Fatima/OK-VQA_train",
        cache_dir=data_dir,
    )["train"]

    test_data = load_dataset(
        path="Multimodal-Fatima/OK-VQA_test",
        cache_dir=data_dir,
    )["test"]

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="cifar100", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
):
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=OKVQATask(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=OKVQATask(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=OKVQATask(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_dataset(transforms: Optional[Any] = None):
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
