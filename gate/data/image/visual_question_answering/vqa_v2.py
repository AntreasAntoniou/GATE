# food101.py
from dataclasses import dataclass
from typing import Any, Optional
from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.visual_question_answering import VQAV2Task

import numpy as np
from datasets import load_dataset


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a VQA V2 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_val_data = load_dataset(
        path="HuggingFaceM4/VQAv2",
        split="train",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="HuggingFaceM4/VQAv2",
        split="validation",
        cache_dir=data_dir,
    )

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
        task=VQAV2Task(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=VQAV2Task(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=VQAV2Task(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_cifar100_dataset(transforms: Optional[Any] = None):
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
