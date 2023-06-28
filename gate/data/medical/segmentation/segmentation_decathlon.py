from math import floor
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
import datasets
from datasets import concatenate_datasets

logger = get_logger(name=__name__)


@dataclass
class TaskOptions:
    BrainTumour: str = "Task01_BrainTumour"
    Heart: str = "Task02_Heart"
    Liver: str = "Task03_Liver"
    Hippocampus: str = "Task04_Hippocampus"
    Prostate: str = "Task05_Prostate"
    Lung: str = "Task06_Lung"
    Pancreas: str = "Task07_Pancreas"
    HepaticVessel: str = "Task08_HepaticVessel"
    Spleen: str = "Task09_Spleen"
    Colon: str = "Task10_Colon"


task_list = vars(TaskOptions()).values()


def build_combined_dataset(dataset_root):
    dataset_list = []

    dataset_dict = datasets.load_dataset(
        "GATE-engine/medical_decathlon",
        cache_dir=dataset_root,
        num_proc=mp.cpu_count(),
    )
    for task_name, task_dataset in dataset_dict.items():
        dataset_list.append(task_dataset)

    dataset = concatenate_datasets(dataset_list)
    return dataset


def build_dataset(
    data_dir: Optional[str] = None,
) -> dict:
    """
    Build a DR dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    torch.manual_seed(42)

    logger.info(
        f"Loading Diabetic retinopathy dataset, will download to {data_dir} if necessary."
    )

    train_set = build_combined_dataset(data_dir)

    # create a random 90-10 train-val split

    dataset_length = len(train_set)
    val_split = 0.1  # Fraction for the validation set (e.g., 10%)
    test_split = 0.1  # Fraction for the test set (e.g., 10%)

    # Calculate the number of samples for train, validation and test sets
    train_length = dataset_length - dataset_length * (val_split + test_split)
    train_length = int(floor(train_length))
    val_length = int(floor(dataset_length * val_split))
    test_length = dataset_length - train_length - val_length

    train_set, val_set, test_set = random_split(
        train_set, [train_length, val_length, test_length]
    )

    dataset_dict = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    return dataset_dict


@configurable(
    group="dataset",
    name="decathlon_dataset",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=4,
) -> dict:
    dataset_dict = build_dataset(data_dir=data_dir)
    train_set = GATEDataset(
        dataset=dataset_dict["train"],
        infinite_sampling=True,
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=dataset_dict["val"],
        infinite_sampling=False,
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=dataset_dict["test"],
        infinite_sampling=False,
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
