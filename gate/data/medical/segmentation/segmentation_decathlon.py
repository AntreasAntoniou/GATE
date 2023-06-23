import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import random_split
import torchvision.transforms as T
from monai.apps import DecathlonDataset
import monai.transforms as mT

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import CustomConcatDataset, GATEDataset
from gate.data.tasks.classification import ClassificationTask

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

transform = T.Compose(
    [
        mT.LoadImaged(keys=["image", "label"]),
        mT.EnsureChannelFirstd(keys=["image", "label"]),
        mT.ScaleIntensityd(keys="image"),
        mT.ToTensord(keys=["image", "label"]),
    ]
)


def build_combined_dataset(set_name, dataset_root):
    dataset_list = []

    for task_name in task_list:
        cur_dataset = DecathlonDataset(
            dataset_root,
            task=task_name,
            section=set_name,
            transform=transform,
            download=True,
            seed=42,
            val_frac=0.0,
            num_workers=mp.cpu_count(),
            progress=True,
            copy_cache=True,
            as_contiguous=True,
            runtime_cache=False,
        )
        dataset_list.append(cur_dataset)

    dataset = CustomConcatDataset(dataset_list)
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

    train_set = build_combined_dataset("training", data_dir)

    # create a random 90-10 train-val split

    dataset_length = len(train_set)
    val_split = 0.1  # Fraction for the validation set (e.g., 10%)

    # Calculate the number of samples for train and validation sets
    val_length = int(dataset_length * val_split)
    train_length = dataset_length - val_length

    train_set, val_set = random_split(train_set, [train_length, val_length])

    test_set = build_combined_dataset("test", data_dir)

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
        task=ClassificationTask(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=dataset_dict["val"],
        infinite_sampling=False,
        task=ClassificationTask(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=dataset_dict["test"],
        infinite_sampling=False,
        task=ClassificationTask(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
