from dataclasses import dataclass
from typing import Any, Optional
import multiprocessing as mp
import numpy as np
import torch

from monai.apps import DecathlonDataset
from gate.boilerplate.decorators import configurable

from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
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


def build_dataset(
    data_dir: Optional[str] = None,
    task_name: str = TaskOptions.BrainTumour,
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

    train_set = DecathlonDataset(
        data_dir,
        task=task_name,
        section="training",
        transform=None,
        download=True,
        seed=42,
        val_frac=0.0,
        num_workers=mp.cpu_count(),
        progress=True,
        copy_cache=True,
        as_contiguous=True,
        runtime_cache=False,
    )

    val_set = DecathlonDataset(
        data_dir,
        task=task_name,
        section="validation",
        transform=None,
        download=True,
        seed=42,
        val_frac=0.0,
        num_workers=mp.cpu_count(),
        progress=True,
        copy_cache=True,
        as_contiguous=True,
        runtime_cache=False,
    )

    test_set = DecathlonDataset(
        data_dir,
        task=task_name,
        section="test",
        transform=None,
        download=True,
        seed=42,
        val_frac=0.0,
        num_workers=mp.cpu_count(),
        progress=True,
        copy_cache=True,
        as_contiguous=True,
        runtime_cache=False,
    )

    options = train_set.get_properties()

    dataset_dict = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "options": options,
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
    task_name: str = TaskOptions.BrainTumour,
    num_classes=4,
) -> dict:
    dataset_dict = build_dataset(task_name=task_name, data_dir=data_dir)
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
