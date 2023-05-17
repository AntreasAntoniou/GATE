from dataclasses import dataclass
import os
import pathlib
from typing import Any, Callable, Optional
import torch

from kaggle.api.kaggle_api_extended import KaggleApi
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image
import numpy as np

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import count_files_recursive, get_logger
from gate.config.variables import DATASET_DIR
from gate.data import download_kaggle_dataset
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask

logger = get_logger(name=__name__, set_rich=True)

FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 70236


# TODO: Test the below code


class HAM10KClassification(Dataset):
    def __init__(
        self, dataset_path: pathlib.Path, transform: Optional[Callable] = None
    ):
        """https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

        Args:
            dataset_path (pathlib.Path): _description_
            transform (Optional[Callable], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.dataset_path = dataset_path
        dataset_path_dict = self.download_and_extract(dataset_path)
        self.labels_frame = pd.read_csv(
            dataset_path_dict["dataset_download_path"] / "trainLabels.csv"
        )
        self.img_dir = (
            dataset_path_dict["dataset_download_path"] / "resized_train"
        )
        self.transform = transform

    def download_and_extract(self, dataset_path: pathlib.Path):
        return download_kaggle_dataset(
            dataset_name="ham10k",
            dataset_path="kmader/skin-cancer-mnist-ham10000",
            target_dir_path=dataset_path,
            count_after_download_and_extract=FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT,
        )

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.img_dir, self.labels_frame.iloc[idx, 0] + ".jpeg"
        )
        image = Image.open(img_name)
        label = self.labels_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}


from torch.utils.data import random_split


def build_dataset(
    train_ratio: float = 0.8,
    val_ratio: float = 0.05,
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
    rng = np.random.RandomState(42)
    torch.manual_seed(42)

    logger.info(
        f"Loading Diabetic retinopathy dataset, will download to {data_dir} if necessary."
    )

    dataset = DiabeticRetinopathyClassification(dataset_path=data_dir)

    train_length = int(len(dataset) * train_ratio)
    val_length = int(len(dataset) * val_ratio)
    test_length = len(dataset) - train_length - val_length

    train_set, val_set, test_set = random_split(
        dataset, [train_length, val_length, test_length]
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict


@configurable(
    group="dataset",
    name="diabetic_retionopathy",
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


def build_dummy_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 4
