import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data import download_kaggle_dataset
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations

logger = logging.getLogger(__name__)


FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 70236


class DiabeticRetinopathyClassification(Dataset):
    def __init__(
        self, dataset_path: pathlib.Path, transform: Optional[Callable] = None
    ):
        super().__init__()
        self.dataset_path = dataset_path
        dataset_path_dict = self.download_and_extract(dataset_path)
        self.labels_frame = pd.read_csv(
            dataset_path_dict["dataset_download_path"] / "trainLabels.csv"
        )
        self.img_dir = (
            dataset_path_dict["dataset_download_path"]
            / "resized_train"
            / "resized_train"
        )
        self.transform = transform

    def download_and_extract(self, dataset_path: pathlib.Path):
        return download_kaggle_dataset(
            dataset_name="diabetic-retinopathy",
            dataset_path="tanlikesmath/diabetic-retinopathy-resized",
            target_dir_path=dataset_path,
            file_count_after_download_and_extract=FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT,
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

    logger.info(
        f"Loading Diabetic retinopathy dataset, will download to {data_dir} if"
        " necessary."
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


def dataset_format_transform(sample: Dict) -> Dict:
    # Example of sample:
    # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x768 at 0x7FD8EDD55C60>, 'label': 0}
    #

    input_dict = {}
    input_dict["image"] = sample["image"]
    input_dict["labels"] = torch.zeros(5)
    input_dict["labels"][sample["label"]] = 1
    return input_dict


class_idx_to_descriptions = {
    0: "no-dr",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "proliferative-dr",
}


@configurable(
    group="dataset",
    name="diabetic_retionopathy",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=5,
    label_idx_to_class_name=class_idx_to_descriptions,
) -> dict:
    dataset_dict = build_dataset(data_dir=data_dir)

    train_set = GATEDataset(
        dataset=dataset_dict["train"],
        infinite_sampling=True,
        transforms=[
            dataset_format_transform,
            StandardAugmentations(image_key="image"),
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=dataset_dict["val"],
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    test_set = GATEDataset(
        dataset=dataset_dict["test"],
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
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
