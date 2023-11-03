import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataset import Dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data import download_kaggle_dataset
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations

logger = logging.getLogger(__name__)

FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 78991


class HappyWhaleDolphinClassification(Dataset):
    """
    PyTorch Dataset for the Happy Whale and Dolphin Classification Task.

    Attributes:
        dataset_path (pathlib.Path): Directory path where the dataset is located.
        transform (Callable): A function/transform to apply to the images.
        labels_frame (DataFrame): Pandas DataFrame containing the image labels.
        img_dir (pathlib.Path): Directory path where the image dataset is located.
        species_to_idx (Dict): A mapping from species labels to unique integer indices.
        individual_to_idx (Dict):  A mapping from individual labels to unique integer indices.
    """

    def __init__(
        self, dataset_path: pathlib.Path, transform: Optional[Callable] = None
    ):
        """
        Constructor for the HappyWhaleDolphinClassification PyTorch Dataset.

        Args:
            dataset_path (pathlib.Path): Directory path where the dataset is located.
            transform (Optional[Callable]): A function/transform to apply to the images.
        """
        super().__init__()
        self.dataset_path = dataset_path
        dataset_path_dict = self.download_and_extract(dataset_path)

        # Pandas dataframe with labels
        self.labels_frame = pd.read_csv(
            dataset_path_dict["dataset_download_path"] / "train.csv"
        )

        # Directory path with images
        self.img_dir = (
            dataset_path_dict["dataset_download_path"] / "train_images"
        )
        self.transform = transform

        # Get unique labels for species and individuals
        species_labels = self.labels_frame.species.unique()
        individual_labels = self.labels_frame.individual_id.unique()

        # Dictionaries to map unique labels to integers
        self.species_to_idx = {
            label: i for i, label in enumerate(species_labels)
        }
        self.individual_to_idx = {
            label: i for i, label in enumerate(individual_labels)
        }

    def download_and_extract(self, dataset_path: pathlib.Path):
        return download_kaggle_dataset(
            dataset_name="happy-whale-and-dolphin-identification",
            dataset_path="happy-whale-and-dolphin",
            target_dir_path=dataset_path,
            file_count_after_download_and_extract=FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT,
            is_competition=True,
        )

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, Any]:
        """
        Get a sample from the dataset at the provided index.

        Args:
            idx (Union[int, torch.Tensor]): The index of the dataset sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the image and the integer-mapped labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        img_name = os.path.join(self.img_dir, self.labels_frame.iloc[idx, 0])

        # Load image using PIL library
        image = Image.open(img_name)

        # Obtain image labels via indexing and map them to integers
        species_label = self.species_to_idx[self.labels_frame.iloc[idx, 1]]
        individual_label = self.individual_to_idx[
            self.labels_frame.iloc[idx, 2]
        ]

        # Apply the transform function if available
        if self.transform:
            image = self.transform(image)

        # Return a dictionary containing the image and the labels
        return {
            "image": image,
            "species_label": species_label,
            "individual_label": individual_label,
        }


def build_dataset(
    train_ratio: float = 0.8,
    val_ratio: float = 0.05,
    data_dir: Optional[str] = None,
) -> dict:
    """
    Build a HWD dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    torch.manual_seed(42)

    logger.info(
        f"Loading Happy Whale and Dolphin dataset, will download to {data_dir} if necessary."
    )

    dataset = HappyWhaleDolphinClassification(dataset_path=data_dir)

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
    input_dict["individual_labels"] = sample["individual_label"]
    input_dict["species_labels"] = sample["species_label"]

    return input_dict


@configurable(
    group="dataset",
    name="diabetic_retionopathy",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=5,
    label_idx_to_class_name=None,
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
