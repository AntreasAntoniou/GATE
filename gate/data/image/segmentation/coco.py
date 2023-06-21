import random
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data

from gate.boilerplate.utils import get_logger

logger = get_logger(__name__)


def download_file(url: str, destination: Path) -> None:
    """
    Downloads a file from the given URL to the specified destination.

    :param url: The URL of the file to download.
    :param destination: The path where the file will be saved.
    """
    if not destination.is_file():
        subprocess.run(["wget", "-nc", "-P", str(destination.parent), url])


def extract_zip(zip_path: Path, destination: Path) -> None:
    """
    Extracts the contents of a zip file to the specified destination.

    :param zip_path: The path of the zip file to extract.
    :param destination: The path where the contents of the zip file will be extracted.
    """
    print(
        f"Extracting {zip_path} to {destination}, {any(destination.glob('*'))}, {destination.glob('*')}"
    )

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(destination)


def download_and_extract_coco_stuff10k(data_dir: str) -> None:
    """
    Downloads and extracts the COCO-Stuff 10k dataset to the specified directory.

    :param data_dir: The directory where the dataset will be stored.
    """
    dataset_url = "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip"
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path / "cocostuff-10k-v1.1.zip"

    if not zip_path.exists():
        download_file(dataset_url, zip_path)
    logger.info(f"Extracting {zip_path} to {data_path}")
    extract_zip(zip_path, data_path)


def download_and_extract_coco_stuff164k(data_dir: str) -> None:
    """
    Downloads and extracts the COCO-Stuff 164k dataset to the specified directory.

    :param data_dir: The directory where the dataset will be stored.
    """
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
    ]
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        file_path = data_path / filename
        extracted_dir = data_path
        extracted_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {url} to {file_path}")
        download_file(url, file_path)
        logger.info(f"Extracting {file_path} to {extracted_dir}")
        extract_zip(file_path, extracted_dir)


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: str = "train",
        ignore_label: int = 255,
    ):
        """
        Initialize the base dataset class.

        Args:
            root: The root path of the dataset.
            split: The dataset split, either "train" or "val" (default: "train").
            ignore_label: The label to ignore during training (default: 255).
            mean_bgr: The mean BGR values to subtract from the images (default: (104.008, 116.669, 122.675)).
            augment: Whether to apply data augmentation (default: True).
            base_size: The base size for scaling (default: None).
            crop_size: The size of the cropped image (default: 321).
            scales: The list of scales to use for augmentation (default: [0.5, 0.75, 1.0, 1.25, 1.5]).
            flip: Whether to apply horizontal flipping for data augmentation (default: True).
        """
        self.root = root
        self.split = split
        self.ignore_label = ignore_label
        self.files = []
        self._setup_dataset_files()

        # cv2.setNumThreads(0)

    def _setup_dataset_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index: The index of the item to fetch.

        Returns:
            A tuple containing the image ID, image, and label.
        """
        image_id, image, label = self._load_data(index)
        return dict(
            id=image_id,
            image=image,
            labels=label,
        )

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            The number of items in the dataset.
        """
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
