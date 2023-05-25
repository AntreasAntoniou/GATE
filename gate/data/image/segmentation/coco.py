import random
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
from torch.utils import data


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
    print(f"Extracting {zip_path} to {data_path}")
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
        extracted_dir = data_path / (
            "images" if "images" in filename else "annotations"
        )
        extracted_dir.mkdir(parents=True, exist_ok=True)

        download_file(url, file_path)
        extract_zip(file_path, extracted_dir)


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: str = "train",
        ignore_label: int = 255,
        mean_bgr: tuple = (104.008, 116.669, 122.675),
        augment: bool = True,
        base_size: Optional[int] = None,
        crop_size: int = 321,
        scales: list = [0.5, 0.75, 1.0, 1.25, 1.5],
        flip: bool = True,
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
        self.mean_bgr = mean_bgr
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self._setup_dataset_files()

        cv2.setNumThreads(0)

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

    def _augmentation(self, image, label):
        """
        Apply data augmentation to the image and label.

        Args:
            image: The input image in numpy.ndarray format.
            label: The input label in numpy.ndarray format.

        Returns:
            The augmented image and label.
        """
        # 🌟 Data augmentation steps
        # 1️⃣ Scaling
        # 2️⃣ Padding
        # 3️⃣ Cropping
        # 4️⃣ Flipping

        # Scaling
        h, w = label.shape
        if self.base_size:
            if h > w:
                h, w = (self.base_size, int(self.base_size * w / h))
            else:
                h, w = (int(self.base_size * h / w), self.base_size)
        scale_factor = random.choice(self.scales)
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(
                image, value=self.mean_bgr, **pad_kwargs
            )
            label = cv2.copyMakeBorder(
                label, value=self.ignore_label, **pad_kwargs
            )

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index: The index of the item to fetch.

        Returns:
            A tuple containing the image ID, image, and label.
        """
        image_id, image, label = self._load_data(index)
        if self.augment:
            image, label = self._augmentation(image, label)
        # Mean subtraction
        image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return dict(
            id=image_id,
            image=image.astype(np.float32),
            labels=label.astype(np.int64),
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
