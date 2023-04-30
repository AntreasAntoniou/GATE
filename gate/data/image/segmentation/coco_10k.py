import pathlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import random_split

from gate.data.image.segmentation.coco import (
    BaseDataset,
    download_and_extract_coco_stuff10k,
)

DEFAULT_SPLIT = "train"
DEFAULT_IGNORE_LABEL = 255
DEFAULT_MEAN_BGR = (104.008, 116.669, 122.675)
DEFAULT_AUGMENT = True
DEFAULT_BASE_SIZE = None
DEFAULT_CROP_SIZE = 321
DEFAULT_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
DEFAULT_FLIP = True
DEFAULT_WARP_IMAGE = True


class COCOStuff10K(BaseDataset):
    """COCO-Stuff 10k dataset"""

    def __init__(
        self,
        root: str,
        split: str = DEFAULT_SPLIT,
        ignore_label: int = DEFAULT_IGNORE_LABEL,
        mean_bgr: Tuple[float, float, float] = DEFAULT_MEAN_BGR,
        augment: bool = DEFAULT_AUGMENT,
        base_size: Optional[int] = DEFAULT_BASE_SIZE,
        crop_size: int = DEFAULT_CROP_SIZE,
        scales: List[float] = DEFAULT_SCALES,
        flip: bool = DEFAULT_FLIP,
        warp_image: bool = DEFAULT_WARP_IMAGE,
        download: bool = False,
    ):
        """
        Initialize the CocoStuff10k dataset class.

        Args:
            root: The root path of the dataset (default: DEFAULT_ROOT).
            split: The dataset split, either "train" or "val"
            (default: DEFAULT_SPLIT["TRAIN"]).
            ignore_label: The label to ignore during training
            (default: DEFAULT_IGNORE_LABEL).
            mean_bgr: The mean BGR values to subtract from the images
            (default: DEFAULT_MEAN_BGR).
            augment: Whether to apply data augmentation (default: True).
            base_size: The base size for scaling (default: DEFAULT_SIZE["BASE"]).
            crop_size: The size of the cropped image (default: DEFAULT_SIZE["TRAIN"]).
            scales: The list of scales to use for augmentation (default: DEFAULT_SCALES).
            flip: Whether to apply horizontal flipping for data augmentation (default: True).
            warp_image: Whether to warp the image for reproducing the official scores on GitHub (default: True).
        """
        self.warp_image = warp_image
        root = pathlib.Path(root)

        if download:
            if pathlib.Path(root / "cocostuff-10k-v1.1.zip").exists():
                print("Dataset already downloaded. Skipping download.")
            else:
                download_and_extract_coco_stuff10k(root)

        super(COCOStuff10K, self).__init__(
            root=root,
            split=split,
            ignore_label=ignore_label,
            mean_bgr=mean_bgr,
            augment=augment,
            base_size=base_size,
            crop_size=crop_size,
            scales=scales,
            flip=flip,
        )

    def _set_files(self):
        """
        Create a file path/image id list based on the dataset split.
        """
        # Create data list via {train, test, all}.txt
        if self.split in ["train", "test", "all"]:
            file_list = self.root / "imageLists" / f"{self.split}.txt"
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError(f"Invalid split name: {self.split}")

    def _load_data(self, index):
        """
        Load the image and label at the given index.

        Args:
            index: The index of the image and label to load.

        Returns:
            A tuple containing the image ID, image, and label.
        """
        # Set paths
        image_id = self.files[index]
        image_path = self.root / "images" / f"{image_id}.jpg"
        label_path = self.root / "annotations" / f"{image_id}.mat"

        # Load an image and label
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR).astype(
            np.float32
        )
        label = sio.loadmat(str(label_path))["S"]
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = self.ignore_label

        # Warping: this is just for reproducing the official scores on GitHub
        if self.warp_image:
            image = cv2.resize(
                image,
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_LINEAR,
            )
            label = Image.fromarray(label).resize(
                (self.crop_size, self.crop_size), resample=Image.NEAREST
            )
            label = np.asarray(label)
        return image_id, image, label


def build_cocostuff10k_dataset(
    data_dir: str,
    split: Optional[str] = None,
    ignore_label: int = 255,
    mean_bgr: Tuple[float, float, float] = (104.008, 116.669, 122.675),
    augment: bool = True,
    base_size: Optional[int] = None,
    scales: list = [0.5, 0.75, 1.0, 1.25, 1.5],
    flip: bool = True,
    warp_image: bool = True,
    download: bool = False,
) -> Tuple[COCOStuff10K, COCOStuff10K, COCOStuff10K]:
    """
    Build a CocoStuff10k dataset using the custom CocoStuff10k class.

    Args:
        root: The root directory where the dataset is stored.
        split: The name of the dataset split to return ("train", "val", or "test").
        ignore_label: The value of the label to be ignored.
        mean_bgr: The mean BGR values.
        augment: Whether to use data augmentation.
        base_size: The base size of the images.
        crop_size: The crop size of the images.
        scales: The list of scales for data augmentation.
        flip: Whether to use horizontal flip for data augmentation.
        warp_image: Whether to warp the image.

    Returns:
        A tuple containing train, val, and test datasets as CocoStuff10k objects.
    """

    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split name: {split}")

    train_data = COCOStuff10K(
        root=data_dir,
        split="train",
        ignore_label=ignore_label,
        mean_bgr=mean_bgr,
        augment=augment,
        base_size=base_size,
        crop_size=321,
        scales=scales,
        flip=flip,
        warp_image=warp_image,
        download=download,
    )

    # 💥 Split the train set into training and validation sets
    train_len = int(0.9 * len(train_data))
    val_len = len(train_data) - train_len

    train_data, val_data = random_split(train_data, [train_len, val_len])

    test_data = COCOStuff10K(
        root=data_dir,
        split="test",
        ignore_label=ignore_label,
        mean_bgr=mean_bgr,
        augment=False,
        base_size=None,
        crop_size=513,
        scales=scales,
        flip=False,
        warp_image=warp_image,
        download=download,
    )

    data_dict = {"train": train_data, "val": val_data, "test": test_data}

    return data_dict[split]
