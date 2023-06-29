import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import count_files_recursive, get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import (
    cocostuff_164k_classes as CLASSES,
)
from gate.data.image.segmentation.coco import (
    BaseDataset,
    download_and_extract_coco_stuff164k,
)
from gate.data.transforms.segmentation_transforms import DualImageRandomCrop

logger = get_logger(__name__, set_rich=True)

DEFAULT_SPLIT = "train"
DEFAULT_IGNORE_LABEL = 255
DEFAULT_MEAN_BGR = (104.008, 116.669, 122.675)
DEFAULT_AUGMENT = True
DEFAULT_BASE_SIZE = None
DEFAULT_CROP_SIZE = 321  # 513
DEFAULT_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
DEFAULT_FLIP = True
DEFAULT_WARP_IMAGE = True


class COCOStuff164K(BaseDataset):
    """COCO-Stuff 164K dataset 📚"""

    def __init__(
        self,
        root: str,
        split: str = DEFAULT_SPLIT,
        download: bool = False,
    ):
        """
        Initialize the CocoStuff164K dataset class. 🚀

        Args:
            root: The root path of the dataset (default: DEFAULT_ROOT).
            split: The dataset split, either "train" or "val"
            (default: DEFAULT_SPLIT["TRAIN"]).
            ignore_label: The label to ignore during training
            (default: DEFAULT_IGNORE_LABEL).

        """
        root = pathlib.Path(root)
        dataset_root = root / "coco_164k"
        logger.info(f"Loading COCO-Stuff 164K dataset from {root}...")
        if download:
            logger.info(
                f"Count of files: {count_files_recursive(dataset_root)}"
            )
            if count_files_recursive(dataset_root) == 246577:
                logger.info("Dataset already downloaded. Skipping download.")
            else:
                logger.info("Downloading and/or extracting dataset...")
                download_and_extract_coco_stuff164k(dataset_root)

        super(COCOStuff164K, self).__init__(
            root=dataset_root,
            split=split + "2017",
        )

    def _setup_dataset_files(self):
        """
        Set the list of files for the dataset split. 🔍
        """
        if self.split in ["train2017", "val2017"]:
            file_list = sorted((self.root / self.split).glob("*.jpg"))
            assert len(file_list) > 0, f"{self.root / self.split} has no image"
            file_list = [f.name.replace(".jpg", "") for f in file_list]
            self.files = file_list
        else:
            raise ValueError(f"Invalid split name: {self.split}")

    def _load_data(self, index):
        """
        Load an image and its corresponding label based on the index. 🖼️
        """
        # Set paths
        image_id = self.files[index]
        image_path = self.root / self.split / f"{image_id}.jpg"
        label_path = self.root / self.split / f"{image_id}.png"

        # Load an image and label
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        return image_id, image, label


def build_dataset(
    split: Optional[str],
    data_dir: str,
    ignore_label: int = 255,
    augment: bool = True,
    base_size: Optional[int] = None,
    scales: list = [0.5, 0.75, 1.0, 1.25, 1.5],
    flip: bool = True,
    warp_image: bool = True,
    download: bool = False,
) -> Tuple[COCOStuff164K, COCOStuff164K, COCOStuff164K]:
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
        raise KeyError(f"Invalid split name: {split}")

    train_data = COCOStuff164K(
        root=data_dir,
        split="train",
        ignore_label=ignore_label,
        download=download,
    )

    # 💥 Split the train set into training and validation sets
    train_len = int(0.9 * len(train_data))
    val_len = len(train_data) - train_len

    train_data, val_data = random_split(train_data, [train_len, val_len])

    test_data = COCOStuff164K(
        root=data_dir,
        split="val",
        ignore_label=ignore_label,
        download=download,
    )

    data_dict = {"train": train_data, "val": val_data, "test": test_data}

    return data_dict[split]


class DatasetTransforms:
    def __init__(
        self,
        input_size: Union[int, List[int]],
        target_size: Union[int, List[int]],
        initial_size: Union[int, List[int]] = 1024,
        crop_size: Optional[Union[int, List[int]]] = None,
    ):
        self.initial_size = (
            initial_size
            if isinstance(initial_size, tuple)
            or isinstance(initial_size, list)
            else (initial_size, initial_size)
        )
        self.input_size = (
            input_size
            if isinstance(input_size, tuple) or isinstance(input_size, list)
            else (input_size, input_size)
        )
        self.target_size = (
            target_size
            if isinstance(target_size, tuple) or isinstance(target_size, list)
            else (target_size, target_size)
        )
        if crop_size is not None:
            self.crop_size = (
                crop_size
                if isinstance(crop_size, list) or isinstance(crop_size, tuple)
                else [crop_size, crop_size]
            )
            self.crop_transform = DualImageRandomCrop(self.crop_size)
        else:
            self.crop_size = None

    def __call__(self, inputs: Dict):
        image = inputs["image"]
        image = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(image)

        annotation = inputs["labels"]
        annotation = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(annotation)

        if self.crop_size is not None:
            image, annotation = self.crop_transform(image, annotation)

        image = T.Resize(
            (self.input_size[0], self.input_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(image)

        annotation = T.Resize(
            (self.target_size[0], self.target_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(annotation)

        annotation = np.array(annotation)
        annotation = torch.from_numpy(annotation).unsqueeze(0)
        annotation = annotation.permute(2, 0, 1)

        return {
            "image": image,
            "labels": annotation.long(),
        }


@configurable(
    group="dataset", name="coco_10k", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=len(CLASSES),
    image_size=512,
    target_image_size=256,
) -> dict:
    train_transforms = DatasetTransforms(
        image_size, target_image_size, initial_size=1024, crop_size=512
    )
    eval_transforms = DatasetTransforms(
        image_size, target_image_size, initial_size=1024, crop_size=None
    )
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[train_transforms, transforms],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
