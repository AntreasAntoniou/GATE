import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import random_split
import torchvision.transforms as T
from gate.boilerplate.decorators import configurable

from gate.boilerplate.utils import count_files_recursive
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.coco import (
    BaseDataset,
    download_and_extract_coco_stuff10k,
)
from gate.data.image.segmentation.classes import (
    cocostuff_10K_classes as CLASSES,
)
from gate.data.transforms.segmentation_transforms import DualImageRandomCrop

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
        """
        root = pathlib.Path(root)
        dataset_root = root / "coco_10k"

        if download:
            if count_files_recursive(dataset_root) == 20004:
                print("Dataset already downloaded. Skipping download.")
            else:
                download_and_extract_coco_stuff10k(root)

        super(COCOStuff10K, self).__init__(
            root=dataset_root,
            split=split,
        )

    def _setup_dataset_files(self):
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
            raise KeyError(f"Invalid split name: {self.split}")

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
        image = Image.open(image_path).convert("RGB")
        label = T.ToPILImage()(torch.tensor(sio.loadmat(str(label_path))["S"]))

        return image_id, image, label


def build_dataset(
    split: str,
    data_dir: str,
    ignore_label: int = 255,
    download: bool = False,
) -> Tuple[COCOStuff10K, COCOStuff10K, COCOStuff10K]:
    """
    Build a CocoStuff10k dataset using the custom CocoStuff10k class.

    Args:
        root: The root directory where the dataset is stored.
        split: The name of the dataset split to return ("train", "val", or "test").
        ignore_label: The value of the label to be ignored.


    Returns:
        A tuple containing train, val, and test datasets as CocoStuff10k objects.
    """

    if split not in ["train", "val", "test"]:
        raise KeyError(f"Invalid split name: {split}")

    train_data = COCOStuff10K(
        root=data_dir,
        split="train",
        ignore_label=ignore_label,
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


if __name__ == "__main__":
    dataset_dict = build_gate_dataset()

    for item in dataset_dict["train"]:
        print(item["labels"])
        break
