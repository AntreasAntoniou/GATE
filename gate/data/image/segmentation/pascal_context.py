# pascal_context.py
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import random_split
from torchvision.datasets import VOCSegmentation

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import (
    pascal_context_classes as CLASSES,
)
from gate.data.transforms.segmentation_transforms import DualImageRandomCrop


def build_dataset(
    set_name: str, data_dir: Optional[str] = None, download: bool = False
):
    """
    Build the Pascal Context dataset.

    Args:
        set_name (str): The name of the dataset split to return
        ("train", "val" or "test").
        data_dir (Optional[str]): The directory where the dataset is stored.
        Default: None.
        download (bool): Whether to download the dataset if
        not already present. Default: False.

    Returns:
        A Dataset object containing the specified dataset split.
    """
    if data_dir is None:
        data_dir = "data/pascal_context"

    if set_name not in ["train", "val", "test"]:
        raise KeyError("❌ Invalid set_name, choose 'train', 'val' or 'test'")

    # 🛠️ Create the Pascal Context dataset using the torchvision
    # VOCSegmentation class
    data_dir = pathlib.Path(data_dir) / "pascal_context"
    try:
        train_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="train",
            download=False,
        )

        test_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="val",
            download=False,
        )
    except RuntimeError:
        train_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="train",
            download=False,
        )

        test_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="val",
            download=False,
        )

    if set_name == "test":
        return test_dataset

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # 💥 Split the train set into training and validation sets
    train_len = int(0.9 * len(train_dataset))
    val_len = len(train_dataset) - train_len

    train_dataset, val_dataset = random_split(
        train_dataset, [train_len, val_len]
    )

    if set_name == "train":
        return train_dataset
    elif set_name == "val":
        return val_dataset


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

        annotation = inputs["annotation"]
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
        annotation = torch.from_numpy(annotation)
        annotation = annotation.unsqueeze(0)

        image = T.ToTensor()(image)

        return {
            "image": image,
            "labels": annotation.long(),
        }


@configurable(
    group="dataset", name="nyu_depth_v2", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=150,
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
        transforms=[
            lambda x: {"image": x[0], "annotation": x[1]},
            train_transforms,
            transforms,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            lambda x: {"image": x[0], "annotation": x[1]},
            eval_transforms,
            transforms,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            lambda x: {"image": x[0], "annotation": x[1]},
            eval_transforms,
            transforms,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
