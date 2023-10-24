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
from gate.data.transforms.segmentation_transforms import (
    BaseDatasetTransforms,
    DualImageRandomCrop,
    KeySelectorTransforms,
)


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
            download=True,
        )

        test_dataset = VOCSegmentation(
            root=data_dir,
            year="2012",
            image_set="val",
            download=True,
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


@configurable(
    group="dataset", name="pascal_context", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=len(CLASSES),
    image_size=1024,
    target_image_size=256,
) -> dict:
    input_transforms = KeySelectorTransforms(
        initial_size=2048, image_label="image", label_label="annotation"
    )

    train_transforms = BaseDatasetTransforms(
        input_size=image_size,
        target_size=target_image_size,
        crop_size=image_size,
        flip_probability=0.5,
        use_photo_metric_distortion=True,
    )

    eval_transforms = BaseDatasetTransforms(
        input_size=image_size,
        target_size=target_image_size,
        crop_size=None,
        flip_probability=None,
        use_photo_metric_distortion=False,
    )

    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            lambda x: {"image": x[0], "annotation": x[1]},
            input_transforms,
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
            input_transforms,
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
            input_transforms,
            eval_transforms,
            transforms,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
    return dataset_dict
