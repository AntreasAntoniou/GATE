import logging
import multiprocessing as mp
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import cocostuff_164k_dict as CLASSES
from gate.data.image.segmentation.label_remap import remap_tensor_values
from gate.data.transforms.segmentation import (
    BaseDatasetTransforms,
    KeySelectorTransforms,
)

logger = logging.getLogger(__name__)


def build_dataset(
    split: Optional[str],
    data_dir: str,
    ignore_label: int = 255,
    download: bool = False,
):
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

    train_data = load_dataset(
        "GATE-engine/COCOStuff164K",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    # 💥 Split the train set into training and validation sets
    train_val_data = train_data.train_test_split(test_size=0.1)
    train_data = train_val_data["train"]
    val_data = train_val_data["test"]

    test_data = load_dataset(
        "GATE-engine/COCOStuff164K",
        split="val",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    data_dict = {"train": train_data, "val": val_data, "test": test_data}

    return data_dict[split]


def remap_train_labels(input_dict: dict[str, torch.Tensor]):
    labels = input_dict["labels"]
    labels = remap_tensor_values(labels, CLASSES)
    input_dict["labels"] = labels
    return input_dict


@configurable(
    group="dataset", name="coco_164k", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=len(CLASSES),
    image_size=1024,
    target_image_size=256,
    ignore_index=0,
) -> dict:
    input_transforms = KeySelectorTransforms(
        initial_size=2048, image_label="image", label_label="mask"
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
            input_transforms,
            train_transforms,
            # transforms,
            remap_train_labels,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            input_transforms,
            eval_transforms,
            # transforms,
            remap_train_labels,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            input_transforms,
            eval_transforms,
            # transforms,
            remap_train_labels,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
