# ade20k.py
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import ade20_classes as CLASSES
from gate.data.transforms.segmentation_transforms import (
    BaseDatasetTransforms,
    KeySelectorTransforms,
)


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a Food-101 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    data = load_dataset(
        "scene_parse_150",
        "instance_segmentation",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )
    train_val_set = data["train"].train_test_split(test_size=0.1, seed=42)
    train_set = train_val_set["train"]
    val_set = train_val_set["test"]

    dataset_dict = {
        "train": train_set,
        "val": val_set,
        "test": data["validation"],
    }

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="ade20k", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=150,
    image_size=512,
    target_image_size=256,
) -> dict:
    input_transforms = KeySelectorTransforms(
        initial_size=1024, image_label="image", label_label="annotation"
    )

    train_transforms = BaseDatasetTransforms(
        input_size=image_size,
        target_size=target_image_size,
        crop_size=512,
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
        transforms=[input_transforms, train_transforms, transforms],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[input_transforms, eval_transforms, transforms],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[input_transforms, eval_transforms, transforms],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
