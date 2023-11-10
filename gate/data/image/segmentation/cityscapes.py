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
from gate.data.transforms.segmentation import (
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
        "Chris1/cityscapes_segmentation",
        "instance_segmentation",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    dataset_dict = {
        "train": data["train"],
        "val": data["validation"],
        "test": data["test"],
    }

    return dataset_dict[set_name]


# NSD and DSC for metrics, and also ensure that the model can compute per class metrics


@configurable(
    group="dataset", name="cityscapes", defaults=dict(data_dir=DATASET_DIR)
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
        initial_size=2048,
        image_label="image",
        label_label="semantic_segmentation",
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
