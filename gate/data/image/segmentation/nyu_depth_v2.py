# cityscapes.py
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import (
    nyu_depth_v2_classes as CLASSES,
)
from gate.data.transforms.segmentation_transforms import DualImageRandomCrop


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a NYU Depth V2 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    train_val_data = load_dataset(
        path="sayakpaul/nyu_depth_v2",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_data = load_dataset(
        path="sayakpaul/nyu_depth_v2",
        split="validation",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_data = train_val_data["train"]
    val_data = train_val_data["test"]

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


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
        print(list(inputs.keys()))
        annotation = inputs["depth_map"]
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
