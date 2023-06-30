from typing import Any, Dict, List, Optional, Union
import multiprocessing as mp

import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import datasets
import torch
from torch.utils.data import random_split
import torchvision.transforms as T

import datasets

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import acdc_labels as CLASSES
from gate.data.transforms.segmentation_transforms import DualImageRandomCrop


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> Dataset:
    """
    Build an ACDC dataset.

    Args:
        set_name: The name of the dataset split to return ("train", "val", or "test").
        data_dir: The directory where the dataset cache is stored.

    Returns:
        A Dataset object containing the specified dataset split.
    """
    # Create a generator with the specified seed
    rng = torch.Generator().manual_seed(42)
    train_data = datasets.load_dataset(
        path="GATE-engine/automated_cardiac_diagnosis_competition.ACDC",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_data = datasets.load_dataset(
        path="GATE-engine/automated_cardiac_diagnosis_competition.ACDC",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    dataset_length = len(train_data)
    val_split = 0.1  # Fraction for the validation set (e.g., 10%)

    # Calculate the number of samples for train and validation sets
    val_length = int(dataset_length * val_split)
    train_length = dataset_length - val_length

    # Split the dataset into train and validation sets using the generator
    train_data, val_data = random_split(
        train_data, [train_length, val_length], generator=rng
    )

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

    def __call__(self, item: Dict):
        print(list(item.keys()))
        image = (
            torch.stack([torch.tensor(i) for i in item["image"]])
            if isinstance(item["image"], list)
            else item["image"]
        )
        annotation = (
            torch.stack([torch.tensor(i) for i in item["label"]])
            if isinstance(item["label"], list)
            else item["label"]
        )
        image = image.permute(0, 3, 1, 2)
        annotation = annotation.permute(0, 3, 1, 2)

        image = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(image)

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

        return {
            "image": image,
            "labels": annotation.long(),
        }


@configurable(
    group="dataset",
    name="acdc",
    defaults=dict(data_dir=DATASET_DIR),
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
        image_size, target_image_size, initial_size=512, crop_size=None
    )
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[train_transforms, transforms],
        meta_data={
            "class_names": CLASSES,
            "num_classes": num_classes,
        },
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={
            "class_names": CLASSES,
            "num_classes": num_classes,
        },
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={
            "class_names": CLASSES,
            "num_classes": num_classes,
        },
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


if __name__ == "__main__":
    dataset_dict = build_gate_dataset()

    for item in dataset_dict["train"]:
        print(item["labels"])
        break
