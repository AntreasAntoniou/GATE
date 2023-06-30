import multiprocessing as mp
from dataclasses import dataclass
from math import floor
from typing import Any, Dict, List, Optional, Union

import numpy as np
import datasets
import torch
from datasets import concatenate_datasets
from torch.utils.data import random_split
import torchvision.transforms as T

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.transforms.segmentation_transforms import DualImageRandomCrop
from gate.data.image.segmentation.classes import (
    medical_decathlon_labels as CLASSES,
)

logger = get_logger(name=__name__)


@dataclass
class TaskOptions:
    BrainTumour: str = "Task01BrainTumour".lower()
    Heart: str = "Task02Heart".lower()
    Liver: str = "Task03Liver".lower()
    Hippocampus: str = "Task04Hippocampus".lower()
    Prostate: str = "Task05Prostate".lower()
    Lung: str = "Task06Lung".lower()
    Pancreas: str = "Task07Pancreas".lower()
    HepaticVessel: str = "Task08HepaticVessel".lower()
    Spleen: str = "Task09Spleen".lower()
    Colon: str = "Task10Colon".lower()


TASK_LIST = vars(TaskOptions()).values()


def build_combined_dataset(dataset_root):
    dataset_list = []

    dataset_dict = datasets.load_dataset(
        "GATE-engine/medical_decathlon",
        cache_dir=dataset_root,
        num_proc=mp.cpu_count(),
    )
    for task_name, task_dataset in dataset_dict.items():
        dataset_list.append(task_dataset)

    dataset = concatenate_datasets(dataset_list)
    return dataset


def build_dataset(
    set_name: str,
    data_dir: Optional[str] = None,
    task_name: str = "task01braintumour",
):
    """
    Build a DR dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    torch.manual_seed(42)

    logger.info(
        f"Loading Diabetic retinopathy dataset, will download to {data_dir} if necessary."
    )

    dataset = datasets.load_dataset(
        "GATE-engine/medical_decathlon",
        split=f"training.{task_name}",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
        keep_in_memory=False,
    )
    # create a random 90-10 train-val split

    val_split = 0.1  # Fraction for the validation set (e.g., 10%)
    test_split = 0.1  # Fraction for the test set (e.g., 10%)

    train_val_test_data = dataset.train_test_split(
        test_size=val_split + test_split
    )
    train_data = train_val_test_data["train"]
    val_test_data = train_val_test_data["test"].train_test_split(0.5)
    val_data = val_test_data["train"]
    test_data = val_test_data["test"]

    dataset_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    return dataset_dict[set_name]


def patient_normalization(input_volume):
    input_volume = (
        (input_volume - input_volume.min())
        / (input_volume.max() - input_volume.min())
        * 255.0
    )

    return input_volume


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
        image = image.permute(3, 0, 1, 2)
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
            "image": patient_normalization(image),
            "labels": annotation.long(),
        }


@configurable(
    group="dataset",
    name="medical_decathlon",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    task_name: str = "task01braintumour",
    num_classes=CLASSES,  # for build_model we must check if num_class is a Dict or an int. If it is a dict, we must use num_classes[task_name]
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
        dataset=build_dataset("train", data_dir=data_dir, task_name=task_name),
        infinite_sampling=True,
        transforms=[train_transforms, transforms],
        meta_data={
            "class_names": CLASSES[task_name],
            "num_classes": num_classes,
        },
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir, task_name=task_name),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={
            "class_names": CLASSES[task_name],
            "num_classes": num_classes,
        },
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir, task_name=task_name),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={
            "class_names": CLASSES[task_name],
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
