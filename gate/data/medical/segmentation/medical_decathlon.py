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
    BrainTumour: str = "Task01_BrainTumour"
    Heart: str = "Task02_Heart"
    Liver: str = "Task03_Liver"
    Hippocampus: str = "Task04_Hippocampus"
    Prostate: str = "Task05_Prostate"
    Lung: str = "Task06_Lung"
    Pancreas: str = "Task07_Pancreas"
    HepaticVessel: str = "Task08_HepaticVessel"
    Spleen: str = "Task09_Spleen"
    Colon: str = "Task10_Colon"


task_list = vars(TaskOptions()).values()


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
        split=task_name,
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
        keep_in_memory=True,
    )

    # create a random 90-10 train-val split

    dataset_length = len(dataset)
    val_split = 0.1  # Fraction for the validation set (e.g., 10%)
    test_split = 0.1  # Fraction for the test set (e.g., 10%)

    # Calculate the number of samples for train, validation and test sets
    train_length = dataset_length - dataset_length * (val_split + test_split)
    train_length = int(floor(train_length))
    val_length = int(floor(dataset_length * val_split))
    test_length = dataset_length - train_length - val_length

    train_set, val_set, test_set = random_split(
        dataset, [train_length, val_length, test_length]
    )

    dataset_dict = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

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
        annotation = annotation.permute(2, 0, 1)[0].unsqueeze(0)

        return {
            "image": image,
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
    num_classes=150,
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
            "class_names": CLASSES[task_name],
            "num_classes": num_classes,
        },
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={
            "class_names": CLASSES[task_name],
            "num_classes": num_classes,
        },
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
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
