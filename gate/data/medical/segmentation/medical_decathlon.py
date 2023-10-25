import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import torchvision.transforms as T
from datasets import concatenate_datasets

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import (
    medical_decathlon_labels as CLASSES_DICT,
)
from gate.data.transforms.segmentation_transforms import (
    DualImageRandomCrop,
    MedicalImageSegmentationTransforms,
    PhotometricParams,
)

logger = get_logger(name=__name__)


class TaskOptions(Enum):
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


class DatasetName(Enum):
    BrainTumour: str = "medical_decathlon_brain_tumour"
    Heart: str = "medical_decathlon_heart"
    Liver: str = "medical_decathlon_liver"
    Hippocampus: str = "medical_decathlon_hippocampus"
    Prostate: str = "medical_decathlon_prostate"
    Lung: str = "medical_decathlon_lung"
    Pancreas: str = "medical_decathlon_pancreas"
    HepaticVessel: str = "medical_decathlon_hepatic_vessel"
    Spleen: str = "medical_decathlon_spleen"
    Colon: str = "medical_decathlon_colon"


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
        f"Loading Medical Decathlon {task_name} dataset, will download to {data_dir} if necessary."
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
    input_volume = (input_volume - input_volume.min()) / (
        input_volume.max() - input_volume.min()
    )

    return input_volume


class DatasetTransforms:
    def __init__(
        self,
        input_size: Union[int, List[int]],
        target_size: Union[int, List[int]],
        initial_size: Union[int, List[int]] = 1024,
        label_size: Union[int, List[int]] = 256,
        crop_size: Optional[Union[int, List[int]]] = None,
        photometric_config: Optional[PhotometricParams] = None,
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

        self.label_size = (
            label_size
            if isinstance(label_size, tuple) or isinstance(label_size, list)
            else (label_size, label_size)
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

        if photometric_config is not None:
            self.med_transforms = MedicalImageSegmentationTransforms(
                photometric_params=photometric_config
            )
        else:
            self.med_transforms = None

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
        image = image.permute(3, 0, 1, 2)[:, :3]
        annotation = annotation.permute(0, 3, 1, 2)[0]

        image = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )(image)

        annotation = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )(annotation)

        image_list = []
        annotation_list = []
        image = image.reshape(-1, image.shape[-2], image.shape[-1])
        annotation = annotation.reshape(
            -1, annotation.shape[-2], annotation.shape[-1]
        )

        for image_item, annotation_item in zip(image, annotation):
            image_item = image_item.unsqueeze(0)
            annotation_item = annotation_item.unsqueeze(0)

            if self.crop_size is not None:
                image_item, annotation_item = self.crop_transform(
                    image_item, annotation_item
                )

            if self.med_transforms is not None:
                image_item = self.med_transforms(image_item, annotation_item)

            image_item = T.Resize(
                (self.input_size[0], self.input_size[1]),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            )(image_item)

            annotation_item = T.Resize(
                (self.label_size[0], self.label_size[1]),
                interpolation=T.InterpolationMode.NEAREST_EXACT,
                antialias=True,
            )(annotation_item)

            image_list.append(image_item)
            annotation_list.append(annotation_item)

        image = torch.stack(image_list)
        annotation = torch.stack(annotation_list)

        image = patient_normalization(image).unsqueeze(2)
        annotation = annotation.long()
        # by this point the shapes are num_scan, slices, channels, height, width, and each patient has about two scans
        # we choose to consider the two scans as one image, so we concatenate them along the slices dimension
        image = image.reshape(
            -1, image.shape[2], image.shape[3], image.shape[4]
        )
        annotation = annotation.reshape(
            -1, annotation.shape[2], annotation.shape[3]
        )

        # convert 1 channel to 3 channels
        image = torch.cat((image, image, image), dim=1)

        return {
            "image": image,
            "labels": annotation,
        }


def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    task_name: str = "task01braintumour",
    image_size: int = 512,
    target_image_size: int = 256,
) -> dict:
    train_transforms = DatasetTransforms(
        512,
        target_image_size,
        initial_size=1024,
        crop_size=image_size,
        photometric_config=None,
    )
    eval_transforms = DatasetTransforms(
        512, target_image_size, initial_size=512, crop_size=image_size
    )
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir, task_name=task_name),
        infinite_sampling=True,
        transforms=[train_transforms, transforms],
        meta_data={
            "class_names": CLASSES_DICT[task_name],
            "num_classes": len(CLASSES_DICT[task_name]),
        },
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir, task_name=task_name),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={
            "class_names": CLASSES_DICT[task_name],
            "num_classes": len(CLASSES_DICT[task_name]),
        },
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir, task_name=task_name),
        infinite_sampling=False,
        transforms=[eval_transforms, transforms],
        meta_data={
            "class_names": CLASSES_DICT[task_name],
            "num_classes": len(CLASSES_DICT[task_name]),
        },
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


@configurable(
    group="dataset",
    name=DatasetName.BrainTumour.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_brain_tumour(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.BrainTumour.value]),
    task_name=TaskOptions.BrainTumour.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.BrainTumour.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Heart.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_brain_tumour(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Heart.value]),
    task_name=TaskOptions.Heart.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Heart.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Liver.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_liver(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Liver.value]),
    task_name=TaskOptions.Liver.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Liver.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Hippocampus.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_hippocampus(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Hippocampus.value]),
    task_name=TaskOptions.Hippocampus.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Hippocampus.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Prostate.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_prostate(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Prostate.value]),
    task_name=TaskOptions.Prostate.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Prostate.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Lung.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_lung(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Lung.value]),
    task_name=TaskOptions.Lung.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Lung.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Pancreas.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_pancreas(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Pancreas.value]),
    task_name=TaskOptions.Pancreas.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Pancreas.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.HepaticVessel.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_hepatic_vessel(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.HepaticVessel.value]),
    task_name=TaskOptions.HepaticVessel.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.HepaticVessel.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Spleen.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_spleen(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Spleen.value]),
    task_name=TaskOptions.Spleen.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Spleen.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Colon.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_colon(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    image_size: int = 512,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Colon.value]),
    task_name=TaskOptions.Colon.value,
    target_image_size: int = 256,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=TaskOptions.Colon.value,
        image_size=image_size,
        target_image_size=target_image_size,
    )
