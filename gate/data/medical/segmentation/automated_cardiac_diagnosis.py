import multiprocessing as mp
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import acdc_labels as CLASSES
from gate.data.medical.segmentation.medical_decathlon import (
    patient_normalization,
)
from gate.data.transforms.segmentation import (
    DualImageRandomCrop,
    DualImageRandomFlip,
    MedicalImageSegmentationTransforms,
    PhotoMetricDistortion,
    PhotometricParams,
)


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
        item["image"] = [sample["img"] for sample in item["frame_data"]]
        item["label"] = [sample["label"] for sample in item["frame_data"]]

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
        512, target_image_size, initial_size=1024, crop_size=image_size
    )
    eval_transforms = DatasetTransforms(
        512,
        target_image_size,
        initial_size=512,
        crop_size=image_size,
        photometric_config=None,
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
