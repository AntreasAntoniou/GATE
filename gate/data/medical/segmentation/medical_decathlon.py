import logging
import multiprocessing as mp
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
import torchvision.transforms as T
from datasets import concatenate_datasets
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import enrichen_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import (
    medical_decathlon_labels as CLASSES_DICT,
)
from gate.data.transforms.segmentation import (
    DualImageRandomCrop,
    MedicalImageSegmentationTransforms,
    PhotometricParams,
)

# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ignore all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


logger = logging.getLogger(__name__)
monai_logger = logging.getLogger("monai")
monai_logger = enrichen_logger(monai_logger)
monai_logger.setLevel(logging.CRITICAL)


class TaskOptions(Enum):
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
    import monai

    logger.info(
        f"Loading Medical Decathlon {task_name} dataset, will download to {data_dir} if necessary."
    )

    train_dataset = monai.apps.DecathlonDataset(
        root_dir=data_dir,
        task=task_name,
        section="training",
        download=True,
        seed=0,
        val_frac=0.0,
        num_workers=mp.cpu_count(),
        progress=True,
        cache_num=0,
        cache_rate=1.0,
        copy_cache=False,
        as_contiguous=True,
        runtime_cache=False,
    )

    # create a random 90-10 train-val split

    dataset_length = len(train_dataset)
    val_split = 0.2  # Fraction for the validation set (e.g., 10%)
    test_split = 0.2
    # Calculate the number of samples for train and validation sets
    val_test_length = int(dataset_length * (val_split + test_split))
    val_length = int(dataset_length * val_split)
    test_length = val_test_length - val_length

    train_length = dataset_length - val_test_length

    # Split the dataset into train and validation sets using the generator
    train_data, val_data, test_data = random_split(
        train_dataset, [train_length, val_length, test_length]
    )

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


def convert_to_b3hw(image):
    image_shape_len = len(image.shape)

    if image_shape_len == 4 and image.shape[1] == 4:  # Case of (b, 4, h, w)
        # calculate the average contribution of alpha channel
        alpha_contrib = image[:, 3:4, :, :] / 3.0
        rgb = image[:, :3, :, :]
        # Add alpha contribution to RGB channels, and clip values to be in [0,1]
        image = rgb + alpha_contrib
    elif image_shape_len == 4 and image.shape[1] == 1:  # Case of (b, 1, h, w)
        # Simply repeat the single channel 3 times
        image = image.expand(-1, 3, -1, -1)
    elif image_shape_len == 3:  # Case of (b, h, w)
        # Expand dimensions before repeating the single channel 3 times
        image = image.unsqueeze(1).expand(-1, 3, -1, -1)
    else:  # Case of (b, 3, h, w). Leave as is if already in the desired format
        raise ValueError(
            f"Invalid image shape {image.shape}, should be one of (b, 4, h, w), (b, 1, h, w), (b, h, w), (b, 3, h, w)"
        )

    return image


class DatasetTransforms:
    def __init__(
        self,
        input_size: Union[int, List[int]],
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
        image = item["image"]
        annotation = item["label"]

        if len(image.shape) == 4:
            image = image.permute(2, 3, 0, 1)
            annotation = annotation.permute(2, 0, 1)
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(1)
            annotation = annotation.permute(2, 0, 1)

        logger.debug(f"input shapes {image.shape}, {annotation.shape}")

        image = torch.tensor(image)
        annotation = torch.tensor(annotation)

        image = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )(image)

        annotation = T.Resize(
            (self.initial_size[0], self.initial_size[1]),
            interpolation=T.InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )(annotation)

        annotation = annotation.unsqueeze(1)

        logger.debug(f"pre crop shapes {image.shape}, {annotation.shape}")

        if self.crop_size is not None:
            image, annotation = self.crop_transform(image, annotation)

        logger.debug(f"post crop shapes {image.shape}, {annotation.shape}")

        if self.med_transforms is not None:
            image, annotation = self.med_transforms(image, annotation)

        logger.debug(f"post med shapes {image.shape}, {annotation.shape}")

        image = T.Resize(
            (self.input_size[0], self.input_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )(image)

        annotation = T.Resize(
            (self.label_size[0], self.label_size[1]),
            interpolation=T.InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )(annotation)

        logger.debug(f"post resize shapes {image.shape}, {annotation.shape}")

        image = convert_to_b3hw(image)
        image = patient_normalization(image)
        annotation = annotation.long()

        logger.debug(
            f"unique annotation values {torch.unique(annotation)}, frequency {torch.bincount(annotation.flatten())}",
        )

        logger.debug(f"post norm shapes {image.shape}, {annotation.shape}")

        return {
            "image": image,
            "labels": annotation,
        }


def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    task_name: str = "task01braintumour",
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    train_transforms = DatasetTransforms(
        input_size=image_size,
        label_size=label_image_size,
        initial_size=train_initial_size,
        crop_size=image_size,
    )
    eval_transforms = DatasetTransforms(
        input_size=image_size,
        label_size=label_image_size,
        initial_size=eval_initial_size,
        crop_size=image_size,
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
    num_classes: int = len(CLASSES_DICT[TaskOptions.BrainTumour.value]),
    task_name=TaskOptions.BrainTumour.value,
    image_size: int = 256,
    label_image_size: int = 256,
    train_initial_size: int = 320,
    eval_initial_size: int = 256,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Heart.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_heart(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Heart.value]),
    task_name=TaskOptions.Heart.value,
    image_size: int = 320,
    label_image_size: int = 256,
    train_initial_size: int = 384,
    eval_initial_size: int = 320,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Liver.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_liver(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Liver.value]),
    task_name=TaskOptions.Liver.value,
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Hippocampus.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_hippocampus(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Hippocampus.value]),
    task_name=TaskOptions.Hippocampus.value,
    image_size: int = 256,
    label_image_size: int = 256,
    train_initial_size: int = 320,
    eval_initial_size: int = 256,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Prostate.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_prostate(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Prostate.value]),
    task_name=TaskOptions.Prostate.value,
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Lung.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_lung(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Lung.value]),
    task_name=TaskOptions.Lung.value,
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Pancreas.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_pancreas(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Pancreas.value]),
    task_name=TaskOptions.Pancreas.value,
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.HepaticVessel.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_hepatic_vessel(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.HepaticVessel.value]),
    task_name=TaskOptions.HepaticVessel.value,
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Spleen.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_spleen(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Spleen.value]),
    task_name=TaskOptions.Spleen.value,
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )


@configurable(
    group="dataset",
    name=DatasetName.Colon.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_md_colon(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: int = len(CLASSES_DICT[TaskOptions.Colon.value]),
    task_name=TaskOptions.Colon.value,
    image_size: int = 512,
    label_image_size: int = 256,
    train_initial_size: int = 640,
    eval_initial_size: int = 512,
    ignore_index=0,
) -> dict:
    return build_gate_dataset(
        data_dir=data_dir,
        transforms=transforms,
        task_name=task_name,
        image_size=image_size,
        label_image_size=label_image_size,
        train_initial_size=train_initial_size,
        eval_initial_size=eval_initial_size,
    )
