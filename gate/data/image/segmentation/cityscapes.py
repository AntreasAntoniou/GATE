import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import load_dataset
from torchvision.datasets import Cityscapes

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.label_remap import remap_tensor_values
from gate.data.transforms.segmentation import (
    BaseDatasetTransforms,
    KeySelectorTransforms,
)

CLASSES = Cityscapes.classes


@dataclass
class CityscapeAnnotation:
    name: str
    id: int
    train_id: int
    category: str
    category_id: int
    has_instances: bool
    ignore_in_eval: bool
    color: tuple


CLASSES = [
    CityscapeAnnotation(
        name=item[0],
        id=item[1],
        train_id=item[2],
        category=item[3],
        category_id=item[4],
        has_instances=item[5],
        ignore_in_eval=item[6],
        color=item[7],
    )
    for item in CLASSES
]


class_set = set([item.train_id for item in CLASSES])
class_set = sorted(list(class_set))
class_set = class_set[1:-1]
class_set.extend([255, -1])
local_id_to_original_id = {i: item for i, item in enumerate(class_set)}
original_id_to_local_id = {item: i for i, item in enumerate(class_set)}
original_id_to_name = {item.train_id: item.name for item in CLASSES}
local_id_to_name = {
    i: original_id_to_name[local_id_to_original_id[i]]
    for i in range(len(local_id_to_original_id))
}
train_id_to_eval_id = {item.id: item.train_id for item in CLASSES}


def to_one_hot(label, num_classes):
    return torch.nn.functional.one_hot(label, num_classes=num_classes)


def remap_train_labels(input_dict: dict[str, torch.Tensor]):
    labels = input_dict["labels"]
    labels = remap_tensor_values(labels, train_id_to_eval_id)
    input_dict["labels"] = labels
    return input_dict


def remap_duds(input_dict: dict[str, torch.Tensor]):
    labels = input_dict["labels"]
    labels[labels == 255] = original_id_to_local_id[255]
    labels[labels == -1] = original_id_to_local_id[255]
    input_dict["labels"] = labels
    return input_dict


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a Cityscapes dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """

    data = load_dataset(
        "Antreas/Cityscapes",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_set = data["train"].train_test_split(test_size=0.1, seed=42)
    train_set = train_val_set["train"]
    val_set = train_val_set["test"]

    dataset_dict = {
        "train": train_set,
        "val": val_set,
        "test": data["val"],
    }

    return dataset_dict[set_name]


def tuple_to_dict(t: dict) -> dict:
    return t


@configurable(
    group="dataset", name="cityscapes", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=len(class_set) - 1,
    image_size=1024,
    target_image_size=256,
    ignore_index=original_id_to_local_id[255],
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
        infinite_sampling=False,
        transforms=[
            tuple_to_dict,
            input_transforms,
            train_transforms,
            # transforms,
            remap_train_labels,
            remap_duds,
        ],
        meta_data={"class_names": class_set, "num_classes": num_classes},
    )

    val_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            tuple_to_dict,
            input_transforms,
            eval_transforms,
            # transforms,
            remap_train_labels,
            remap_duds,
        ],
        meta_data={"class_names": class_set, "num_classes": num_classes},
    )

    test_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            tuple_to_dict,
            input_transforms,
            eval_transforms,
            # transforms,
            remap_train_labels,
            remap_duds,
        ],
        meta_data={"class_names": class_set, "num_classes": num_classes},
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
