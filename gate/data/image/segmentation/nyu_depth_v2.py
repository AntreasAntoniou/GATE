# cityscapes.py
import multiprocessing as mp
from typing import Any, Optional

from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import \
    nyu_depth_v2_classes as CLASSES
from gate.data.transforms.segmentation import (BaseDatasetTransforms,
                                               KeySelectorTransforms)


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


@configurable(
    group="dataset", name="nyu_depth_v2", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=len(CLASSES),
    image_size=1024,
    target_image_size=256,
    ignore_index=-1,
) -> dict:
    input_transforms = KeySelectorTransforms(
        initial_size=2048, image_label="image", label_label="depth_map"
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
