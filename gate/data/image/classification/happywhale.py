import logging
import multiprocessing as mp
from typing import Any, Optional

import torchvision.transforms as T
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations

logger = logging.getLogger(__name__)


def build_dataset(
    data_dir: Optional[str] = None,
) -> dict:
    """
    Build a HWD dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """

    logger.info(
        f"Loading Happy Whale and Dolphin dataset, will download to {data_dir} if necessary."
    )

    logger.info(
        f"Loading Food-101 dataset, will download to {data_dir} if necessary."
    )

    train_set = load_dataset(
        path="GATE-engine/happy-whale-dolphin-classification",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    val_set = load_dataset(
        path="GATE-engine/happy-whale-dolphin-classification",
        split="val",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_set = load_dataset(
        path="GATE-engine/happy-whale-dolphin-classification",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict


def get_label_dict(dataset):
    return {
        "species": list(dataset.species_to_idx.keys()),
        "individual": list(dataset.individual_to_idx.keys()),
    }


def key_selection_transform(input_dict: dict):
    return {
        "image": input_dict["image"],
        "labels": {
            "species": input_dict["species"],
            "individual": input_dict["individual"],
        },
    }


def initial_resize(input_dict):
    input_dict["image"] = T.Resize(
        (224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True
    )(input_dict["image"])
    return input_dict


@configurable(
    group="dataset",
    name="happy_whale_dolphin_classification",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes={"species": 30, "individual": 15587},
    label_idx_to_class_name=None,
) -> dict:
    dataset_dict = build_dataset(data_dir=data_dir)
    train_set = GATEDataset(
        dataset=dataset_dict["train"],
        infinite_sampling=True,
        transforms=[
            key_selection_transform,
            initial_resize,
            StandardAugmentations(image_key="image"),
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=dataset_dict["val"],
        infinite_sampling=False,
        transforms=[key_selection_transform, initial_resize, transforms],
    )

    test_set = GATEDataset(
        dataset=dataset_dict["test"],
        infinite_sampling=False,
        transforms=[key_selection_transform, initial_resize, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict
