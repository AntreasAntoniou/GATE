# food101.py
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional

from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations

logger = logging.getLogger(__name__)


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a Flickr30K dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """

    logger.info(
        f"Loading NewYorker Caption Contest dataset, will download to {data_dir} if necessary."
    )

    dataset = load_dataset(
        path="nlphuji/flickr30k",
        cache_dir=data_dir,
        split="test",
        num_proc=mp.cpu_count(),
    )

    train_val_test_data = dataset.train_test_split(test_size=0.20)
    train_set = train_val_test_data["train"]
    val_test_set = train_val_test_data["test"].train_test_split(test_size=0.75)
    val_set = val_test_set["train"]
    test_set = val_test_set["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


import numpy as np


def dataset_format_transform(sample: Dict) -> Dict:
    # Example of sample:
    #
    # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375
    # at 0x7F4A005A6740>, 'caption': ['A young, blond boy sitting in a white
    # chair beside cactus plants is eating a sandwich.', "A little boy site
    # in a white rocking chair eating a sandwich out of a child's plate.",
    # "A young boy eats a sandwich on his family's porch.",
    #  'Blond boy eating a sandwich on a white chair.',
    # 'A young boy sits while eating.'],
    # 'sentids': ['8760', '8761', '8762', '8763', '8764'], 'split': 'train', 'img_id': '1
    # 752', 'filename': '14989976.jpg'}

    input_dict = {}
    input_dict["image"] = sample["image"]
    input_dict["text"] = np.random.choice(sample["caption"])[0]
    return input_dict


@configurable(
    group="dataset",
    name="newyorkercaptioncontest",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            dataset_format_transform,
            StandardAugmentations(image_key="image"),
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
