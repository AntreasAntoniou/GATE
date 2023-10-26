# food101.py
import logging
import multiprocessing as mp
from ast import Dict
from dataclasses import dataclass
from typing import Any, Optional

from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations

# Removed unused import statement
# import numpy as np


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
        f"Loading Flickr dataset, will download to {data_dir} if necessary."
    )

    dataset = load_dataset(
        path="nlphuji/flickr30k",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    train_val_test_data = dataset.train_test_split(test_size=0.30)
    train_set = train_val_test_data["train"]
    val_test_set = train_val_test_data["test"].train_test_split(test_size=0.50)
    val_set = val_test_set["train"]
    test_set = val_test_set["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


import numpy as np


def dataset_format_transform(sample: Dict) -> Dict:
    # Example of sample:
    #
    # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x333 at 0x7F4A0059A2C0>,
    # 'caption': ['A man with a red shirt and black hat suspends himself in the air by sitting on
    # a pole and clinging to two wires with his hands.', 'A man sits on top of a streetlight post
    # with a picture of a thing in bones looking up to him in the focus.', 'A man in gray pants
    # and orange shorts is sitting on a pole that has cables running from it.', 'A man is sitting
    # on the top of a pole while holding on to big heavy wires.', 'A guy on a city electric pole
    # watching over.'], 'sentids': ['135295', '135296', '135297', '135298', '135299'],
    # 'split': 'train', 'img_id': '27059', 'filename': '5888131040.jpg'}

    input_dict = {}
    input_dict["image"] = sample["image"]
    input_dict["text"] = np.random.choice(sample["caption"])[0]
    return input_dict


@configurable(
    group="dataset", name="flickr30k", defaults=dict(data_dir=DATASET_DIR)
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
