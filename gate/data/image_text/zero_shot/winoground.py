# food101.py
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchvision.transforms as T
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations
from gate.data.transforms.image import convert_to_rgb

logger = logging.getLogger(__name__)


HF_DATASET_PATH = "facebook/winoground"


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
        f"Loading Winoground dataset, will download to {data_dir} if necessary."
    )

    dataset = load_dataset(
        path=HF_DATASET_PATH,
        cache_dir=data_dir,
        split="test",
        token=os.environ["HF_TOKEN"],
        num_proc=mp.cpu_count(),
    )

    train_val_test_data = dataset.train_test_split(test_size=0.20)
    train_set = train_val_test_data["train"]
    val_test_set = train_val_test_data["test"].train_test_split(test_size=0.75)
    val_set = val_test_set["train"]
    test_set = val_test_set["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


def dataset_format_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Example of sample:
    #
    # {
    #     "id": 167,
    #     'image_0': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1280 at 0x7F4A00404250>,
    #     'image_1': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1291 at 0x7F4A00407340>,
    #     "caption_0": "an adult is flexing beside a child",
    #     "caption_1": "a child is flexing beside an adult",
    #     "tag": "Noun",
    #     "secondary_tag": "",
    #     "num_main_preds": 1,
    #     "collapsed_tag": "Object",
    # }

    input_dict = {}
    input_dict["image"] = [sample["image_0"], sample["image_1"]]
    input_dict["text"] = [sample["caption_0"], sample["caption_1"]]
    return input_dict


TranformOrListOfTransforms = Union[Callable, List[Callable]]


class WinogroundTransformAdapter:
    def __init__(
        self,
        image_transforms: Optional[TranformOrListOfTransforms] = None,
        text_transforms: Optional[TranformOrListOfTransforms] = None,
    ):
        super().__init__()
        self.image_transforms = image_transforms
        self.text_transforms = text_transforms

    def __call__(self, input_dict):
        input_dict["image"][0] = convert_to_rgb(input_dict["image"][0])
        input_dict["image"][1] = convert_to_rgb(input_dict["image"][1])

        if self.image_transforms is not None:
            for transform in self.image_transforms:
                input_dict["image"][0] = transform(
                    {"image": input_dict["image"][0]}
                )["image"]
                input_dict["image"][1] = transform(
                    {"image": input_dict["image"][1]}
                )["image"]

        if self.text_transforms is not None:
            for transform in self.text_transforms:
                input_dict["text"] = transform({"text": input_dict["text"]})[
                    "text"
                ]

        input_dict["image"] = torch.stack(input_dict["image"])

        return input_dict


@configurable(
    group="dataset", name="winoground", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    aug_transforms = StandardAugmentations("image")

    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            dataset_format_transform,
            WinogroundTransformAdapter(
                image_transforms=[aug_transforms, transforms],
                text_transforms=[transforms],
            ),
        ],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            dataset_format_transform,
            WinogroundTransformAdapter(
                image_transforms=[transforms],
                text_transforms=[transforms],
            ),
        ],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            dataset_format_transform,
            WinogroundTransformAdapter(
                image_transforms=[transforms],
                text_transforms=[transforms],
            ),
        ],
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
