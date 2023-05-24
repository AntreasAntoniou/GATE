# food101.py
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from datasets import load_dataset
import torch

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask

logger = get_logger(name=__name__, set_rich=True)

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
        use_auth_token=os.environ["HF_TOKEN"],
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


@configurable(
    group="dataset", name="winoground", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[dataset_format_transform, transforms],
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
