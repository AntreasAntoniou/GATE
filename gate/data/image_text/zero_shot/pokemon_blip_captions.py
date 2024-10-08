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


HF_DATASET_PATH = "lambdalabs/pokemon-blip-captions"


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
        path=HF_DATASET_PATH,
        cache_dir=data_dir,
        split="train",
        num_proc=mp.cpu_count(),
    )

    train_val_test_data = dataset.train_test_split(test_size=0.50)
    train_set = train_val_test_data["train"]
    val_test_set = train_val_test_data["test"].train_test_split(test_size=0.50)
    val_set = val_test_set["train"]
    test_set = val_test_set["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


def dataset_format_transform(sample: Dict) -> Dict:
    # Example of sample:
    #
    # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x1280 at 0x7F4A005A6CE0>,
    # 'text': 'a purple and blue dragon flying through the air'}

    input_dict = {}
    input_dict["image"] = sample["image"]
    input_dict["text"] = sample["text"]
    return input_dict


@configurable(
    group="dataset",
    name="pokemonblipcaptions",
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


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
