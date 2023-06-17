# ade20k.py
import multiprocessing as mp
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a Food-101 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    data = load_dataset(
        "scene_parse_150",
        "instance_segmentation",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    dataset_dict = {
        "train": data["train"],
        "val": data["validation"],
        "test": data["test"],
    }

    return dataset_dict[set_name]


class DatasetTransforms:
    def __init__(self, input_size, target_size):
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

    def __call__(self, inputs: Dict):
        image = inputs["image"]
        image = T.Resize(
            (self.input_size[0], self.input_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(image)

        annotation = inputs["annotation"]
        annotation = T.Resize(
            (self.target_size[0], self.target_size[1]),
            interpolation=T.InterpolationMode.BICUBIC,
        )(annotation)
        annotation = np.array(annotation)
        annotation = torch.from_numpy(annotation)
        annotation = annotation.permute(2, 0, 1)[0].unsqueeze(0)

        return {
            "image": image,
            "labels": annotation.long(),
        }


@configurable(
    group="dataset", name="ade20k", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=150,
    image_size=1024,
    target_image_size=256,
) -> dict:
    dataset_transforms = DatasetTransforms(image_size, target_image_size)
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[dataset_transforms, transforms],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[dataset_transforms, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[dataset_transforms, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


if __name__ == "__main__":
    dataset_dict = build_gate_dataset()

    for item in dataset_dict["train"]:
        print(item["labels"])
        break
