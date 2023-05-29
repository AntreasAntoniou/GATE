# ade20k.py
from typing import Any, Dict, Optional

import numpy as np
from datasets import load_dataset
import multiprocessing as mp

import torch

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


import torchvision.transforms as T


def transform_wrapper(inputs: Dict, target_size=224):
    print(inputs)
    # return {
    #     "image": T.Resize(size=(target_size, target_size))(
    #         inputs["image"].convert("RGB")
    #     ),
    #     "text": inputs["question"],
    #     "labels": torch.tensor(int(inputs["label"])).long(),
    #     "answer_type": inputs["template"],
    #     "question_family_idx": len(inputs["template"]) * [0],
    # }


@configurable(
    group="dataset", name="ade20k", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=11,
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[transform_wrapper, transforms],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


if __name__ == "__main__":
    dataset_dict = build_gate_dataset()

    for item in dataset_dict["train"]:
        print(item)
        break
