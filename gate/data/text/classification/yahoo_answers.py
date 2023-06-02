# yahoo_answers.py
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import numpy as np
from datasets import load_dataset
from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.text_classification import YahooAnswersTask

def dataset_format_transform(sample: Dict) -> Dict:
    input_dict = {}
    input_dict["text"] = sample["question_title"] + sample['question_content']
    input_dict["labels"] = sample["topic"]
    return input_dict

class_description = {
    0: 'Society & Culture',
    1: 'Science & Mathematics',
    2: 'Health',
    3: 'Education & Reference',
    4: 'Computers & Internet',
    5: 'Sports',
    6: 'Business & Finance',
    7: 'Entertainment & Music',
    8: 'Family & Relationships',
    9: 'Politics & Government'
}

def build_yahoo_answers_dataset(
    data_dir: str, set_name: str
) -> dict:
    """
    Build a Yahoo Answers dataset using the Hugging Face datasets library.

    :param data_dir: The directory where the dataset cache is stored.
    :type data_dir: str
    :param set_name: The name of the dataset split to return ("train", "val", or "test").
    :type set_name: str
    :return: A dictionary containing the dataset split.
    :rtype: dict
    """
    train_val_data = load_dataset(
        path="yahoo_answers_topics",
        split="train",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="yahoo_answers_topics",
        split="test",
        cache_dir=data_dir,
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_set = train_val_data["train"]
    val_set = train_val_data["test"]

    dataset_dict = {"train": train_set, "val": val_set, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="yahoo_answers", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_yahoo_answers_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=10
) -> dict:
    train_set = GATEDataset(
        dataset=build_yahoo_answers_dataset(data_dir, "train"),
        infinite_sampling=True,
        transforms=[dataset_format_transform, transforms],
    )

    val_set = GATEDataset(
        dataset=build_yahoo_answers_dataset(data_dir, "val"),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    test_set = GATEDataset(
        dataset=build_yahoo_answers_dataset(data_dir, "test"),
        infinite_sampling=False,
        transforms=[dataset_format_transform, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict

@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 32
    eval_batch_size: int = 128
    num_classes: int = 10


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_yahoo_answers_dataset(os.environ["DATASET_DIR"], "train")
    print(train_data[0])
    print("GATE DATASET")
    data = build_gate_yahoo_answers_dataset(os.environ["DATASET_DIR"])
    print(data["train"][0])
    print(data["val"][0])
    print(data["test"][0])
