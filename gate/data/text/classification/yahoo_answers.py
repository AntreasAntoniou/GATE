# yahoo_answers.py
import multiprocessing as mp
from typing import Optional

import numpy as np
from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.text_classification import YahooAnswersTask


def build_yahoo_answers_dataset(
    set_name: str, data_dir: Optional[str] = None
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
    group="dataset", name="yaho_answers", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_yahoo_answers_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_yahoo_answers_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=YahooAnswersTask(),
        key_remapper_dict={"label": "labels"},
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_yahoo_answers_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=YahooAnswersTask(),
        key_remapper_dict={"label": "labels"},
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_yahoo_answers_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=YahooAnswersTask(),
        key_remapper_dict={"label": "label"},
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_yahoo_answers_dataset("train")
    print(train_data[0])
    print("GATE DATASET")
    data = build_gate_yahoo_answers_dataset()
    print(data["train"][0])
    print(data["val"][0])
    print(data["test"][0])
