# race_middle.py
import multiprocessing as mp
from typing import Any, Optional

from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.reading_comprehension import RACETask


def build_race_middle_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a race_middle dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/race

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    train_data = load_dataset(
        "race",
        "middle",
        split="train",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    val_data = load_dataset(
        "race",
        "middle",
        split="validation",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    test_data = load_dataset(
        "race",
        "middle",
        split="test",
        cache_dir=data_dir,
        num_proc=mp.cpu_count(),
    )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="race_middle", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_race_middle_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_race_middle_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=RACETask(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_race_middle_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=RACETask(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_race_middle_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=RACETask(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    data = build_race_middle_dataset("test")
    print(data[12])
    print("GATE DATASET")
    data = build_gate_race_middle_dataset()
    print(data["train"][12])
    print(data["val"][12])
    print(data["test"][12])
