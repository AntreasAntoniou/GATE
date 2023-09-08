# hellaswag.py
from typing import Any, Optional

from datasets import load_dataset

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.tasks.common_sense import HellaSwagTask


def build_hellaswag_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a hellaswag dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/hellaswag

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    train_data = load_dataset(
        path="hellaswag",
        split="train",
        cache_dir=data_dir,
    )

    val_data = load_dataset(
        path="hellaswag",
        split="validation",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="hellaswag",
        split="test",
        cache_dir=data_dir,
    )

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="hellaswag", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_hellaswag_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_hellaswag_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        task=HellaSwagTask(),
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_hellaswag_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        task=HellaSwagTask(),
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_hellaswag_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        task=HellaSwagTask(),
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


# For debugging and testing purposes
if __name__ == "__main__":
    print("BEFORE TRANSFORMING THE DATASET")
    train_data = build_hellaswag_dataset("train")
    print(train_data[0])
    print("GATE DATASET")
    data = build_gate_hellaswag_dataset()
    print(data["train"][0])
    print(data["val"][0])
    print(data["test"][0])
