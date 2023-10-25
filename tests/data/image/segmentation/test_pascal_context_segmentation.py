# test_food101_dataset.py
import os
import pathlib

import numpy as np
import pytest
import torchvision.transforms as T
import torchvision.transforms.functional as F
from rich import print

from gate.boilerplate.wandb_utils import visualize_volume
from gate.data.image.segmentation.pascal_context import (
    PascalContextDataset,
    build_dataset,
    build_gate_dataset,
)

DATA_DIR = pathlib.Path(os.environ.get("PYTEST_DIR"))


def test_build_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_dataset("train", data_dir=DATA_DIR)
    assert train_set is not None, "Train set should not be None"

    val_set = build_dataset("val", data_dir=DATA_DIR)
    assert val_set is not None, "Validation set should not be None"

    test_set = build_dataset("test", data_dir=DATA_DIR)
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_dataset("invalid_set_name", data_dir=DATA_DIR)


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=DATA_DIR)
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"
    print(gate_dataset["train"][0])

    from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     gate_dataset["train"], batch_size=128, num_workers=1
    # )
    # for item in train_dataloader:
    #     print(list(item.keys()))
    #     assert item["image"] is not None, "Image should not be None"
    #     assert item["labels"] is not None, "Label should not be None"
    #     unique_labels = item["labels"].unique()
    #     print(f"len {len(unique_labels)}")
    #     print(unique_labels)
    #     break

    dataset = PascalContextDataset(
        root_dir=DATA_DIR,
        subset="train",
        transform=[
            T.Resize(
                (512, 512), interpolation=F.InterpolationMode.NEAREST_EXACT
            ),
            lambda x: np.array(x),
        ],
    )
    train_dataloader = DataLoader(
        dataset, batch_size=1024, num_workers=1, shuffle=True
    )

    for item in train_dataloader:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        unique_labels = item["labels"].unique()
        print(f"len {len(unique_labels)}")
        print(unique_labels)
        break


def test_build_gate_visualize_dataset():
    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item, name="training-pascal-context")
        break

    for item in gate_dataset["val"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item, name="validation-pascal-context")
        break

    for item in gate_dataset["test"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item, name="test-nyu-pascal-context")
        break
