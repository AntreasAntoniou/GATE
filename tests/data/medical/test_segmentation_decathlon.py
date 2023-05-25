import os

import pytest

from gate.data.medical.segmentation.segmentation_decathlon import (
    build_dataset,
    build_gate_dataset,
)


def test_build_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert train_set is not None, "Train set should not be None"

    val_set = build_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert val_set is not None, "Validation set should not be None"

    test_set = build_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert test_set is not None, "Test set should not be None"


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["label"] is not None, "Label should not be None"
        break
