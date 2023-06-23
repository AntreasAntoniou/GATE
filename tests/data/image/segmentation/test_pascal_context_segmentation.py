import os
import pathlib
import pytest

from gate.data.image.segmentation.pascal_context import (
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
