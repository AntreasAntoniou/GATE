# test_mscoco_10k.py
import os
import pathlib

import pytest

from gate.data.image.segmentation.coco_164k import (
    build_dataset,
    build_gate_dataset,
)

DATA_DIR = pathlib.Path(os.environ.get("PYTEST_DIR"))


def test_invalid_set_name():
    with pytest.raises(KeyError):
        build_dataset(
            data_dir=DATA_DIR,
            split="invalid",
        )


# Note: This test requires internet connection and may take a while to complete
@pytest.mark.parametrize("set_name", ["train", "test"])
def test_download(set_name):
    dataset = build_dataset(
        split=set_name,
        data_dir=DATA_DIR,
        download=True,
    )
    assert len(dataset) > 0, f"{set_name} dataset should not be empty"


@pytest.mark.parametrize("set_name", ["train", "val", "test"])
def test_set_name(set_name):
    dataset = build_dataset(
        split=set_name,
        data_dir=DATA_DIR,
    )
    assert len(dataset) > 0, f"{set_name} dataset should not be empty"

    # Check if the images and annotations are properly loaded
    idx, img, ann = dataset[0]
    assert img is not None, f"{set_name} dataset should have non-empty images"
    assert (
        ann is not None
    ), f"{set_name} dataset should have non-empty annotations"


def test_build_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_dataset("train", data_dir=os.environ.get("PYTEST_DIR"))
    assert train_set is not None, "Train set should not be None"

    val_set = build_dataset("val", data_dir=os.environ.get("PYTEST_DIR"))
    assert val_set is not None, "Validation set should not be None"

    test_set = build_dataset("test", data_dir=os.environ.get("PYTEST_DIR"))
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_dataset(
            "invalid_set_name", data_dir=os.environ.get("PYTEST_DIR")
        )


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    print(gate_dataset["train"][10])
