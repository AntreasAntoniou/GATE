# test_imagenet1k.py
import os

import pytest
import torch

from gate.data.few_shot.omniglot import build_dataset, build_gate_dataset


def test_build_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_dataset(
        "train", data_dir=os.environ.get("PYTEST_DIR"), num_episodes=10000
    )
    assert train_set is not None, "Train set should not be None"

    val_set = build_dataset(
        "val", data_dir=os.environ.get("PYTEST_DIR"), num_episodes=600
    )
    assert val_set is not None, "Validation set should not be None"

    test_set = build_dataset(
        "test", data_dir=os.environ.get("PYTEST_DIR"), num_episodes=600
    )
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_dataset(
            "invalid_set_name",
            data_dir=os.environ.get("PYTEST_DIR"),
            num_episodes=600,
        )


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for idx, item in enumerate(gate_dataset["train"]):
        if idx == 20:
            break

    for idx, item in enumerate(gate_dataset["val"]):
        if idx == 20:
            break

    for idx, item in enumerate(gate_dataset["test"]):
        if idx == 20:
            break
