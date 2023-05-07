import os

import pytest

from gate.data.video.classification.build_kinetics_400 import build_kinetics_400_dataset


def test_build_kinetics_400_dataset():
    # Test if the function returns the correct dataset split

    datasets = build_kinetics_400_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["train", "val", "test"],
    )
    train_set = datasets["train"]
    val_set = datasets["val"]
    test_set = datasets["test"]
    assert train_set is not None, "Train set should not be None"
    assert val_set is not None, "Val set should not be None"
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_kinetics_400_dataset(
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


if __name__ == "__main__":
    test_build_kinetics_400_dataset()
