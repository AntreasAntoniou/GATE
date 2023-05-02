import os

import pytest

from gate.data.video.classification.build_gulp_sparsesample import (
    build_gulp_dataset,
    build_squeezed_gulp_dataset,
)


def test_build_epic_kitchens_100_dataset():
    # Test if the function returns the correct dataset split

    datasets = build_gulp_dataset(
        dataset_name="epic-kitchens-100-gulprgb",
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["train", "val"],
    )
    train_set = datasets["train"]
    val_set = datasets["val"]
    assert train_set is not None, "Train set should not be None"
    assert val_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_gulp_dataset(
            dataset_name="epic-kitchens-100-gulprgb",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_build_epic_kitchens_100_squeezed_dataset():
    # Test if the function returns the correct dataset split

    for data_format in ["BTCHW", "BCTHW"]:
        datasets = build_squeezed_gulp_dataset(
            dataset_name="epic-kitchens-100-gulprgb",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["train", "val"],
            data_format=data_format,
        )
        train_set = datasets["train"]
        val_set = datasets["val"]
        assert train_set is not None, "Train set should not be None"
        assert val_set is not None, "Val set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_squeezed_gulp_dataset(
            dataset_name="epic-kitchens-100-gulprgb",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


if __name__ == "__main__":
    test_build_epic_kitchens_100_dataset()
    test_build_epic_kitchens_100_squeezed_dataset()
