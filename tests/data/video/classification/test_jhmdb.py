import os

import pytest

from gate.data.video.classification.build_gulp_sparsesample_tube import (
    build_gulp_tube_dataset,
)


def test_build_jhmdb_tube_dataset():
    # Test if the function returns the correct dataset split

    datasets = build_gulp_tube_dataset(
        dataset_name="jhmdb",
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["train", "test"],
    )
    train_set = datasets["train"]
    test_set = datasets["test"]
    assert train_set is not None, "Train set should not be None"
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_gulp_tube_dataset(
            dataset_name="ucf-101-24",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


if __name__ == "__main__":
    test_build_jhmdb_tube_dataset()
