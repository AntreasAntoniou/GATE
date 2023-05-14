import os

import pytest

from gate.data.video.classification.build_gulp_sparsesample_skeleton import (
    build_gulp_skeleton_dataset,
)


def test_build_hmdb51_skeleton_dataset():
    # Test if the function returns the correct dataset split

    for split_num in range(1, 4):
        datasets = build_gulp_skeleton_dataset(
            dataset_name="hmdb-51",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["train", "test"],
            split_num=split_num,
        )
        train_set = datasets["train"]
        test_set = datasets["test"]
        assert train_set is not None, "Train set should not be None"
        assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_gulp_skeleton_dataset(
            dataset_name="hmdb-51",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


if __name__ == "__main__":
    test_build_hmdb51_skeleton_dataset()
