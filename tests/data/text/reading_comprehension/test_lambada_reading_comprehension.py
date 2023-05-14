# test_lambada_reding_comprehension.py
import os

import pytest

from gate.data.text.reading_comprehension.lambada import build_lambada_dataset


def test_build_lambada_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_lambada_dataset("train", data_dir=os.environ.get("TEST_DIR"))
    assert train_set is not None, "Train set should not be None"

    val_set = build_lambada_dataset("val", data_dir=os.environ.get("TEST_DIR"))
    assert val_set is not None, "Validation set should not be None"

    test_set = build_lambada_dataset("test", data_dir=os.environ.get("TEST_DIR"))
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_lambada_dataset(
            "invalid_set_name", data_dir=os.environ.get("TEST_DIR")
        )


if __name__ == "__main__":
    test_build_lambada_dataset()