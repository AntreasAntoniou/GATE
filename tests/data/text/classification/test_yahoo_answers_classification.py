# test_yahoo_answers_classification.py
import os

import pytest

from gate.data.text.classification.yahoo_answers import (
    build_yahoo_answers_dataset,
)


def test_build_yahoo_answers_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_yahoo_answers_dataset(
        os.environ.get("DATASET_DIR"), "train"
    )
    assert train_set is not None, "Train set should not be None"

    val_set = build_yahoo_answers_dataset(
        os.environ.get("DATASET_DIR"), "val"
    )
    assert val_set is not None, "Validation set should not be None"

    test_set = build_yahoo_answers_dataset(
        os.environ.get("DATASET_DIR"), "test"
    )
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_yahoo_answers_dataset(
            os.environ.get("DATASET_DIR"), "invalid_set_name"
        )


if __name__ == "__main__":
    test_build_yahoo_answers_dataset()
