# test_food101.py
import os

import pytest

from gate.data.image.visual_question_answering.ok_vqa import (
    build_dataset as build_ok_vqa_dataset,
)
from gate.data.image.visual_question_answering.ok_vqa import (
    build_gate_dataset as build_ok_vqa_gate_dataset,
)
from gate.models.task_specific_models.visual_question_answering.clip import (
    build_model,
)


def test_build_ok_vqa_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_ok_vqa_dataset(
        "train", data_dir=os.environ.get("PYTEST_DIR")
    )
    assert train_set is not None, "Train set should not be None"

    val_set = build_ok_vqa_dataset(
        "val", data_dir=os.environ.get("PYTEST_DIR")
    )
    assert val_set is not None, "Validation set should not be None"

    test_set = build_ok_vqa_dataset(
        "test", data_dir=os.environ.get("PYTEST_DIR")
    )
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_ok_vqa_dataset("invalid_set_name")


def test_build_ok_vqa_gate_dataset():
    # Test if the function returns the correct dataset split
    vqa_model = build_model()

    train_set = build_ok_vqa_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"), transforms=vqa_model.transform
    )
    assert train_set is not None, "Train set should not be None"
