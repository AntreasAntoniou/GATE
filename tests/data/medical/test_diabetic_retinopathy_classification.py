# test_food101.py
import os

from gate.data.medical.classification.diabetic_retinopathy import (
    build_dataset, build_gate_dataset)


def test_build_dataset():
    # Test if the function returns the correct dataset split

    main_dataset = build_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert main_dataset["train"] is not None, "Train set should not be None"


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        break
