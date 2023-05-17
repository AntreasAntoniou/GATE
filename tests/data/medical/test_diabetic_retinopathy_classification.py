# test_food101.py
import os

import pytest

from gate.data.medical.classification.diabetic_retinopathy import (
    build_gate_dataset,
)


def test_build_dataset():
    # Test if the function returns the correct dataset split

    dataset_dict = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert dataset_dict is not None, "Dataset should not be None"
