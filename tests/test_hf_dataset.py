import os
from tempfile import TemporaryDirectory

import pytest

from gate.data import build_dataset


@pytest.fixture(scope="module")
def beans_dir():
    # Download and extract the Food101 dataset
    with TemporaryDirectory() as tmp_dir:
        os.system(f"python -m datasets.download.download beans {tmp_dir}")
        yield tmp_dir


def test_build_dataset(beans_dir):
    # Test that the function correctly loads the
    # train split of the Food101 dataset
    data = build_dataset("beans", data_dir=beans_dir, set_name="train")
