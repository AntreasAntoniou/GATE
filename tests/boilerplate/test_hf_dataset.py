import os
from tempfile import TemporaryDirectory

import pytest

from gate.data.image.classification.cifar100 import build_cifar100_dataset


def test_build_dataset(beans_dir):
    # Test that the function correctly loads the
    # train split of the Food101 dataset
    data = build_cifar100_dataset(data_dir=beans_dir, set_name="train")
