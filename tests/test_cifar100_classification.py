# test_food101_dataset.py
import pytest
from gate.data.image.classification.cifar100 import build_cifar100_dataset


def test_build_svhn_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_cifar100_dataset("train")
    assert train_set is not None, "Train set should not be None"

    val_set = build_cifar100_dataset("val")
    assert val_set is not None, "Validation set should not be None"

    test_set = build_cifar100_dataset("test")
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_cifar100_dataset("invalid_set_name")
