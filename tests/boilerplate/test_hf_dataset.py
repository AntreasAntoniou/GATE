import os

from gate.data.image.classification.cifar100 import build_dataset


def test_build_dataset():
    # Test that the function correctly loads the
    # train split of the Food101 dataset
    data = build_dataset(data_dir=os.environ["PYTEST_DIR"], set_name="train")
