import os

import pytest
from torch.utils.data import DataLoader

from gate.data.video.classification.build_gulp_sparsesample import (
    build_gulp_dataset,
    build_squeezed_gulp_dataset,
)


def test_build_hmdb51_dataset():
    # Test if the function returns the correct dataset split

    for split_num in range(1, 4):
        datasets = build_gulp_dataset(
            dataset_name="hmdb51-gulprgb",
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
        datasets = build_gulp_dataset(
            dataset_name="hmdb51-gulprgb",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_build_hmdb51_squeezed_dataset():
    # Test if the function returns the correct dataset split

    for data_format in ["BTCHW", "BCTHW"]:
        for split_num in range(1, 4):
            datasets = build_squeezed_gulp_dataset(
                dataset_name="hmdb51-gulprgb",
                data_dir=os.environ.get("PYTEST_DIR"),
                sets_to_include=["train", "test"],
                split_num=split_num,
                data_format=data_format,
            )
            train_set = datasets["train"]
            test_set = datasets["test"]
            assert train_set is not None, "Train set should not be None"
            assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_squeezed_gulp_dataset(
            dataset_name="hmdb51-gulprgb",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_hmdb51_dataloader():
    datasets = build_gulp_dataset(
        dataset_name="hmdb51-gulprgb",
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["test"],
    )
    test_set = datasets["test"]

    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    for batch in test_loader:
        assert batch["pixel_values"].shape == (2, 3, 8, 224, 224)
        assert batch["labels"].shape == (2,)
        assert batch["video_ids"].shape == (2,)
        break


if __name__ == "__main__":
    test_build_hmdb51_dataset()
    test_build_hmdb51_squeezed_dataset()
    test_hmdb51_dataloader()
