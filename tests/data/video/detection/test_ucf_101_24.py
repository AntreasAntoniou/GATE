import os

import pytest
from torch.utils.data import DataLoader

from gate.data.video.detection.build_gulp_sparsesample_tube import (
    build_gulp_tube_dataset,
)


def test_build_ucf101_24_tube_dataset():
    # Test if the function returns the correct dataset split

    datasets = build_gulp_tube_dataset(
        dataset_name="ucf-101-24",
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["train", "test"],
    )
    train_set = datasets["train"]
    test_set = datasets["test"]
    assert train_set is not None, "Train set should not be None"
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_gulp_tube_dataset(
            dataset_name="ucf-101-24",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_ucf101_24_tube_dataloader():
    datasets = build_gulp_tube_dataset(
        dataset_name="ucf-101-24",
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["train", "test"],
    )
    train_set = datasets["train"]
    test_set = datasets["test"]

    batch_size = 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    for batch in train_loader:
        assert batch["pixel_values"].shape == (batch_size, 3, 8, 224, 224)
        assert batch["labels"].shape == (batch_size,)
        assert batch["video_ids"].shape == (batch_size,)
        # NOTE: To access the tube metadata, use the indices to index into the tubes list
        # This is because the tube metadata has arbitrary length
        # print(train_set.tubes[batch["indices"][0].item()])
        break

    for batch in test_loader:
        assert batch["pixel_values"].shape == (batch_size, 3, 8, 224, 224)
        assert batch["labels"].shape == (batch_size,)
        assert batch["video_ids"].shape == (batch_size,)
        break


if __name__ == "__main__":
    test_build_ucf101_24_tube_dataset()
    test_ucf101_24_tube_dataloader()
