import os

import pytest
from torch.utils.data import DataLoader

from gate.data.video.classification.build_kinetics_400 import (
    build_kinetics_400_dataset,
)


def test_build_kinetics_400_dataset():
    # Test if the function returns the correct dataset split

    datasets = build_kinetics_400_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["train", "val", "test"],
    )
    train_set = datasets["train"]
    val_set = datasets["val"]
    test_set = datasets["test"]
    assert train_set is not None, "Train set should not be None"
    assert val_set is not None, "Val set should not be None"
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(ValueError):
        datasets = build_kinetics_400_dataset(
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_kinetics_400_dataloader():
    datasets = build_kinetics_400_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["val"],
    )
    val_set = datasets["val"]

    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

    for batch in val_loader:
        assert batch["pixel_values"].shape == (2, 3, 8, 224, 224)
        assert batch["labels"].shape == (2,)
        assert batch["video_ids"].shape == (2,)
        break


if __name__ == "__main__":
    test_build_kinetics_400_dataset()
    test_kinetics_400_dataloader()
