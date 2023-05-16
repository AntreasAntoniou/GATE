import os

import pytest
from torch.utils.data import DataLoader

from gate.data.video.skeleton.build_gulp_sparsesample_skeleton import (
    build_gulp_skeleton_dataset,
)


def test_build_ucf101_skeleton_dataset():
    # Test if the function returns the correct dataset split

    for split_num in range(1, 4):
        datasets = build_gulp_skeleton_dataset(
            dataset_name="ucf-101",
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
        datasets = build_gulp_skeleton_dataset(
            dataset_name="ucf-101",
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_ucf101_skeleton_dataloader():
    datasets = build_gulp_skeleton_dataset(
        dataset_name="ucf-101",
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["test"],
    )
    test_set = datasets["test"]

    batch_size = 2
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    for batch in test_loader:
        assert batch["pixel_values"].shape == (batch_size, 3, 8, 224, 224)
        assert batch["labels"].shape == (batch_size,)
        assert batch["video_ids"].shape == (batch_size,)
        assert batch["skeleton_num_persons"].shape == (batch_size,)
        assert batch["skeleton_num_frames"].shape == (batch_size,)
        assert batch["skeleton_keypoints"].shape == (batch_size, 30, 1775, 17, 2)
        assert batch["skeleton_keypoints_scores"].shape == (batch_size, 30, 1775, 17)
        break


if __name__ == "__main__":
    test_build_ucf101_skeleton_dataset()
    test_ucf101_skeleton_dataloader()
