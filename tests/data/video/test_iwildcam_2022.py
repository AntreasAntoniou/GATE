import os

import pytest
from torch.utils.data import DataLoader

from gate.data.video.build_iwildcam_2022 import build_iwildcam_2022_dataset


def test_build_iwildcam_2022_dataset():
    # Test if the function returns the correct dataset split

    datasets = build_iwildcam_2022_dataset(
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
        datasets = build_iwildcam_2022_dataset(
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_iwildcam_2022_dataloader():
    datasets = build_iwildcam_2022_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["val"],
    )
    val_set = datasets["val"]

    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

    for batch in val_loader:
        assert batch["video"].shape == (2, 10, 3, 224, 224)
        assert batch["index"].shape == (2,)
        assert len(batch["counts"]) == 2
        assert len(batch["num_frames"]) == 2
        assert batch["locations"].shape == (2, 10)
        assert batch["sub_locations"].shape == (2, 10)
        assert batch["sub_location_available"].shape == (2, 10)
        assert batch["utc_timestamps"].shape == (2, 10)
        assert batch["category_ids"].shape == (2, 10)
        assert batch["gps_locations"].shape == (2, 10, 2)
        assert batch["gps_sub_locations"].shape == (2, 10, 2)
        assert batch["gps_available"].shape == (2, 10)
        assert batch["gps_sub_available"].shape == (2, 10)
        assert batch["instance_masks"].shape == (2, 10, 224, 224)
        assert batch["max_detection_confs"].shape == (2, 10)
        assert batch["num_detections"].shape == (2, 10)
        assert batch["detection_categories"].shape == (2, 10, 23)
        assert batch["detection_confs"].shape == (2, 10, 23)
        assert batch["detection_bboxes"].shape == (2, 10, 23, 4)
        break


if __name__ == "__main__":
    test_build_iwildcam_2022_dataset()
    test_iwildcam_2022_dataloader()
