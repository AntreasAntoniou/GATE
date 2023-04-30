# test_mscoco_10k.py
import os

import pytest

from gate.data.image.segmentation.coco_10k import build_cocostuff10k_dataset


def test_invalid_set_name():
    with pytest.raises(ValueError):
        build_cocostuff10k_dataset(
            data_dir=os.environ.get("TEST_DIR"), split="invalid"
        )


@pytest.mark.parametrize("set_name", ["train", "val", "test"])
def test_set_name(set_name):
    dataset = build_cocostuff10k_dataset(
        split=set_name, data_dir=os.environ.get("TEST_DIR")
    )
    assert len(dataset) > 0, f"{set_name} dataset should not be empty"

    # Check if the images and annotations are properly loaded
    idx, img, ann = dataset[0]
    assert img is not None, f"{set_name} dataset should have non-empty images"
    assert (
        ann is not None
    ), f"{set_name} dataset should have non-empty annotations"


# Note: This test requires internet connection and may take a while to complete
@pytest.mark.parametrize("set_name", ["train", "test"])
def test_download(set_name):
    dataset = build_cocostuff10k_dataset(
        split=set_name, data_dir=os.environ.get("TEST_DIR"), download=True
    )
    assert len(dataset) > 0, f"{set_name} dataset should not be empty"
