# test_mscoco_10k.py
import os

import pytest

from gate.data.image.segmentation.coco_164k import build_cocostuff164k_dataset


def test_invalid_set_name():
    with pytest.raises(ValueError):
        build_cocostuff164k_dataset(
            data_dir=os.environ.get("PYTEST_DIR") + "/coco_164k",
            split="invalid",
        )


# Note: This test requires internet connection and may take a while to complete
@pytest.mark.parametrize("set_name", ["train", "test"])
def test_download(set_name):
    dataset = build_cocostuff164k_dataset(
        split=set_name,
        data_dir=os.environ.get("PYTEST_DIR") + "/coco_164k",
        download=True,
    )
    assert len(dataset) > 0, f"{set_name} dataset should not be empty"


@pytest.mark.parametrize("set_name", ["train", "val", "test"])
def test_set_name(set_name):
    dataset = build_cocostuff164k_dataset(
        split=set_name,
        data_dir=os.environ.get("PYTEST_DIR") + "/coco_164k",
        download=True,
    )
    assert len(dataset) > 0, f"{set_name} dataset should not be empty"

    # Check if the images and annotations are properly loaded
    idx, img, ann = dataset[0]
    assert img is not None, f"{set_name} dataset should have non-empty images"
    assert (
        ann is not None
    ), f"{set_name} dataset should have non-empty annotations"


if __name__ == "__main__":
    test_invalid_set_name()
    test_download("train")
    test_set_name("train")
