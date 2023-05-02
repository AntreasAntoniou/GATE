import pytest
import os
from gate.data.medical.acdc import download_and_extract_file, ACDCDataset

# You can set this to the path where you have the dataset stored on your machine
DATASET_PATH = os.environ.get("PYTEST_DIR")


def test_download_and_extract_file():
    dataset_path = download_and_extract_file(DATASET_PATH)
    assert dataset_path.is_dir()
    assert (dataset_path / "database").is_dir()


@pytest.mark.parametrize("mode", ["train", "test"])
def test_acdc_dataset(mode):
    dataset = ACDCDataset(root_dir=DATASET_PATH, mode=mode)

    assert len(dataset) > 0

    sample = dataset[0]

    assert "four_d_img" in sample
    assert "frame_data" in sample

    four_d_img = sample["four_d_img"]
    assert four_d_img.dim() == 4

    frame_data = sample["frame_data"]
    assert len(frame_data) > 0

    for frame in frame_data:
        assert "img" in frame
        assert "label" in frame
        img = frame["img"]
        label = frame["label"]
        assert img.dim() == 3
        assert label.dim() == 3
        assert img.shape == label.shape
