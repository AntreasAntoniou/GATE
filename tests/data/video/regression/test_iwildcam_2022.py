import os

import pytest
from torch.utils.data import DataLoader

from gate.boilerplate.utils import visualize_video_with_labels
from gate.data.video.regression.build_iwildcam_2022 import (
    build_dataset,
    build_gate_dataset,
)


# Helper function to initialize wandb if you wish to visualize
def init_wandb():
    import wandb

    wandb.init(project="video-dataset-visualization", job_type="dataset_test")


def test_build_iwildcam_2022_dataset():
    # Test if the function returns the correct dataset split

    datasets = build_dataset(
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
        datasets = build_dataset(
            data_dir=os.environ.get("PYTEST_DIR"),
            sets_to_include=["invalid_set_name"],
        )


def test_iwildcam_2022_dataloader():
    datasets = build_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
    )
    val_set = datasets["val"]

    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

    for set_name, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for idx, item in enumerate(dataloader):
            assert item["video"].shape == (2, 10, 3, 224, 224)
            assert item["index"].shape == (2,)
            assert len(item["counts"]) == 2
            assert len(item["num_frames"]) == 2
            break


# Test for visualization in wandb
@pytest.mark.visual
def test_visualize_in_wandb():
    datasets = build_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
    )

    init_wandb()  # Initialize wandb

    for set_name, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for idx, item in enumerate(dataloader):
            # Replace 'visualize_video' with your actual visualization function
            visualize_video_with_labels(
                item["video"],
                targets=item["counts"],
                name=f"{set_name}-visualization",
            )
            if idx > 2:  # Limit the number of visualizations
                break


if __name__ == "__main__":
    test_build_iwildcam_2022_dataset()
    test_iwildcam_2022_dataloader()
