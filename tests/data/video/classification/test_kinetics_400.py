import os
from pathlib import Path

import pytest
import wandb
from torch.utils.data import DataLoader

from gate.boilerplate.utils import visualize_video_with_labels
from gate.data.video.classification.build_kinetics_400 import (
    build_dataset,
    build_gate_dataset,
)


def test_build_kinetics_400_dataset():
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


def test_kinetics_400_dataloader():
    datasets = build_dataset(
        data_dir=os.environ.get("PYTEST_DIR"),
        sets_to_include=["train", "val", "test"],
    )
    train_set = datasets["train"]
    val_set = datasets["val"]
    test_set = datasets["test"]

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    for batch in train_loader:
        assert batch["video"].shape == (2, 8, 3, 224, 224)
        assert batch["labels"].shape == (2,)
        assert batch["video_ids"].shape == (2,)
        break

    for batch in val_loader:
        assert batch["video"].shape == (2, 8, 3, 224, 224)
        assert batch["labels"].shape == (2,)
        assert batch["video_ids"].shape == (2,)
        break

    for batch in test_loader:
        assert batch["video"].shape == (2, 8, 3, 224, 224)
        assert batch["labels"].shape == (2,)
        assert batch["video_ids"].shape == (2,)
        break


# Helper function to initialize wandb if you wish to visualize
def init_wandb():
    import wandb

    wandb.init(project="video-dataset-visualization", job_type="dataset_test")


# Test for build_dataset
def test_build_dataset():
    sets_to_include = ["train", "val", "test"]

    datasets = build_dataset(
        os.environ.get("PYTEST_DIR"), sets_to_include=sets_to_include
    )

    assert datasets is not None, "Dataset should not be None"
    for set_name in sets_to_include:
        assert set_name in datasets, f"{set_name} should be in the dataset"


# Test for build_gate_dataset
def test_build_gate_dataset():
    datasets = build_gate_dataset(os.environ.get("PYTEST_DIR"))

    assert datasets is not None, "Dataset should not be None"
    for set_name in ["train", "val", "test"]:
        assert set_name in datasets, f"{set_name} should be in the dataset"


# Test for visualization in wandb
@pytest.mark.visual
def test_visualize_in_wandb():
    datasets = build_gate_dataset(os.environ.get("PYTEST_DIR"))

    init_wandb()  # Initialize wandb

    for set_name, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for idx, item in enumerate(dataloader):
            # Replace 'visualize_video' with your actual visualization function
            wandb.log(
                visualize_video_with_labels(
                    item["video"],
                    logits=item["labels"],
                    labels=item["labels"],
                    name=f"{set_name}-visualization",
                )
            )
            if idx > 2:  # Limit the number of visualizations
                break
