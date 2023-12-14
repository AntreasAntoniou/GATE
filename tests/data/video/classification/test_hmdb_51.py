import os

import pytest
import wandb
from torch.utils.data import DataLoader

from gate.boilerplate.utils import visualize_video_with_labels
from gate.data.video.classification.build_gulp_sparsesample import (
    build_hmdb51_gate_dataset,
)


# Helper function to initialize wandb if you wish to visualize
def init_wandb():
    import wandb

    wandb.init(project="video-dataset-visualization", job_type="dataset_test")


# Test for build_gate_dataset
def test_build_gate_dataset():
    datasets = build_hmdb51_gate_dataset(os.environ.get("PYTEST_DIR"))

    assert datasets is not None, "Dataset should not be None"
    for set_name in ["train", "val", "test"]:
        assert set_name in datasets, f"{set_name} should be in the dataset"


# Test for visualization in wandb
@pytest.mark.visual
def test_visualize_in_wandb():
    datasets = build_hmdb51_gate_dataset(os.environ.get("PYTEST_DIR"))

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
