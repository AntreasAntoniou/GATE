import numpy as np
import pytest
import torch
import wandb

from gate.boilerplate.utils import log_wandb_3d_volumes_and_masks


def test_wandb_mri_segmentation():
    # Create some random data
    b, s, c, h, w = (
        2,
        10,
        3,
        64,
        64,
    )  # batch size, slices, channels, height, width
    input_volumes = torch.rand((b, s, c, h, w)).float()
    predicted_volumes = torch.randint(0, 2, (b, s, c, h, w)).long()
    label_volumes = torch.randint(0, 2, (b, s, c, h, w)).long()

    # Start a Weights & Biases run
    run = wandb.init(
        project="mri-visualization", name="mri-visualization-test"
    )

    # Visualize the data
    log_wandb_3d_volumes_and_masks(
        input_volumes, predicted_volumes, label_volumes
    )

    # Finish the run
    run.finish()
