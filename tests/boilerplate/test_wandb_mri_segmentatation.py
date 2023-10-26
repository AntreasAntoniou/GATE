import torch
import wandb

from gate.boilerplate.wandb_utils import log_wandb_3d_volumes_and_masks


def test_wandb_mri_segmentation_channel_1():
    # Create some random data
    b, s, c, h, w = (
        2,
        300,
        1,
        256,
        256,
    )  # batch size, slices, channels, height, width
    input_volumes = torch.rand((b, s, c, h, w)).float()
    predicted_volumes = torch.randint(0, 2, (b, s, h, w)).long()
    label_volumes = torch.randint(0, 2, (b, s, h, w)).long()

    # Start a Weights & Biases run
    run = wandb.init(
        project="mri-visualization", name="mri-visualization-test"
    )

    # Visualize the data
    wandb.log(
        log_wandb_3d_volumes_and_masks(
            input_volumes, predicted_volumes, label_volumes
        )
    )


def test_wandb_mri_segmentation_channel_3():
    # Create some random data
    b, s, c, h, w = (
        2,
        300,
        3,
        256,
        256,
    )  # batch size, slices, channels, height, width
    input_volumes = torch.rand((b, s, c, h, w)).float()
    predicted_volumes = torch.randint(0, 2, (b, s, h, w)).long()
    label_volumes = torch.randint(0, 2, (b, s, h, w)).long()

    # Start a Weights & Biases run
    run = wandb.init(
        project="mri-visualization", name="mri-visualization-test"
    )

    # Visualize the data
    wandb.log(
        log_wandb_3d_volumes_and_masks(
            input_volumes, predicted_volumes, label_volumes
        )
    )
