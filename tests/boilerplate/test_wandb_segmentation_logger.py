import pytest
import torch
import wandb

from gate.boilerplate.wandb_utils import log_wandb_masks

wandb.init()


def test_log_wandb_masks():
    images = torch.rand(5, 3, 256, 256)
    predicted_masks = torch.randint(0, 19, (5, 256, 256))
    labels = torch.randint(0, 19, (5, 256, 256))
    label_descriptions = [
        "background",
        "person",
        "car",
        "dog",
        "cat",
        "tree",
        "sky",
        "building",
        "flower",
        "water",
        "grass",
        "animal",
        "road",
        "mountain",
        "beach",
        "food",
        "snow",
        "sidewalk",
        "airplane",
    ]
    label_idx_to_description_dict = {
        idx: desc for idx, desc in enumerate(label_descriptions)
    }

    try:
        wandb.log(
            log_wandb_masks(
                wandb.run,
                images,
                predicted_masks,
                labels,
                label_idx_to_description_dict,
                prefix="test",
            )
        )
    except Exception as e:
        pytest.fail(f"Failed to log masks with wandb: {e}")
