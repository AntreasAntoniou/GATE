import pytest
import torch
import numpy as np
import wandb

from gate.boilerplate.utils import log_wandb_masks

wandb.init()


def test_log_wandb_masks():
    images = torch.rand(5, 3, 256, 256)
    predicted_masks = np.random.randint(0, 19, (5, 256, 256))
    labels = np.random.randint(0, 19, (5, 256, 256))
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
        log_wandb_masks(
            images,
            predicted_masks,
            labels,
            label_idx_to_description_dict,
            num_to_log=5,
        )
    except Exception as e:
        pytest.fail(f"Failed to log masks with wandb: {e}")
