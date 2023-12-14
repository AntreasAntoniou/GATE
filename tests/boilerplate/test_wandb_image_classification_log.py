import pytest
import torch
import wandb
from PIL import Image
from torchvision import transforms as T

from gate.boilerplate.wandb_utils import log_wandb_image_classification

# Mock WandB init and finish
wandb.init(project="image_classification_visualization_test")


# Test case with a single image, label, random logits
def test_log_wandb_image_classification_single():
    images = [Image.new("RGB", (128, 128)) for _ in range(3)]
    images = torch.stack([T.ToTensor()(img) for img in images])
    labels = [0, 1, 2]
    logits = torch.tensor([0.1, 0.9, 0.5]).unsqueeze(0).repeat(3, 1)
    try:
        log_wandb_image_classification(images, labels, logits)
    except Exception as e:
        pytest.fail(f"Test failed with single image and label. Error: {e}")


# Test case with multiple images, labels, random logits
def test_log_wandb_image_classification_multiple():
    images = [Image.new("RGB", (128, 128)) for _ in range(3)]
    images = torch.stack([T.ToTensor()(img) for img in images])
    labels = [0, 1, 2]
    logits = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    try:
        log_wandb_image_classification(images, labels, logits)
    except Exception as e:
        pytest.fail(f"Test failed with multiple images and labels. Error: {e}")


# Test case with a dictionary of logits and labels
def test_log_wandb_image_classification_dict_logits_labels():
    images = [Image.new("RGB", (128, 128)) for _ in range(3)]
    images = torch.stack([T.ToTensor()(img) for img in images])
    labels = {"label1": [0] * 3, "label2": [1] * 3, "label3": [2] * 3}
    logits = {
        "logit1": torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0).repeat(3, 1),
        "logit2": torch.tensor([0.4, 0.5, 0.6]).unsqueeze(0).repeat(3, 1),
        "logit3": torch.tensor([0.7, 0.8, 0.9]).unsqueeze(0).repeat(3, 1),
    }
    try:
        log_wandb_image_classification(images, labels, logits)
    except Exception as e:
        pytest.fail(
            f"Test failed with dictionary logits and labels. Error: {e}"
        )
