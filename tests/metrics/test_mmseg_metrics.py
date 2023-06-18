import pytest
import torch
import torch.nn.functional as F
from mmseg.models.losses import (
    DiceLoss,
    FocalLoss,
    CrossEntropyLoss,
    LovaszLoss,
)

from gate.metrics.segmentation import one_hot_encoding


def test_dice_loss():
    torch.manual_seed(2306)

    dice_loss = DiceLoss()

    # Prepare sample logits and labels
    logits = (
        torch.randn(8, 150, 128, 128).permute([0, 2, 3, 1]).reshape(-1, 150)
    )

    labels = torch.randint(0, 149, size=(8, 128, 128)).view(-1)

    print(f"logits: {logits.shape}, labels: {labels.shape}")

    loss = dice_loss(logits, labels)

    expected_loss = torch.tensor(0.0034)

    assert torch.isclose(loss, expected_loss, atol=1e-4)


def test_focal_loss():
    torch.manual_seed(2306)

    focal_loss = FocalLoss()

    # Prepare sample logits and labels
    logits = (
        torch.randn(8, 150, 128, 128).permute([0, 2, 3, 1]).reshape(-1, 150)
    )

    labels = torch.randint(0, 149, size=(8, 128, 128)).view(-1)

    print(f"logits: {logits.shape}, labels: {labels.shape}")

    loss = focal_loss(logits, labels)

    expected_loss = torch.tensor(0.1733)

    assert torch.isclose(loss, expected_loss, atol=1e-4)


def test_mask_cross_entropy():
    torch.manual_seed(2306)

    mask_bce_loss = CrossEntropyLoss(use_mask=True)

    # Prepare sample logits and labels
    logits = (
        torch.randn(8, 150, 128, 128).permute([0, 2, 3, 1]).reshape(-1, 150)
    )

    labels = torch.randint(0, 149, size=(8, 128, 128)).view(-1)
    labels = one_hot_encoding(labels, num_classes=150, dim=1)

    print(f"logits: {logits.shape}, labels: {labels.shape}")

    loss = mask_bce_loss.forward(logits, labels, ignore_index=None)

    print(loss)

    expected_loss = torch.tensor(0.8062)

    assert torch.isclose(loss, expected_loss, atol=1e-4)
