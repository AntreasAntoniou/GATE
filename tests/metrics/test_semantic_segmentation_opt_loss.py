import torch
import pytest

from gate.metrics.segmentation import (
    DiceLoss,
    FocalLoss,
    WeightedCrossEntropyLoss,
)


@pytest.fixture
def logits():
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def labels():
    return torch.randint(0, 3, (4, 1, 32, 32))


def test_dice_loss_shape(logits, labels):
    dice_loss = DiceLoss()
    loss = dice_loss(logits, labels)
    assert loss.shape == torch.Size([]), "DiceLoss should return a scalar loss"


def test_dice_loss_value(logits, labels):
    dice_loss = DiceLoss()
    loss = dice_loss(logits, labels)
    assert (
        0 <= loss.item() <= 1
    ), "DiceLoss should return a value between 0 and 1"


def test_focal_loss_shape(logits, labels):
    focal_loss = FocalLoss()
    loss = focal_loss(logits, labels)
    assert loss.shape == torch.Size(
        []
    ), "FocalLoss should return a scalar loss"


def test_focal_loss_value(logits, labels):
    focal_loss = FocalLoss()
    loss = focal_loss(logits, labels)
    assert 0 <= loss.item(), "FocalLoss should return a non-negative value"


def test_weighted_cross_entropy_loss_shape(logits, labels):
    weighted_cross_entropy_loss = WeightedCrossEntropyLoss()
    loss = weighted_cross_entropy_loss(logits, labels)
    assert loss.shape == torch.Size(
        []
    ), "WeightedCrossEntropyLoss should return a scalar loss"


def test_weighted_cross_entropy_loss_value(logits, labels):
    weighted_cross_entropy_loss = WeightedCrossEntropyLoss()
    loss = weighted_cross_entropy_loss(logits, labels)
    assert (
        0 <= loss.item()
    ), "WeightedCrossEntropyLoss should return a non-negative value"
