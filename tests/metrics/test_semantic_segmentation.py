import torch

from gate.metrics.segmentation import (
    normalized_surface_dice_loss,
    dice_loss,
    miou_loss,
    generalized_dice,
    roc_auc_score,
)


def test_dice_loss():
    torch.manual_seed(42)
    labels = torch.randint(0, 100, (1, 1, 224, 224))
    labels[0, 0, 112, 112] = 1
    preds = torch.zeros(1, 100, 224, 224)
    preds[0, 1, 112, 112] = 1  # Predict class 1 for the center pixel
    dice_loss(preds, labels, label_dim=1, num_classes=100).mean().item()
    preds[0, 1, 0, 0] = 1  # Predict class 1 for another pixel
    dice_loss(preds, labels, label_dim=1, num_classes=100).mean().item()


def test_miou_loss():
    torch.manual_seed(42)
    labels = torch.randint(0, 100, (1, 1, 224, 224))
    labels[0, 0, 112, 112] = 1
    preds = torch.zeros(1, 100, 224, 224)
    preds[0, 1, 112, 112] = 1  # Predict class 1 for the center pixel
    miou_loss(preds, labels, label_dim=1, num_classes=100).mean().item()
    preds[0, 1, 0, 0] = 1  # Predict class 1 for another pixel
    miou_loss(preds, labels, label_dim=1, num_classes=100).mean().item()


def test_generalized_dice():
    torch.manual_seed(42)
    labels = torch.randint(0, 100, (1, 1, 224, 224))
    labels[0, 0, 112, 112] = 1
    preds = torch.zeros(1, 100, 224, 224)
    preds[0, 1, 112, 112] = 1  # Predict class 1 for the center pixel
    generalized_dice(preds, labels, label_dim=1, num_classes=100).mean().item()
    preds[0, 1, 0, 0] = 1  # Predict class 1 for another pixel
    generalized_dice(preds, labels, label_dim=1, num_classes=100).mean().item()


def test_roc_auc_score():
    torch.manual_seed(42)
    labels = torch.randint(0, 100, (1, 1, 224, 224))
    labels[0, 0, 112, 112] = 1
    preds = torch.zeros(1, 100, 224, 224)
    preds[0, 1, 112, 112] = 1  # Predict class 1 for the center pixel
    roc_auc_score(preds, labels, label_dim=1, num_classes=100).mean().item()
    preds[0, 1, 0, 0] = 1  # Predict class 1 for another pixel
    roc_auc_score(preds, labels, label_dim=1, num_classes=100).mean().item()
