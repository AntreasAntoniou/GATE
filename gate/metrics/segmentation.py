import torch
import numpy as np


def dice_loss(predictions, labels, smooth=1.0):
    """
    Compute the Dice Loss for binary segmentation.

    Args:
        predictions (torch.Tensor): the network's predictions (B, Classes, H, W)
        labels (torch.Tensor): the ground truth segmentation masks (B, 1, H, W)
        smooth (float, optional): a smoothing constant to prevent division by zero. Defaults to 1.0.

    Returns:
        torch.Tensor: the computed Dice loss.
    """
    intersection = (predictions * labels).sum(dim=2).sum(dim=2)
    loss = 1 - (
        (2.0 * intersection + smooth)
        / (
            predictions.sum(dim=2).sum(dim=2)
            + labels.sum(dim=2).sum(dim=2)
            + smooth
        )
    )

    return loss.mean()


def compute_surface_distances(prediction, label):
    """
    Compute surface distances between predicted and ground truth masks using PyTorch.
    """
    prediction_dist = 1.0 - prediction
    label_dist = 1.0 - label

    pred_dist_to_border = torch.where(
        prediction,
        prediction_dist,
        torch.tensor(float("inf")).to(prediction.device),
    )
    label_dist_to_border = torch.where(
        label, label_dist, torch.tensor(float("inf")).to(label.device)
    )

    surface_distances_pred = torch.where(prediction, label_dist_to_border, 0)
    surface_distances_label = torch.where(label, pred_dist_to_border, 0)

    return surface_distances_pred, surface_distances_label


def normalized_surface_dice(predictions, labels):
    """
    Compute the Normalized Surface Dice (NSD) between predicted and ground truth masks using PyTorch.
    """
    (
        surface_distances_pred,
        surface_distances_label,
    ) = compute_surface_distances(predictions, labels)

    numerator = 2 * torch.sum(surface_distances_pred + surface_distances_label)
    denominator = torch.sum(predictions) + torch.sum(labels)

    nsd = 1 - (numerator / denominator)

    return nsd


def miou_loss(predictions, labels, num_classes):
    """
    Compute the Mean Intersection Over Union (MIOU) loss for the given prediction and ground truth.
    """
    ious = []
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = predictions == cls
        label_inds = labels == cls
        intersection = (pred_inds[label_inds]).long().sum().data.cpu().item()
        union = (
            pred_inds.long().sum().data.cpu().item()
            + label_inds.long().sum().data.cpu().item()
            - intersection
        )
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    return torch.tensor(
        np.nanmean(ious)
    )  # Return the average IoU over all classes
