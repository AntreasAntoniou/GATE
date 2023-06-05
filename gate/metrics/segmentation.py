import monai
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score as compute_roc_auc_score


def one_hot_encoding(tensor, num_classes, dim):
    # Ensure the tensor is a LongTensor
    tensor = tensor.long()

    # Get the size of the tensor
    size = list(tensor.size())

    # Insert the number of classes at the specified dimension
    size.insert(dim, num_classes)

    # Create a new tensor of zeros with the extended size
    one_hot = torch.zeros(size, device=tensor.device)

    # Use scatter to input the original tensor into the one-hot tensor
    one_hot.scatter_(dim, tensor.unsqueeze(dim), 1)

    return one_hot


def loss_adapter(
    loss_fn,
    logits,
    labels,
    label_dim,
    num_classes,
    remove_dim: bool = True,
    **kwargs,
):
    if remove_dim:
        labels = labels.squeeze(label_dim)

    labels_one_hot = one_hot_encoding(
        labels,
        num_classes=num_classes,
        dim=label_dim,
    )

    # print(
    #     f"labels_one_hot.shape: {labels_one_hot.shape}, logits.shape: {logits.shape}"
    # )

    return loss_fn(
        logits,
        labels_one_hot,
        **kwargs,
    )


def dice_loss(logits, targets):
    b, classes, h, w = logits.shape
    logits = torch.softmax(logits, dim=1)
    targets_one_hot = (
        torch.zeros(b, classes, h, w)
        .to(targets.device)
        .scatter_(1, targets.unsqueeze(1), 1)
    )

    smooth = 1e-6
    intersection = torch.sum(targets_one_hot * logits, dim=(0, 2, 3))
    union = torch.sum(targets_one_hot, dim=(0, 2, 3)) + torch.sum(
        logits, dim=(0, 2, 3)
    )
    dice_coefficient = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_coefficient.mean()

    return dice_loss


def miou_loss(logits, targets):
    b, classes, h, w = logits.shape
    logits = torch.softmax(logits, dim=1)
    targets_one_hot = (
        torch.zeros(b, classes, h, w)
        .to(targets.device)
        .scatter_(1, targets.unsqueeze(1), 1)
    )

    smooth = 1e-6
    intersection = torch.sum(targets_one_hot * logits, dim=(0, 2, 3))
    union = (
        torch.sum(targets_one_hot, dim=(0, 2, 3))
        + torch.sum(logits, dim=(0, 2, 3))
        - intersection
    )
    iou = (intersection + smooth) / (union + smooth)
    miou_loss = 1 - iou.mean()

    return miou_loss


from einops import rearrange

import torch


def int_labels_to_one_hot(labels, num_classes):
    """
    Converts integer labels to one-hot encoded labels.

    Args:
        labels (torch.Tensor): A tensor of integer labels with shape (N, *), where N is the number of samples.
        num_classes (int): The number of unique classes.

    Returns:
        torch.Tensor: A tensor of one-hot encoded labels with shape (N, num_classes, *).
    """
    # Get the shape of the input labels tensor
    shape = labels.shape

    # Create a tensor with the same shape as the input tensor, but with an additional dimension for the classes
    one_hot = torch.zeros(
        *shape, num_classes, device=labels.device, dtype=torch.float32
    )

    # Scatter ones along the class dimension at the positions specified by the integer labels
    one_hot.scatter_(-1, labels.unsqueeze(-1), 1)

    # Rearrange the dimensions to have the class dimension as the second dimension
    one_hot = one_hot.permute(0, -1, *range(1, len(shape)))

    return one_hot


def roc_auc_score(logits, targets):
    logits = rearrange(logits, "b c h w -> (b h w) c")
    targets = rearrange(targets, "b c h w -> (b h w c)")

    logits = torch.softmax(logits, dim=1)
    logits_flat = logits.cpu().detach().numpy()

    # Convert targets to one-hot encoding
    targets_one_hot = int_labels_to_one_hot(
        targets, num_classes=logits.shape[1]
    )
    targets_flat = (
        targets_one_hot.view(-1, targets_one_hot.shape[1])
        .cpu()
        .detach()
        .numpy()
    )

    print(
        f"targets_flat.shape: {targets_flat.shape}, logits_flat.shape: {logits_flat.shape}"
    )

    roc_auc = compute_roc_auc_score(
        targets_flat, logits_flat, multi_class="ovr", average="macro"
    )

    return roc_auc


def generalized_dice_loss(logits, targets):
    b, classes, h, w = logits.shape
    logits = torch.softmax(logits, dim=1)
    targets_one_hot = (
        torch.zeros(b, classes, h, w)
        .to(targets.device)
        .scatter_(1, targets.unsqueeze(1), 1)
    )

    smooth = 1e-6
    intersection = torch.sum(targets_one_hot * logits, dim=(0, 2, 3))
    union = torch.sum(targets_one_hot, dim=(0, 2, 3)) + torch.sum(
        logits, dim=(0, 2, 3)
    )

    class_weights = 1 / (
        (torch.sum(targets_one_hot, dim=(0, 2, 3)) ** 2) + smooth
    )

    generalized_dice_coefficient = (
        2 * torch.sum(intersection * class_weights) + smooth
    ) / (torch.sum(union * class_weights) + smooth)
    generalized_dice_loss = 1 - generalized_dice_coefficient

    return generalized_dice_loss


def diff_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = one_hot_encoding(
        targets,
        num_classes=inputs.shape[1],
        dim=1,
    )

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def diff_sigmoid_focal_loss(
    inputs, targets, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    targets = targets = one_hot_encoding(
        targets,
        num_classes=inputs.shape[1],
        dim=1,
    )
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()
