import monai
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score as compute_roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from einops import rearrange


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


def dice_loss(logits, labels):
    smooth = 1.0
    logits = torch.softmax(logits, dim=1)
    logits = logits.view(-1)
    labels = labels.view(-1)
    intersection = (logits * labels).sum()
    return 1 - (
        (2.0 * intersection + smooth) / (logits.sum() + labels.sum() + smooth)
    )


def miou_loss(logits, labels):
    smooth = 1.0
    logits = torch.softmax(logits, dim=1)
    logits = logits.view(-1)
    labels = labels.view(-1)
    intersection = (logits * labels).sum()
    union = logits.sum() + labels.sum() - intersection
    return 1 - ((intersection + smooth) / (union + smooth))


def roc_auc_score(logits, labels):
    logits = torch.softmax(logits, dim=1)
    logits = (
        logits.permute(0, 2, 3, 1)
        .reshape(-1, logits.shape[1])
        .cpu()
        .detach()
        .numpy()
    )
    labels = labels.view(-1).cpu().detach().numpy()

    lb = LabelBinarizer()
    lb.fit(labels)
    labels_binarized = lb.transform(labels)

    roc_auc = compute_roc_auc_score(
        labels_binarized, logits, multi_class="ovr"
    )
    return roc_auc


def generalized_dice_loss(logits, labels):
    smooth = 1.0
    logits = torch.softmax(logits, dim=1)
    logits = logits.view(-1)
    labels = labels.view(-1)
    intersection = (logits * labels).sum()
    sum_ = logits.sum() + labels.sum()
    w = 1 / (sum_**2 + smooth)
    return 1 - ((2 * w * intersection + smooth) / (w * sum_ + smooth))


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
