import time
import monai
import torch
import torch.nn.functional as F
import evaluate

from gate.boilerplate.utils import get_logger

logger = get_logger(__name__)


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


def normalized_surface_dice_loss(
    logits, labels, label_dim, num_classes, class_thresholds: list = [0.5]
):
    logits = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    labels = labels.permute(0, 2, 3, 1).reshape(-1)
    return loss_adapter(
        loss_fn=monai.metrics.compute_surface_dice,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
        remove_dim=True,
        class_thresholds=class_thresholds,
    )


def dice_loss(logits, labels, label_dim, num_classes):
    return loss_adapter(
        loss_fn=monai.metrics.compute_meandice,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
    )


def miou_loss(logits, labels, label_dim, num_classes):
    return loss_adapter(
        loss_fn=monai.metrics.compute_meaniou,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
    )


def generalized_dice_loss(logits, labels, label_dim, num_classes):
    return loss_adapter(
        loss_fn=monai.metrics.compute_generalized_dice,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
    )


def roc_auc_score(logits, labels, label_dim, num_classes):
    return loss_adapter(
        loss_fn=monai.metrics.compute_roc_auc,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
        remove_dim=False,
    )


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


import torch
from typing import Optional, Dict


def intersect_and_union(
    pred_label,
    label,
    num_labels,
    ignore_index: int,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = label != ignore_index
    mask = torch.not_equal(label, ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]

    area_intersect = torch.histogram(
        intersect.float(), bins=num_labels, range=(0, num_labels - 1)
    )[0]
    area_pred_label = torch.histogram(
        pred_label.float(), bins=num_labels, range=(0, num_labels - 1)
    )[0]
    area_label = torch.histogram(
        label.float(), bins=num_labels, range=(0, num_labels - 1)
    )[0]

    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: int,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    total_area_intersect = torch.zeros((num_labels,), dtype=torch.float64)
    total_area_union = torch.zeros((num_labels,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_labels,), dtype=torch.float64)
    total_area_label = torch.zeros((num_labels,), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        (
            area_intersect,
            area_union,
            area_pred_label,
            area_label,
        ) = intersect_and_union(
            result,
            gt_seg_map,
            num_labels,
            ignore_index,
            label_map,
            reduce_labels,
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )


def mean_iou(
    logits,
    labels,
    num_labels,
    ignore_index: int,
    nan_to_num: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    ) = total_intersect_and_union(
        logits,
        labels,
        num_labels,
        ignore_index,
        label_map,
        reduce_labels,
    )

    # compute metrics
    metrics = dict()

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label

    metrics["mean_iou"] = torch.nanmean(iou)
    metrics["mean_accuracy"] = torch.nanmean(acc)
    metrics["overall_accuracy"] = all_acc
    # metrics["per_category_iou"] = iou
    # metrics["per_category_accuracy"] = acc

    if nan_to_num is not None:
        metrics = dict(
            {
                metric: torch.nan_to_num(
                    metric_value, nan=torch.tensor(nan_to_num)
                )
                for metric, metric_value in metrics.items()
            }
        )

    return metrics


def fast_miou(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0
):
    """
    Compute mean Intersection over Union (IoU) for a batch of predicted segmentation masks and ground truth labels.

    Args:
        logits (torch.Tensor): Predicted segmentation masks, shape (batch_size, num_classes, height, width).
        labels (torch.Tensor): Ground truth labels, shape (batch_size, 1, height, width).

    Returns:
        mean_iou (torch.Tensor): Mean IoU for the batch.
    """
    # Ensure the logits are a probability distribution (i.e., softmax has been applied)
    # Then, get the predicted class for each pixel (shape: batch_size, height, width)
    num_classes = logits.shape[1]
    logits = logits.argmax(dim=1).detach().cpu()

    # Remove the channel dimension from labels (shape: batch_size, height, width)
    labels = labels.squeeze(1).detach().cpu()

    # Inputs
    # Mandatory inputs

    # predictions (List[ndarray]): List of predicted segmentation maps, each of shape (height, width).
    # Each segmentation map can be of a different size.
    # references (List[ndarray]): List of ground truth segmentation maps, each of shape (height, width).
    # Each segmentation map can be of a different size.
    # num_labels (int): Number of classes (categories).
    # ignore_index (int): Index that will be ignored during evaluation.
    # Optional inputs

    # nan_to_num (int): If specified, NaN values will be replaced by the number defined by the user.
    # label_map (dict): If specified, dictionary mapping old label indices to new label indices.
    # reduce_labels (bool): Whether or not to reduce all label values of segmentation maps by 1.
    # Usually used for datasets where 0 is used for background, and background itself is not included
    # in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
    # The default value is False.

    # mean_iou = evaluate.load("mean_iou")
    return mean_iou(
        logits=logits,
        labels=labels,
        num_labels=num_classes,
        ignore_index=ignore_index,
        nan_to_num=1e-8,
    )


def fast_miou_numpy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0
):
    """
    Compute mean Intersection over Union (IoU) for a batch of predicted segmentation masks and ground truth labels.

    Args:
        logits (torch.Tensor): Predicted segmentation masks, shape (batch_size, num_classes, height, width).
        labels (torch.Tensor): Ground truth labels, shape (batch_size, 1, height, width).

    Returns:
        mean_iou (torch.Tensor): Mean IoU for the batch.
    """
    # Ensure the logits are a probability distribution (i.e., softmax has been applied)
    # Then, get the predicted class for each pixel (shape: batch_size, height, width)
    num_classes = logits.shape[1]
    logits = logits.argmax(dim=1)

    # Remove the channel dimension from labels (shape: batch_size, height, width)
    labels = labels.squeeze(1)

    # Inputs
    # Mandatory inputs

    # predictions (List[ndarray]): List of predicted segmentation maps, each of shape (height, width).
    # Each segmentation map can be of a different size.
    # references (List[ndarray]): List of ground truth segmentation maps, each of shape (height, width).
    # Each segmentation map can be of a different size.
    # num_labels (int): Number of classes (categories).
    # ignore_index (int): Index that will be ignored during evaluation.
    # Optional inputs

    # nan_to_num (int): If specified, NaN values will be replaced by the number defined by the user.
    # label_map (dict): If specified, dictionary mapping old label indices to new label indices.
    # reduce_labels (bool): Whether or not to reduce all label values of segmentation maps by 1.
    # Usually used for datasets where 0 is used for background, and background itself is not included
    # in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
    # The default value is False.

    mean_iou_numpy = evaluate.load("mean_iou")
    return mean_iou_numpy.compute(
        predictions=logits,
        references=labels,
        num_labels=num_classes,
        ignore_index=ignore_index,
        nan_to_num=1e-8,
    )
