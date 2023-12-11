import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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


class IoUMetric:
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        class_idx_to_name: Optional[dict] = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.class_idx_to_name = class_idx_to_name
        self.total_updates = 0

        self.total_area_intersect = torch.zeros(num_classes)
        self.total_area_union = torch.zeros(num_classes)
        self.total_area_pred = torch.zeros(num_classes)
        self.total_area_label = torch.zeros(num_classes)
        logger.info(self)

    def __repr__(self):
        return (
            f"IoUMetric(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, "
            f"class_idx_to_name={self.class_idx_to_name})"
        )

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        pred = pred.clone().cpu().view(-1)
        label = label.clone().cpu().view(-1)

        if self.ignore_index is not None:
            keep_mask = label != self.ignore_index

            keep_mask = keep_mask.to(pred.device)
            pred = pred[keep_mask]
            label = label[keep_mask]

        intersect = pred[pred == label]
        area_intersect = torch.bincount(intersect, minlength=self.num_classes)

        area_union = (
            torch.bincount(pred, minlength=self.num_classes)
            + torch.bincount(label, minlength=self.num_classes)
            - area_intersect
        )

        area_label = torch.bincount(label, minlength=self.num_classes)

        self.total_area_intersect += area_intersect.float()
        self.total_area_union += area_union.float()
        self.total_area_label += area_label.float()
        self.total_updates += 1

    def reset(self):
        self.total_area_intersect = torch.zeros(self.num_classes)
        self.total_area_union = torch.zeros(self.num_classes)
        self.total_area_label = torch.zeros(self.num_classes)
        self.total_updates = 0

    def compute_metrics(self):
        # IoU Calculation
        iou = torch.zeros_like(self.total_area_union)
        non_zero_union_mask = self.total_area_union > 0
        iou[non_zero_union_mask] = self.total_area_intersect[
            non_zero_union_mask
        ] / (self.total_area_union[non_zero_union_mask] + 1e-6)
        iou[~non_zero_union_mask] = torch.tensor(float("nan"))

        valid_iou = iou[~torch.isnan(iou)]
        miou = valid_iou.mean().item() * 100.0

        # Per-class Accuracy
        per_class_acc = torch.zeros_like(self.total_area_label)
        valid_label_mask = self.total_area_label > 0
        per_class_acc[valid_label_mask] = (
            self.total_area_intersect[valid_label_mask]
            / (self.total_area_label[valid_label_mask] + 1e-6)
        ) * 100.0
        per_class_acc[~valid_label_mask] = torch.tensor(float("nan"))

        # Overall Accuracy
        valid_intersect = (
            self.total_area_intersect[~torch.isnan(per_class_acc)].sum().item()
        )
        valid_label = (
            self.total_area_label[~torch.isnan(per_class_acc)].sum().item()
        )
        overall_acc = (valid_intersect / (valid_label + 1e-6)) * 100.0

        # Mean Accuracy
        valid_acc = per_class_acc[~torch.isnan(per_class_acc)]
        mean_acc = valid_acc.mean().item()

        # Convert to dictionary format
        per_class_iou = iou * 100.0

        if self.class_idx_to_name:
            per_class_iou = {
                self.class_idx_to_name[i]: val.item()
                for i, val in enumerate(per_class_iou)
            }
            per_class_acc = {
                self.class_idx_to_name[i]: val.item()
                for i, val in enumerate(per_class_acc)
            }

        return {
            "mIoU": miou,
            "per_class_iou": per_class_iou,
            "per_class_accuracy": per_class_acc,
            "overall_accuracy": overall_acc,
            "mean_accuracy": mean_acc,
        }

    def pretty_print(self, metrics: Optional[dict] = None):
        from rich.console import Console
        from rich.table import Table

        if metrics is None:
            metrics = self.compute_metrics()

        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Class")
        table.add_column("IoU")
        table.add_column("Accuracy")

        for idx in range(self.num_classes):
            if self.class_idx_to_name is not None:
                class_name = self.class_idx_to_name[idx]
            else:
                class_name = str(idx)

            table.add_row(
                class_name,
                f"{metrics['per_class_iou'][class_name]:.2f}",
                f"{metrics['per_class_accuracy'][class_name]:.2f}",
            )

        console.print(table)
        console.print(f"mIoU: {metrics['mIoU']:.2f}")
        console.print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}")
        console.print(f"Mean Accuracy: {metrics['mean_accuracy']:.2f}")


def one_hot(labels: torch.Tensor, num_classes: int):
    """
    Convert labels to one-hot vectors.

    Args:
        labels (torch.Tensor): A 1D tensor containing the class labels.
        num_classes (int): The number of distinct classes.

    Returns:
        torch.Tensor: A 2D tensor of one-hot encoded labels with shape (len(labels), num_classes).
    """
    one_hot_vectors = torch.zeros(
        labels.size(0), num_classes, dtype=torch.float32, device=labels.device
    )
    one_hot_vectors.scatter_(1, labels.unsqueeze(1), 1.0)
    return one_hot_vectors


class FocalLoss(nn.Module):
    def __init__(
        self, alpha=0.25, gamma=2.0, reduction="mean", ignore_index=None
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        b, c, h, w = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(-1, c)
        labels = labels.view(-1)

        if self.ignore_index is not None:
            # The computation of ignore_mask can potentially be sped up by avoiding the unnecessary expansion of dimensions and broadcasting directly where it's used.
            ignore_mask = labels != self.ignore_index

            # Ensure all tensors are on the same device
            ignore_mask = ignore_mask.to(logits.device)
            logits = logits[ignore_mask]
            labels = labels[ignore_mask]

        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction="mean", ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        b, c, h, w = logits.shape
        logits = F.softmax(logits, dim=1)
        labels = labels.squeeze(1)

        labels_one_hot = torch.zeros_like(logits)

        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

        if self.ignore_index is not None:
            # The computation of ignore_mask can potentially be sped up by avoiding the unnecessary expansion of dimensions and broadcasting directly where it's used.
            ignore_mask = labels != self.ignore_index

            # Ensure all tensors are on the same device
            ignore_mask = ignore_mask.to(logits.device)
            labels_one_hot = labels_one_hot.to(logits.device)

            # Use in-place broadcasting and multiplication instead of expanding ignore_mask unnecessarily
            labels_one_hot *= ignore_mask.unsqueeze(1)

        intersection = torch.sum(logits * labels_one_hot, dim=(2, 3))
        union = torch.sum(logits, dim=(2, 3)) + torch.sum(
            labels_one_hot, dim=(2, 3)
        )

        dice_scores = (2.0 * intersection + self.smooth) / (
            union + self.smooth
        )
        dice_loss = 1.0 - dice_scores

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


def compute_class_weights(labels, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.float).to(
        labels.device
    )

    for cls in range(num_classes):
        class_counts[cls] = torch.sum(labels == cls).float()

    # To avoid division by zero, add a small epsilon
    epsilon = 1e-6
    class_weights = 1.0 / (class_counts + epsilon)

    # Normalize the weights so that they sum up to 1
    class_weights -= class_weights.max()
    class_weights = torch.exp(class_weights)
    class_weights /= torch.sum(class_weights)

    return class_weights


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        ignore_index: int = -1,
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def compute_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights based on label frequency.

        :param labels: torch.Tensor with shape (b, h, w)
        :return: torch.Tensor with shape (num_classes,)
        """
        unique_labels, counts = torch.unique(labels, return_counts=True)
        counts_float = counts.type(torch.float)

        # Apply the inverse of the class frequency
        class_weights = 1 / counts_float + 1e-6

        # Normalize the vector using a softmax function
        class_weights = torch.softmax(class_weights, dim=0)

        return class_weights

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the weighted cross-entropy loss.

        :param logits: torch.Tensor with shape (b, num_classes, h, w)
        :param labels: torch.Tensor with shape (b, h, w)
        :return: torch.Tensor representing the loss
        """
        labels = labels.squeeze(1)
        class_weights = self.compute_class_weights(labels) * 1000

        # Perform standard cross-entropy loss computation
        loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        ignore_index: int = -1,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the weighted cross-entropy loss.

        :param logits: torch.Tensor with shape (b, num_classes, h, w)
        :param labels: torch.Tensor with shape (b, h, w)
        :return: torch.Tensor representing the loss
        """
        labels = labels.squeeze(1)

        # Perform standard cross-entropy loss computation
        loss = nn.functional.cross_entropy(
            logits,
            labels,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        return loss
