import monai
import torch
import torch.nn.functional as F


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


SMOOTH = 1e-6


def fast_miou(logits: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    print(logits.shape, labels.shape)
    logits = logits.argmax(dim=1).squeeze(
        1
    )  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)

    print(logits.shape, labels.shape)

    intersection = (
        (logits & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (
        (logits | labels).float().sum((1, 2))
    )  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (
        union + SMOOTH
    )  # We smooth our devision to avoid 0/0

    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds

    return (
        thresholded.mean()
    )  # Or thresholded.mean() if you are interested in average across the batch
