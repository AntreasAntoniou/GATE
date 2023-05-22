import monai
import torch


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
):
    if remove_dim:
        return loss_fn(
            logits,
            one_hot_encoding(
                labels.squeeze(label_dim),
                num_classes=num_classes,
                dim=label_dim,
            ),
        )
    else:
        return loss_fn(
            logits,
            one_hot_encoding(
                labels,
                num_classes=num_classes,
                dim=label_dim,
            ),
        )


def normalized_surface_dice_loss(
    logits, labels, label_dim, num_classes, class_thresholds: list = [0.5]
):
    logits = logits.permute(0, 2, 3, 1).view(-1, num_classes)
    labels = labels.permute(0, 2, 3, 1).view(-1)
    return loss_adapter(
        loss_fn=monai.metrics.compute_surface_dice,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
        remove_dim=False,
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


def generalized_dice(logits, labels, label_dim, num_classes):
    return loss_adapter(
        loss_fn=monai.metrics.compute_generalized_dice,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
    )


def roc_auc_score(logits, labels, label_dim, num_classes):
    logits = logits.permute(0, 2, 3, 1).view(-1, num_classes)
    labels = labels.permute(0, 2, 3, 1).view(-1)
    return loss_adapter(
        loss_fn=monai.metrics.compute_roc_auc,
        logits=logits,
        labels=labels,
        label_dim=label_dim,
        num_classes=num_classes,
        remove_dim=False,
    )
