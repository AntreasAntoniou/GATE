import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


Tensor = torch.Tensor

EPS = 1e-8


def compute_prototypes(
    support: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Computes the class prototypes from the support set without using a for-loop.

    Args:
        embeddings (Tensor): The support set embeddings with shape [N, D], where N is
                          the number of samples and D is the dimensionality.
        labels (Tensor): The labels corresponding to the support set with shape [N].

    Returns:
        Tensor: Class prototypes with shape [C, D], where C is the number of unique
                classes in the support set.
    """
    if len(support.shape) == 3:
        support = support.view(-1, support.shape[-1])

    if len(labels.shape) == 2:
        labels = labels.view(-1)

    unique_labels, class_counts = torch.unique(labels, return_counts=True)

    # Initialize prototypes tensor
    prototypes = torch.zeros(
        num_classes,
        *support.shape[1:],
        device=support.device,
        dtype=support.dtype,
    )

    # Scatter-add to sum embeddings belonging to the same class
    prototypes.index_add_(0, labels, support)

    # Divide by the class counts to get the mean, being careful about zero division
    non_zero_counts = class_counts > 0
    prototypes[unique_labels[non_zero_counts]] /= (
        class_counts[non_zero_counts].view(-1, 1).float()
    )

    return prototypes


def euclidean_distance(prototypes: Tensor, queries: Tensor) -> Tensor:
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.

    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_examples, embedding_size)`.

    labels : `torch.LongTensor` instance
        A tensor containing the labels of the query points. This tensor has
        shape `(batch_size, num_examples)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    n = prototypes.size(0)
    m = queries.size(0)
    prototypes = prototypes.unsqueeze(0).expand(m, n, -1)
    queries = queries.unsqueeze(1).expand(m, n, -1)
    distance = (prototypes - queries).pow(2).sum(dim=2)
    return distance


def compute_prototypical_logits(prototypes: Tensor, queries: Tensor) -> Tensor:
    """Computes the logits for queries based on prototypes.

    Args:
        prototypes (Tensor): Class prototypes with shape [C, D].
        queries (Tensor): Query points with shape [N, D].

    Returns:
        Tensor: Logits matrix with shape [N, C].
    """
    if len(queries.shape) == 3:
        queries = queries.view(-1, queries.shape[-1])

    return -euclidean_distance(prototypes, queries)


def compute_prototypical_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Computes the cross-entropy loss between logits and targets.

    Args:
        logits (Tensor): Logits matrix with shape [N, C].
        targets (Tensor): Ground truth labels with shape [N].

    Returns:
        Tensor: The cross-entropy loss as a scalar tensor.
    """

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss


def compute_prototypical_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Computes the accuracy based on logits and targets.

    Args:
        logits (Tensor): Logits matrix with shape [N, C].
        targets (Tensor): Ground truth labels with shape [N].

    Returns:
        float: The accuracy as a float in the range [0, 1].
    """
    acc = (logits.argmax(dim=1).long() == labels.long()).sum().float()
    return acc / logits.size(0)


def learning_scheduler_smart_autofill(
    lr_scheduler_config, num_train_samples, batch_size
):
    """
    This function is used to autofill the learning scheduler config options.
    """
    if lr_scheduler_config["_target_"].split(".")[-1] == "CosineAnnealingLR":
        if "T_max" not in lr_scheduler_config:
            lr_scheduler_config["T_max"] = num_train_samples / batch_size
    elif (
        lr_scheduler_config["_target_"].split(".")[-1]
        == "CosineAnnealingWarmRestarts"
    ):
        if "T_0" not in lr_scheduler_config:
            lr_scheduler_config["T_0"] = num_train_samples / batch_size // 2

    elif lr_scheduler_config["_target_"].split(".")[-1] == "ReduceLROnPlateau":
        lr_scheduler_config["patience"] = (
            lr_scheduler_config["patience"] * torch.cuda.device_count()
            if torch.cuda.is_available()
            else 1
        )

    return lr_scheduler_config


def get_cosine_distances(query_embeddings, support_embeddings):
    """Compute the cosine distances between all query/support combinations.

    Parameters
    ----------
    query_embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor
        has shape `(batch_size, num_queries, embedding_size)`.

    support_embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_support, embedding_size)`.

    Returns
    -------
    cosine_distances : `torch.FloatTensor` instance
        A tensor containing the distances between all query and support examples. This tensor has shape
        `(batch_size, num_examples, num_queries)`.
    """

    cosine_distances = F.cosine_similarity(
        query_embeddings.unsqueeze(1), support_embeddings.unsqueeze(2), dim=-1
    )

    return cosine_distances


def matching_logits(cosine_distances, labels, num_classes):
    """Compute the matching network logits for each query belonging to each class.

    Parameters
    ----------
    cosine_distances : `torch.FloatTensor` instance
        A tensor containing the distances between all query and support examples. This tensor has shape
        `(batch_size, num_examples, num_queries)`.

    labels : `torch.LongTensor` instance
        A tensor containing the labels of the support points. This tensor has
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    logits : `torch.FloatTensor` instance
        A tensor containing the logits of each query belonging to each class. This tensor has shape
        `(batch_size, num_classes, num_queries)`.
    """
    batch_size, num_queries = cosine_distances.size(0), cosine_distances.size(
        2
    )

    num_samples = get_num_samples(
        labels, num_classes, dtype=cosine_distances.dtype
    )
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    # attentions = cosine_distances
    # For probabilistic attentions as in original paper use softmax:
    attentions = F.softmax(cosine_distances, dim=1)

    logits = attentions.new_zeros((batch_size, num_classes, num_queries))
    indices = labels.unsqueeze(2).expand_as(attentions)
    logits.scatter_add_(1, indices, attentions).div_(num_samples)

    return logits


def matching_loss(logits, labels):
    """Compute the loss (i.e. negative log-likelihood) for the matching
    network, on the test/query points.

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        A tensor containing the logits of each query belonging to each class. This tensor has shape
        `(batch_size, num_classes, num_queries)`.

    labels : `torch.LongTensor` instance
        A tensor containing the labels of the query points. This tensor has
        shape `(batch_size, num_queries)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    (batch_size, num_classes, num_queries) = logits.shape
    logits = logits.permute(0, 2, 1).view(
        batch_size * num_queries, num_classes
    )
    labels = labels.view(-1)
    return F.cross_entropy(logits, labels)


def get_matching_accuracy(logits, labels):
    """Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        A tensor containing the logits of each query belonging to each class. This tensor has shape
        `(batch_size, num_classes, num_queries)`.
    labels : `torch.LongTensor` instance
        A tensor containing the labels of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    _, predictions = torch.max(logits, dim=1)
    return torch.mean(predictions.eq(labels).float())


def inner_gaussian_product(means, precisions, labels, num_classes):
    """Compute the product of n Gaussians for each class (where n can vary by class) from their means and precisions.
    Parameters
    ----------
    means : `torch.FloatTensor` instance
        A tensor containing the means of the Gaussian embeddings. This tensor has shape
        `(batch_size, num_examples, embedding_size)`.
    precisions : `torch.FloatTensor` instance
        A tensor containing the precisions of the Gaussian embeddings. This tensor has shape
        `(batch_size, num_examples, embedding_size)`.
    Returns
    -------
    product_mean : `torch.FloatTensor` instance
        A tensor containing the mean of the resulting product Gaussian. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    product_precision : `torch.FloatTensor` instance
        A tensor containing the precision of resulting product Gaussians. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    log_product_normalisation: `torch.FloatTensor` instance
        A tensor containing the log of the normalisation of resulting product Gaussians. This tensor has shape
        `(batch_size, num_classes)`.
    """
    assert means.shape == precisions.shape
    batch_size, num_examples, embedding_size = means.shape

    num_samples = get_num_samples(labels, num_classes, dtype=means.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(
        num_samples, torch.ones_like(num_samples)
    )  # Backup for testing only, always >= 1-shot in practice

    indices = labels.unsqueeze(-1).expand_as(means)

    # NOTE: If this approach doesn't work well, try first normalising precisions by number of samples with:
    # precisions.div_(num_samples)
    product_precision = precisions.new_zeros(
        (batch_size, num_classes, embedding_size)
    )
    product_precision.scatter_add_(1, indices, precisions)

    product_mean = means.new_zeros((batch_size, num_classes, embedding_size))
    product_mean = torch.reciprocal(
        product_precision
    ) * product_mean.scatter_add_(1, indices, precisions * means)

    product_normalisation_exponent = means.new_zeros(
        (batch_size, num_classes, embedding_size)
    )
    product_normalisation_exponent = 0.5 * (
        product_precision * torch.square(product_mean)
        - product_normalisation_exponent.scatter_add_(
            1, indices, precisions * torch.square(means)
        )
    )

    log_product_normalisation = means.new_zeros(
        (batch_size, num_classes, embedding_size)
    )
    log_product_normalisation = (
        (0.5 * (1 - num_samples))
        * torch.log(torch.ones_like(num_samples) * (2 * math.pi))
        + 0.5
        * (
            log_product_normalisation.scatter_add_(
                1, indices, torch.log(precisions)
            )
            - torch.log(product_precision)
        )
        + product_normalisation_exponent
    )

    log_product_normalisation = log_product_normalisation.sum(dim=-1)

    return (
        product_mean,
        product_precision,
        log_product_normalisation,
    )


def outer_gaussian_product(x_mean, x_precision, y_mean, y_precision):
    """
    Computes all Gaussian product pairs between Gaussian x and y.
    Args:
        x_mean : `torch.FloatTensor` instance
            A tensor containing the means of the query Gaussians. This tensor has shape
            `(batch_size, num_query_examples, embedding_size)`.
        x_precision : `torch.FloatTensor` instance
            A tensor containing the precisions of the query Gaussians. This tensor has shape
            `(batch_size, num_query_examples, embedding_size)`.
        y_mean : `torch.FloatTensor` instance
            A tensor containing the means of the proto Gaussians. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        y_precision : `torch.FloatTensor` instance
            A tensor containing the precisions of the proto Gaussians. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
    Returns:
    product_mean : `torch.FloatTensor` instance
        A tensor containing the mean of the resulting product Gaussian. This tensor has shape
        `(batch_size, num_classes, num_query_examples, embedding_size)`.
    product_precision : `torch.FloatTensor` instance
        A tensor containing the precision of resulting product Gaussians. This tensor has shape
        `(batch_size, num_classes, num_query_examples, embedding_size)`.
    log_product_normalisation: `torch.FloatTensor` instance
        A tensor containing the log of the normalisation of resulting product Gaussians. This tensor has shape
        `(batch_size, num_classes, num_query_examples)`.
    """

    assert x_mean.shape == x_precision.shape
    assert y_mean.shape == y_precision.shape
    (batch_size, num_query_examples, embedding_size) = x_mean.shape
    num_classes = y_mean.size(1)
    assert x_mean.size(0) == y_mean.size(0)
    assert x_mean.size(2) == y_mean.size(2)

    x_mean = x_mean.unsqueeze(1).expand(
        batch_size, num_classes, num_query_examples, embedding_size
    )
    x_precision = x_precision.unsqueeze(1).expand(
        batch_size, num_classes, num_query_examples, embedding_size
    )
    y_mean = y_mean.unsqueeze(2).expand(
        batch_size, num_classes, num_query_examples, embedding_size
    )
    y_precision = y_precision.unsqueeze(2).expand(
        batch_size, num_classes, num_query_examples, embedding_size
    )

    product_precision = x_precision + y_precision
    product_mean = torch.reciprocal(product_precision) * (
        x_precision * x_mean + y_precision * y_mean
    )
    product_normalisation_exponent = 0.5 * (
        product_precision * torch.square(product_mean)
        - x_precision * torch.square(x_mean)
        - y_precision * torch.square(y_mean)
    )
    log_product_normalisation = (
        -0.5
        * torch.log(
            torch.ones_like(product_normalisation_exponent) * (2 * math.pi)
        )
        + 0.5
        * (
            torch.log(x_precision)
            + torch.log(y_precision)
            - torch.log(product_precision)
        )
        + product_normalisation_exponent
    ).sum(dim=-1)
    return product_mean, product_precision, log_product_normalisation


def replace_with_counts(labels):
    target_counts = torch.zeros_like(labels)
    unique_labels, counts = labels.unique(return_counts=True)
    for target, count in zip(unique_labels, counts):
        target_counts[labels == target] = count
    return target_counts
