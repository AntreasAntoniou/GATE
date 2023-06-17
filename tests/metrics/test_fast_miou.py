import pytest
import torch
from rich import print

from gate.metrics.segmentation import miou_metrics, fast_miou_numpy


def test_fast_miou():
    # Case 1: Perfect prediction
    logits = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[0, 1], [1, 0]]]])  # shape: (1, 1, 2, 2)
    metrics = miou_metrics(logits, labels)
    # print(metrics)
    assert metrics["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 2: Completely wrong prediction
    logits = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[1, 0], [0, 1]]]])  # shape: (1, 1, 2, 2)
    assert miou_metrics(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 3: Half correct prediction
    logits = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.6, 0.4], [0.4, 0.6]]]]
    )  # probabilities
    labels = torch.tensor([[[[0, 1], [1, 1]]]])  # class labels
    assert miou_metrics(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 4: All zeros
    logits = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[0, 0], [0, 0]]]])  # shape: (1, 1, 2, 2)
    assert miou_metrics(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 5: All ones
    logits = torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[1, 1], [1, 1]]]])  # shape: (1, 1, 2, 2)
    assert miou_metrics(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 6: large batch with images and image masks
    logits = torch.randn((96, 150, 64, 64))  # shape: (1, 2, 2, 2)
    labels = torch.randint(150, (96, 1, 64, 64))  # shape: (1, 1, 2, 2)
    metrics = miou_metrics(logits, labels)
    print(metrics)
    assert metrics["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )
