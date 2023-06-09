import pytest
import torch

from gate.metrics.segmentation import fast_miou, fast_miou_numpy


def test_fast_miou():
    # Case 1: Perfect prediction
    logits = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[0, 1], [1, 0]]]])  # shape: (1, 1, 2, 2)
    assert fast_miou(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 2: Completely wrong prediction
    logits = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[1, 0], [0, 1]]]])  # shape: (1, 1, 2, 2)
    assert fast_miou(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 3: Half correct prediction
    logits = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.6, 0.4], [0.4, 0.6]]]]
    )  # probabilities
    labels = torch.tensor([[[[0, 1], [1, 1]]]])  # class labels
    assert fast_miou(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 4: All zeros
    logits = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[0, 0], [0, 0]]]])  # shape: (1, 1, 2, 2)
    assert fast_miou(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )

    # Case 5: All ones
    logits = torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]]
    )  # shape: (1, 2, 2, 2)
    labels = torch.tensor([[[[1, 1], [1, 1]]]])  # shape: (1, 1, 2, 2)
    assert fast_miou(logits, labels)["mean_iou"].item() == pytest.approx(
        fast_miou_numpy(logits, labels)["mean_iou"].item()
    )
