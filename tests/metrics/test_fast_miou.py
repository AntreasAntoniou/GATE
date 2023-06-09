import pytest
import torch

from gate.metrics.segmentation import fast_miou


def test_fast_miou():
    # Case 1: Perfect prediction
    logits = torch.tensor([[[[0, 1], [1, 0]]]])
    labels = torch.tensor([[[[0, 1], [1, 0]]]])
    assert pytest.approx(fast_miou(logits, labels).item(), 1.0)

    # Case 2: Completely wrong prediction
    logits = torch.tensor([[[[0, 1], [1, 0]]]])
    labels = torch.tensor([[[[1, 0], [0, 1]]]])
    assert pytest.approx(fast_miou(logits, labels).item(), 0.0)

    # Case 3: Half correct prediction
    logits = torch.tensor([[[[0, 1], [1, 0]]]])
    labels = torch.tensor([[[[0, 1], [1, 1]]]])
    assert pytest.approx(fast_miou(logits, labels).item(), 0.5)

    # Case 4: All zeros
    logits = torch.tensor([[[[0, 0], [0, 0]]]])
    labels = torch.tensor([[[[0, 0], [0, 0]]]])
    assert pytest.approx(fast_miou(logits, labels).item(), 1.0)

    # Case 5: All ones
    logits = torch.tensor([[[[1, 1], [1, 1]]]])
    labels = torch.tensor([[[[1, 1], [1, 1]]]])
    assert pytest.approx(fast_miou(logits, labels).item(), 1.0)
