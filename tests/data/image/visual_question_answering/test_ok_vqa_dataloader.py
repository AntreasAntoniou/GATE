# Add this import at the top of the file
from torch.utils.data import DataLoader
import os

import pytest
from gate.data.core import collate_fn_with_token_pad

from gate.data.image.visual_question_answering.ok_vqa import (
    build_dataset as build_ok_vqa_dataset,
    build_gate_dataset as build_ok_vqa_gate_dataset,
)
from gate.models.visual_question_answering.clip import build_model


def test_collate_fn_with_token_pad():
    # Test if the function correctly pads the tensors
    vqa_model = build_model()
    train_set = build_ok_vqa_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"), transforms=vqa_model.transform
    )["train"]
    assert train_set is not None, "Train set should not be None"

    # Create a DataLoader with the custom collate function
    train_loader = DataLoader(
        train_set, batch_size=2, collate_fn=collate_fn_with_token_pad
    )

    # Iterate over the DataLoader
    for batch in train_loader:
        # Verify that all tensors in a batch have the same size
        for key in batch:
            tensor_sizes = [item.size(0) for item in batch[key]]
            assert all(
                size == tensor_sizes[0] for size in tensor_sizes
            ), f"All tensors in {key} should have the same size"
