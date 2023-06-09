# Add this import at the top of the file
import os

import pytest
from torch.utils.data import DataLoader

from gate.data.core import collate_fn_with_token_pad
from gate.data.image.visual_question_answering.ok_vqa import (
    build_dataset as build_ok_vqa_dataset,
)
from gate.data.image.visual_question_answering.ok_vqa import (
    build_gate_dataset as build_ok_vqa_gate_dataset,
)
from gate.models.task_specific_models.visual_question_answering.clip import (
    build_model,
)


def test_collate_fn_with_token_pad():
    # Test if the function correctly pads the tensors
    vqa_model = build_model()
    train_set = build_ok_vqa_gate_dataset(
        data_dir=os.environ.get("PYTEST_DIR"), transforms=vqa_model.transform
    )["train"]
    assert train_set is not None, "Train set should not be None"

    for batch in train_set:
        print(batch)
        break
    # Create a DataLoader with the custom collate function
    train_loader = DataLoader(
        train_set,
        batch_size=16,
        collate_fn=collate_fn_with_token_pad,
        shuffle=True,
    )

    for batch in train_loader:
        for key, value in batch.items():
            print(f"{key}: {value}")
        break
