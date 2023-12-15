# test_mscoco_10k.py
import os

import pytest
import torch
from tqdm import tqdm

from gate.boilerplate.wandb_utils import visualize_volume
from gate.data.image.segmentation.coco_164k import (
    build_dataset,
    build_gate_dataset,
)


def test_build_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_dataset("train", data_dir=os.environ.get("PYTEST_DIR"))
    assert train_set is not None, "Train set should not be None"

    val_set = build_dataset("val", data_dir=os.environ.get("PYTEST_DIR"))
    assert val_set is not None, "Validation set should not be None"

    test_set = build_dataset("test", data_dir=os.environ.get("PYTEST_DIR"))
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_dataset(
            "invalid_set_name", data_dir=os.environ.get("PYTEST_DIR")
        )


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    print(gate_dataset["train"][0])


def test_build_gate_visualize_dataset():
    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item, name="training-visualization/coco_164k")
        break

    for item in gate_dataset["val"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item, name="validation-visualization/coco_164k")
        break

    for item in gate_dataset["test"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item, name="test-visualization/coco_164k")
        break

    from torch.utils.data import DataLoader

    mixed_dataset = torch.utils.data.ConcatDataset(
        [gate_dataset["train"], gate_dataset["val"], gate_dataset["test"]]
    )

    mixed_dataloader = DataLoader(
        mixed_dataset, batch_size=256, num_workers=32, shuffle=True
    )

    unique_labels = set()
    for item in tqdm(mixed_dataloader):
        unique_labels.update(
            item["labels"].unique().tolist()
        )  # Convert tensor to list and update the set

    print(f"len {len(unique_labels)}")
    print(unique_labels)
