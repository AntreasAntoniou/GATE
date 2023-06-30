# test_food101_dataset.py
import os
import pathlib

import pytest
import wandb
import torch.nn.functional as F

from gate.boilerplate.utils import log_wandb_3d_volumes_and_masks
from gate.data.image.segmentation.nyu_depth_v2 import (
    build_dataset,
    build_gate_dataset,
)

DATA_DIR = pathlib.Path(os.environ.get("PYTEST_DIR"))


def test_build_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_dataset("train", data_dir=DATA_DIR)
    assert train_set is not None, "Train set should not be None"

    val_set = build_dataset("val", data_dir=DATA_DIR)
    assert val_set is not None, "Validation set should not be None"

    test_set = build_dataset("test", data_dir=DATA_DIR)
    assert test_set is not None, "Test set should not be None"

    # Test if the function raises an error when an invalid set_name is given
    with pytest.raises(KeyError):
        build_dataset("invalid_set_name", data_dir=DATA_DIR)


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=DATA_DIR)
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    print(gate_dataset["train"][0])


def visualize_volume(item):
    input_volumes = item["image"]
    input_volumes = input_volumes.float()
    predicted_volumes = item["labels"].float()
    label_volumes = item["labels"].float()

    predicted_volumes[predicted_volumes == -1] = 10
    label_volumes[label_volumes == -1] = 10

    print(
        f"Input volumes shape: {input_volumes.shape}, dtype: {input_volumes.dtype}, min: {input_volumes.min()}, max: {input_volumes.max()}, mean: {input_volumes.mean()}, std: {input_volumes.std()}"
    )
    print(
        f"Predicted volumes shape: {predicted_volumes.shape}, dtype: {predicted_volumes.dtype}, min: {predicted_volumes.min()}, max: {predicted_volumes.max()}, mean: {predicted_volumes.mean()}, std: {predicted_volumes.std()}"
    )
    print(
        f"Label volumes shape: {label_volumes.shape}, dtype: {label_volumes.dtype}, min: {label_volumes.min()}, max: {label_volumes.max()}, mean: {label_volumes.mean()}, std: {label_volumes.std()}"
    )

    # Start a Weights & Biases run
    run = wandb.init(
        project="gate-visualization", job_type="visualize_dataset"
    )

    # Visualize the data
    log_wandb_3d_volumes_and_masks(
        F.interpolate(
            input_volumes.reshape(-1, input_volumes.shape[-3], 512, 512),
            size=(256, 256),
            mode="bicubic",
        ).reshape(*input_volumes.shape[:-2] + (256, 256)),
        predicted_volumes.long(),
        label_volumes.long(),
    )

    # Finish the run
    run.finish()


def test_build_gate_visualize_dataset():
    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item)
        break

    for item in gate_dataset["val"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item)
        break

    for item in gate_dataset["test"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        visualize_volume(item)
        break
