import os

import pytest
import wandb
from gate.boilerplate.utils import log_wandb_3d_volumes_and_masks

from gate.data.medical.segmentation.medical_decathlon import (
    build_dataset,
    build_gate_dataset,
)


def test_build_dataset():
    # Test if the function returns the correct dataset split

    train_set = build_dataset(
        set_name="train", data_dir=os.environ.get("PYTEST_DIR")
    )
    assert train_set is not None, "Train set should not be None"

    val_set = build_dataset(
        set_name="val", data_dir=os.environ.get("PYTEST_DIR")
    )
    assert val_set is not None, "Validation set should not be None"

    test_set = build_dataset(
        set_name="test", data_dir=os.environ.get("PYTEST_DIR")
    )
    assert test_set is not None, "Test set should not be None"


def test_build_gate_dataset():
    # Test if the function returns the correct dataset split

    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        break


def test_build_gate_visualize_dataset():
    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"

        input_volumes = item["image"].unsqueeze(2)
        predicted_volumes = item["labels"].float()
        label_volumes = item["labels"].float()

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
            project="mri-visualization", name="mri-visualization-test"
        )

        # Visualize the data
        log_wandb_3d_volumes_and_masks(
            input_volumes, predicted_volumes.long(), label_volumes.long()
        )

        # Finish the run
        run.finish()
