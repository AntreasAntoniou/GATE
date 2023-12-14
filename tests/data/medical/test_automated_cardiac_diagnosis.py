import os

import torch
import wandb
from tqdm import tqdm

from gate.boilerplate.wandb_utils import visualize_volume
from gate.data.medical.segmentation.automated_cardiac_diagnosis import (
    build_dataset,
    build_gate_dataset,
)

# You can set this to the path where you have the dataset stored on your machine
DATASET_PATH = os.environ.get("PYTEST_DIR")


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


def test_build_gate_visualize_dataset(visualize: bool = False):
    wandb.init(project="gate_visualization_pytest")
    gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    train_loader = torch.utils.data.DataLoader(
        gate_dataset["train"], batch_size=1, shuffle=False, num_workers=16
    )
    val_loader = torch.utils.data.DataLoader(
        gate_dataset["val"], batch_size=1, shuffle=False, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        gate_dataset["test"], batch_size=1, shuffle=False, num_workers=16
    )

    with tqdm(total=200, smoothing=0.0) as pbar:
        for idx, item in enumerate(train_loader):
            print(list(item.keys()))
            assert item["image"] is not None, "Image should not be None"
            assert item["labels"] is not None, "Label should not be None"
            if visualize:
                wandb.log(visualize_volume(item, prefix=f"acdc/train"))
            pbar.update(1)
            if idx > 200:
                break
    with tqdm(total=200, smoothing=0.0) as pbar:
        for idx, item in enumerate(val_loader):
            print(list(item.keys()))
            assert item["image"] is not None, "Image should not be None"
            assert item["labels"] is not None, "Label should not be None"
            if visualize:
                wandb.log(visualize_volume(item, prefix=f"acdc/val"))
            pbar.update(1)
            if idx > 200:
                break
    with tqdm(total=200, smoothing=0.0) as pbar:
        for idx, item in enumerate(test_loader):
            print(list(item.keys()))
            assert item["image"] is not None, "Image should not be None"
            assert item["labels"] is not None, "Label should not be None"
            if visualize:
                wandb.log(visualize_volume(item, prefix=f"acdc/test"))
            pbar.update(1)
            if idx > 200:
                break


if __name__ == "__main__":
    test_build_gate_visualize_dataset(visualize=False)
