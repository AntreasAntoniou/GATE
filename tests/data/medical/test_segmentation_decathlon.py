import os
from enum import Enum

import pytest
import torch
import wandb
from tqdm import tqdm

from gate.boilerplate.wandb_utils import visualize_volume
from gate.data.medical.segmentation.medical_decathlon import (
    build_gate_md_brain_tumour,
    build_gate_md_colon,
    build_gate_md_heart,
    build_gate_md_hepatic_vessel,
    build_gate_md_hippocampus,
    build_gate_md_liver,
    build_gate_md_lung,
    build_gate_md_pancreas,
    build_gate_md_prostate,
    build_gate_md_spleen,
)


class TaskOptions(Enum):
    BrainTumour: str = "Task01_BrainTumour"
    Heart: str = "Task02_Heart"
    Liver: str = "Task03_Liver"
    Hippocampus: str = "Task04_Hippocampus"
    Prostate: str = "Task05_Prostate"
    Lung: str = "Task06_Lung"
    Pancreas: str = "Task07_Pancreas"
    HepaticVessel: str = "Task08_HepaticVessel"
    Spleen: str = "Task09_Spleen"
    Colon: str = "Task10_Colon"


# def test_build_dataset():
#     # Test if the function returns the correct dataset split

#     train_set = build_dataset(
#         set_name="train", data_dir=os.environ.get("PYTEST_DIR")
#     )
#     assert train_set is not None, "Train set should not be None"

#     val_set = build_dataset(
#         set_name="val", data_dir=os.environ.get("PYTEST_DIR")
#     )
#     assert val_set is not None, "Validation set should not be None"

#     test_set = build_dataset(
#         set_name="test", data_dir=os.environ.get("PYTEST_DIR")
#     )
#     assert test_set is not None, "Test set should not be None"


# def test_build_gate_dataset():
#     # Test if the function returns the correct dataset split

#     gate_dataset = build_gate_dataset(data_dir=os.environ.get("PYTEST_DIR"))
#     assert gate_dataset["train"] is not None, "Train set should not be None"
#     assert gate_dataset["val"] is not None, "Validation set should not be None"
#     assert gate_dataset["test"] is not None, "Test set should not be None"

#     for item in gate_dataset["train"]:
#         print(list(item.keys()))
#         assert item["image"] is not None, "Image should not be None"
#         assert item["labels"] is not None, "Label should not be None"
#         break


@pytest.mark.parametrize(
    "gate_dataset_class",
    [
        build_gate_md_brain_tumour,
        build_gate_md_heart,
        build_gate_md_liver,
        build_gate_md_hepatic_vessel,
        build_gate_md_hippocampus,
        build_gate_md_lung,
        build_gate_md_pancreas,
        build_gate_md_prostate,
        build_gate_md_spleen,
        build_gate_md_colon,
    ],
)
def test_build_gate_visualize_dataset(gate_dataset_class):
    visualize = False
    wandb.init(project="gate_visualization_pytest")
    task_name = gate_dataset_class.__name__
    print(f"Testing {task_name}")
    gate_dataset = gate_dataset_class(data_dir=os.environ.get("PYTEST_DIR"))

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

    assert len(train_loader) > 0, "Train loader should not be empty"
    with tqdm(total=200, smoothing=0.0) as pbar:
        for idx, item in enumerate(train_loader):
            print(list(item.keys()))
            assert item["image"] is not None, "Image should not be None"
            assert item["labels"] is not None, "Label should not be None"
            if visualize:
                wandb.log(visualize_volume(item, prefix=f"{task_name}/train"))
            pbar.update(1)
            if idx > 200:
                break

    assert len(val_loader) > 0, "Val loader should not be empty"
    with tqdm(total=200, smoothing=0.0) as pbar:
        for idx, item in enumerate(val_loader):
            print(list(item.keys()))
            assert item["image"] is not None, "Image should not be None"
            assert item["labels"] is not None, "Label should not be None"
            if visualize:
                wandb.log(visualize_volume(item, prefix=f"{task_name}/val"))
            pbar.update(1)
            if idx > 200:
                break

    assert len(test_loader) > 0, "Test loader should not be empty"
    with tqdm(total=200, smoothing=0.0) as pbar:
        for idx, item in enumerate(test_loader):
            print(list(item.keys()))
            assert item["image"] is not None, "Image should not be None"
            assert item["labels"] is not None, "Label should not be None"
            if visualize:
                wandb.log(visualize_volume(item, prefix=f"{task_name}/test"))
            pbar.update(1)
            if idx > 200:
                break
