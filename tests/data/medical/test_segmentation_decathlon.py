import os
from asyncio import Task
from dataclasses import dataclass
from enum import Enum

import pytest
import torch.nn.functional as F
import torchvision.transforms as T

import wandb
from gate.boilerplate.wandb_utils import log_wandb_3d_volumes_and_masks
from gate.data.medical.segmentation.medical_decathlon import (
    build_dataset,
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


def visualize_volume(item, prefix: str):
    input_volumes = item["image"].float()
    predicted_volumes = item["labels"].float()
    label_volumes = item["labels"].float()

    # predicted_volumes[predicted_volumes == -1] = 10
    # label_volumes[label_volumes == -1] = 10

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

    target_size = 384
    # Visualize the data
    return log_wandb_3d_volumes_and_masks(
        F.interpolate(
            input_volumes.view(
                -1,
                input_volumes.shape[-3],
                input_volumes.shape[-2],
                input_volumes.shape[-1],
            ),
            size=(target_size, target_size),
            mode="bicubic",
        ).view(*input_volumes.shape[:-2] + (target_size, target_size)),
        F.interpolate(
            predicted_volumes.view(
                -1,
                predicted_volumes.shape[-3],
                predicted_volumes.shape[-2],
                predicted_volumes.shape[-1],
            ),
            size=(target_size, target_size),
            mode="nearest-exact",
        )
        .view(*predicted_volumes.shape[:-2] + (target_size, target_size))
        .long(),
        F.interpolate(
            label_volumes.view(
                -1,
                predicted_volumes.shape[-3],
                predicted_volumes.shape[-2],
                predicted_volumes.shape[-1],
            ),
            size=(target_size, target_size),
            mode="nearest-exact",
        )
        .view(*predicted_volumes.shape[:-2] + (target_size, target_size))
        .long(),
        prefix=prefix,
    )


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
    wandb.init(project="gate_visualization_pytest")
    task_name = gate_dataset_class.__name__
    print(f"Testing {task_name}")
    gate_dataset = gate_dataset_class(data_dir=os.environ.get("PYTEST_DIR"))
    assert gate_dataset["train"] is not None, "Train set should not be None"
    assert gate_dataset["val"] is not None, "Validation set should not be None"
    assert gate_dataset["test"] is not None, "Test set should not be None"

    for item in gate_dataset["train"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        wandb.log(visualize_volume(item, prefix=f"{task_name}/train"))
        break

    for item in gate_dataset["val"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        wandb.log(visualize_volume(item, prefix=f"{task_name}/val"))
        break

    for item in gate_dataset["test"]:
        print(list(item.keys()))
        assert item["image"] is not None, "Image should not be None"
        assert item["labels"] is not None, "Label should not be None"
        wandb.log(visualize_volume(item, prefix=f"{task_name}/test"))
        break
