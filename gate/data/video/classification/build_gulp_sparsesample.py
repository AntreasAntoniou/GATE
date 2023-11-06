# Refactoring the code for readability, maintainability, and efficiency.
# Adding docstrings and comments for better understanding.

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from accelerate import Accelerator
from huggingface_hub import snapshot_download
from torch.utils.data import random_split

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.transforms.video import BaseVideoTransform, TrainVideoTransform
from gate.data.video.classification import DatasetNames
from gate.data.video.utils.loader.gulp_sparsesample_dataset import (
    GulpSparsesampleDataset,
)


@dataclass
class DatasetConfig:
    dataset_name: str
    set_name: str
    split_num: int
    data_dir: Path = field(default_factory=Path)

    def __post_init__(self):
        self.input_frame_length = 8
        self.gulp_dir_path = self.data_dir / "gulp_rgb"

        if self.dataset_name == "hmdb51-gulprgb":
            self._handle_hmdb51_gulprgb()

        elif self.dataset_name == "ucf-101-gulprgb":
            self._handle_ucf_101_gulprgb()

        elif self.dataset_name == "epic-kitchens-100-gulprgb":
            self._handle_epic_kitchens_100_gulprgb()

        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

    def _handle_hmdb51_gulprgb(self):
        self.mode = self.set_name
        self.csv_path = (
            self.data_dir
            / "splits_gulp_rgb"
            / f"{self.set_name}{self.split_num}.csv"
        )

    def _handle_ucf_101_gulprgb(self):
        self.mode = self.set_name
        self.csv_path = (
            self.data_dir
            / "splits_gulp_rgb"
            / f"{self.set_name}list{self.split_num:02d}.txt"
        )

    def _handle_epic_kitchens_100_gulprgb(self):
        if self.set_name == "train":
            self.gulp_dir_path = self.gulp_dir_path / "train"
            self.mode = "train"
            self.csv_path = (
                self.data_dir
                / "verbnoun_splits_gulp_rgb"
                / "train_partial90.csv"
            )
        elif self.set_name == "val":
            self.gulp_dir_path = self.gulp_dir_path / "train"
            self.mode = "test"
            self.csv_path = (
                self.data_dir
                / "verbnoun_splits_gulp_rgb"
                / "train_partial10.csv"
            )
        elif self.set_name == "test":
            self.gulp_dir_path = self.gulp_dir_path / "val"
            self.mode = "test"
            self.csv_path = (
                self.data_dir / "verbnoun_splits_gulp_rgb" / "val.csv"
            )
        else:
            raise ValueError(f"Unknown set_name: {self.set_name}")


def download_dataset(
    dataset_name: str,
    data_dir: Path,
    cache_dir: str,
    accelerator: Optional[Accelerator] = None,
) -> None:
    """
    Downloads the dataset snapshot from huggingface_hub.

    Parameters:
        dataset_name: The name of the dataset to download.
        data_dir: The directory where the dataset should be downloaded.
        cache_dir: The directory to use for caching downloaded files.
        accelerator: An optional Hugging Face `Accelerator` object for distributed training.
    """
    if accelerator is None or accelerator.is_local_main_process:
        snapshot_download(
            repo_id=f"kiyoonkim/{dataset_name}",
            repo_type="dataset",
            resume_download=True,
            local_dir=data_dir,
            cache_dir=cache_dir,
            max_workers=mp.cpu_count(),
        )
    if accelerator is not None:
        accelerator.wait_for_everyone()


def build_specific_dataset(
    dataset_name: str,
    data_dir: Path,
) -> Dict[str, Any]:
    """
    Build a specific dataset using the given dataset class and parameters.

    Parameters:
        dataset_name: The name of the dataset.
        data_dir: The directory where the dataset resides.

    Returns:
        A dictionary containing the datasets for the specified splits.
    """
    train_config = DatasetConfig(
        dataset_name=dataset_name,
        set_name="train",
        split_num=1,
        data_dir=data_dir,
    )
    val_config = DatasetConfig(
        dataset_name=dataset_name,
        set_name="val",
        split_num=1,
        data_dir=data_dir,
    )
    test_config = DatasetConfig(
        dataset_name=dataset_name,
        set_name="test",
        split_num=1,
        data_dir=data_dir,
    )
    dataset = {}
    need_subsets = (
        False  # Flag to check if subsets from the training set are needed
    )
    train_subset = (
        val_subset
    ) = test_subset = None  # Initialize subsets to None

    try:
        dataset["train"] = GulpSparsesampleDataset(
            csv_file=train_config.csv_path,
            mode=train_config.mode,
            num_frames=train_config.input_frame_length,
            gulp_dir_path=train_config.gulp_dir_path,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not generate 'train' set: {e}. Halting execution."
        ) from e

    try:
        dataset["val"] = GulpSparsesampleDataset(
            csv_file=val_config.csv_path,
            mode=val_config.mode,
            num_frames=val_config.input_frame_length,
            gulp_dir_path=val_config.gulp_dir_path,
        )
    except Exception as e:
        print(f"Could not generate 'val' set: {e}")
        need_subsets = True

    try:
        dataset["test"] = GulpSparsesampleDataset(
            csv_file=test_config.csv_path,
            mode=test_config.mode,
            num_frames=test_config.input_frame_length,
            gulp_dir_path=test_config.gulp_dir_path,
        )
    except Exception as e:
        print(f"Could not generate 'test' set: {e}")
        need_subsets = True

    if need_subsets:
        train_size = len(dataset["train"])
        subset_size = int(train_size * 0.1)
        remaining = (
            train_size - 2 * subset_size
        )  # Two subsets: one for 'val', one for 'test'

        # Split the original training dataset
        train_subset, val_subset, test_subset = random_split(
            dataset["train"], [remaining, subset_size, subset_size]
        )

    if "val" not in dataset and val_subset:
        dataset["val"] = val_subset

    if "test" not in dataset and test_subset:
        dataset["test"] = test_subset

    # Only update the training set if subsets were actually used
    if need_subsets:
        dataset["train"] = train_subset

    return dataset


def build_dataset(
    dataset_name: str,
    data_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Build a dataset for Gulp-formatted video data.

    Parameters:
        dataset_name: The name of the dataset (e.g., 'hmdb51-gulprgb').
        data_dir: The directory where the dataset resides.
        sets_to_include: List of dataset splits to include (e.g., ['train', 'test']).
        **kwargs: Additional keyword arguments to pass to the dataset class constructor.

    Returns:
        A dictionary containing the datasets for the specified splits.
    """
    data_dir = Path(data_dir)
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    if cache_dir is None:
        raise ValueError("Environment variable HF_CACHE_DIR must be set.")

    data_dir = data_dir / dataset_name

    # Ensure the dataset is downloaded
    download_dataset(dataset_name, data_dir, cache_dir)

    return build_specific_dataset(
        dataset_name,
        data_dir,
    )


def key_selector(input_dict):
    return {"video": input_dict["video"], "labels": input_dict["labels"]}


def build_gate_dataset(
    dataset_name: str,
    data_dir: Union[str, Path],
    transforms: Optional[Any] = None,
) -> Dict[str, GATEDataset]:
    """
    Build a GATE dataset for video data.

    Parameters:
        dataset_name: The name of the dataset (e.g., 'hmdb51-gulprgb').
        data_dir: The directory where the dataset resides.
        transforms: Optional data transformations to apply.
        **kwargs: Additional keyword arguments to use for data preprocessing.

    Returns:
        A dictionary containing GATEDataset instances for the specified splits.
    """
    datasets = build_dataset(dataset_name, data_dir)
    dataset_dict = {}

    if "train" in datasets:
        dataset_dict["train"] = GATEDataset(
            dataset=datasets["train"],
            infinite_sampling=True,
            transforms=[key_selector, TrainVideoTransform(), transforms],
        )

    if "val" in datasets:
        dataset_dict["val"] = GATEDataset(
            dataset=datasets["val"],
            infinite_sampling=False,
            transforms=[key_selector, BaseVideoTransform(), transforms],
        )

    if "test" in datasets:
        dataset_dict["test"] = GATEDataset(
            dataset=datasets["test"],
            infinite_sampling=False,
            transforms=[key_selector, BaseVideoTransform(), transforms],
        )

    return dataset_dict


@configurable(
    group="dataset",
    name=DatasetNames.HMDB51_GULPRGB.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_hmdb51_gate_dataset(
    data_dir: str,
    transforms: Optional[Any] = None,
    num_classes=51,
) -> dict:
    return build_gate_dataset(
        dataset_name=DatasetNames.HMDB51_GULPRGB.value,
        data_dir=data_dir,
        transforms=transforms,
    )


@configurable(
    group="dataset",
    name=DatasetNames.UCF_101_GULPRGB.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_ucf_101_gate_dataset(
    data_dir: str,
    transforms: Optional[Any] = None,
    num_classes=101,
) -> dict:
    return build_gate_dataset(
        dataset_name=DatasetNames.UCF_101_GULPRGB.value,
        data_dir=data_dir,
        transforms=transforms,
    )
