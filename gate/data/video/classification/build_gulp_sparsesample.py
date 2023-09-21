import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from huggingface_hub import snapshot_download

from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask
from gate.data.transforms.video_transforms import (
    BaseVideoTransform,
    TrainVideoTransform,
)

from ..loader.gulp_sparsesample_dataset import GulpSparsesampleDataset
from ..loader.gulp_sparsesample_squeezed_dataset import (
    GulpSparsesampleSqueezedDataset,
)


def build_dataset(
    dataset_name: str,
    data_dir: str | Path,
    sets_to_include=None,
    train_jitter_min=224,
    train_jitter_max=336,
    train_horizontal_flip=True,
    test_scale=256,
    test_num_spatial_crops=1,
    split_num: int = 1,
    crop_size=224,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    normalise=True,
    bgr=False,
    ensure_installed=True,
    accelerator: Accelerator | None = None,
):
    assert dataset_name in [
        "hmdb51-gulprgb",
        "ucf-101-gulprgb",
        "epic-kitchens-100-gulprgb",
    ]
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if dataset_name == "hmdb51-gulprgb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "hmdb51":
            data_dir = data_dir / "hmdb51"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "ucf-101-gulprgb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "ucf101":
            data_dir = data_dir / "ucf101"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "epic-kitchens-100-gulprgb":
        assert split_num == 1
        if data_dir.name != "epic-kitchens-100":
            data_dir = data_dir / "epic-kitchens-100"
        if sets_to_include is None:
            sets_to_include = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    assert sets_to_include is not None

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            snapshot_download(
                repo_id=f"kiyoonkim/{dataset_name}",
                repo_type="dataset",
                resume_download=True,
                local_dir=data_dir,
                cache_dir=cache_dir,
                max_workers=mp.cpu_count(),
                # allow_patterns="splits_gulp_rgb/*",
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        gulp_dir_path = data_dir / "gulp_rgb"

        if dataset_name == "hmdb51-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = (
                    data_dir / "splits_gulp_rgb" / f"train{split_num}.csv"
                )
            elif set_name == "test":
                mode = "test"
                csv_path = (
                    data_dir / "splits_gulp_rgb" / f"test{split_num}.csv"
                )
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        elif dataset_name == "ucf-101-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = (
                    data_dir
                    / "splits_gulp_rgb"
                    / f"trainlist{split_num:02d}.txt"
                )
            elif set_name == "test":
                mode = "test"
                csv_path = (
                    data_dir
                    / "splits_gulp_rgb"
                    / f"testlist{split_num:02d}.txt"
                )
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        elif dataset_name == "epic-kitchens-100-gulprgb":
            if set_name == "train":
                gulp_dir_path = gulp_dir_path / "train"
                mode = "train"
                csv_path = (
                    data_dir
                    / "verbnoun_splits_gulp_rgb"
                    / "train_partial90.csv"
                )
            elif set_name == "val":
                gulp_dir_path = gulp_dir_path / "train"
                mode = "test"
                csv_path = (
                    data_dir
                    / "verbnoun_splits_gulp_rgb"
                    / "train_partial10.csv"
                )
            elif set_name == "test":
                gulp_dir_path = gulp_dir_path / "val"
                mode = "test"
                csv_path = data_dir / "verbnoun_splits_gulp_rgb" / "val.csv"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        data = GulpSparsesampleDataset(
            csv_path,
            mode,
            input_frame_length,
            gulp_dir_path,
            train_jitter_min=train_jitter_min,
            train_jitter_max=train_jitter_max,
            train_horizontal_flip=train_horizontal_flip,
            test_scale=test_scale,
            test_num_spatial_crops=test_num_spatial_crops,
            crop_size=crop_size,
            mean=mean,
            std=std,
            normalise=normalise,
            bgr=bgr,
            greyscale=False,
            sample_index_code="pyvideoai",
            processing_backend="pil",
            frame_neighbours=1,
            pil_transforms_after=None,
        )
        dataset[set_name] = data

    return dataset


def build_squeezed_gulp_dataset(
    dataset_name: str,
    data_dir: str | Path,
    sets_to_include=None,
    split_num: int = 1,
    size=224,
    data_format="BTCHW",
    ensure_installed=True,
    accelerator: Accelerator | None = None,
):
    assert dataset_name in [
        "hmdb51-gulprgb",
        "ucf-101-gulprgb",
        "epic-kitchens-100-gulprgb",
    ]
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None
    assert data_format in ["BTCHW", "BCTHW"]

    data_dir = Path(data_dir)

    if dataset_name == "hmdb51-gulprgb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "hmdb51":
            data_dir = data_dir / "hmdb51"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "ucf-101-gulprgb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "ucf101":
            data_dir = data_dir / "ucf101"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "epic-kitchens-100-gulprgb":
        assert split_num == 1
        if data_dir.name != "epic-kitchens-100":
            data_dir = data_dir / "epic-kitchens-100"
        if sets_to_include is None:
            sets_to_include = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    assert sets_to_include is not None

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            snapshot_download(
                repo_id=f"kiyoonkim/{dataset_name}",
                repo_type="dataset",
                resume_download=True,
                local_dir=data_dir,
                cache_dir=cache_dir,
                max_workers=mp.cpu_count(),
                # allow_patterns="splits_gulp_rgb/*",
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        gulp_dir_path = data_dir / "gulp_rgb"
        if dataset_name == "hmdb51-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = (
                    data_dir / "splits_gulp_rgb" / f"train{split_num}.csv"
                )
            elif set_name == "test":
                mode = "test"
                csv_path = (
                    data_dir / "splits_gulp_rgb" / f"test{split_num}.csv"
                )
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        elif dataset_name == "ucf-101-gulprgb":
            if set_name == "train":
                mode = "train"
                csv_path = (
                    data_dir
                    / "splits_gulp_rgb"
                    / f"trainlist{split_num:02d}.txt"
                )
            elif set_name == "test":
                mode = "test"
                csv_path = (
                    data_dir
                    / "splits_gulp_rgb"
                    / f"testlist{split_num:02d}.txt"
                )
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        elif dataset_name == "epic-kitchens-100-gulprgb":
            if set_name == "train":
                gulp_dir_path = gulp_dir_path / "train"
                mode = "train"
                csv_path = (
                    data_dir
                    / "verbnoun_splits_gulp_rgb"
                    / "train_partial90.csv"
                )
            elif set_name == "val":
                gulp_dir_path = gulp_dir_path / "train"
                mode = "test"
                csv_path = (
                    data_dir
                    / "verbnoun_splits_gulp_rgb"
                    / "train_partial10.csv"
                )
            elif set_name == "test":
                gulp_dir_path = gulp_dir_path / "val"
                mode = "test"
                csv_path = data_dir / "verbnoun_splits_gulp_rgb" / "val.csv"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        data = GulpSparsesampleSqueezedDataset(
            csv_path,
            mode,
            input_frame_length,
            gulp_dir_path,
            size=size,
            data_format=data_format,
            pil_transforms_after=None,
        )
        dataset[set_name] = data

    return dataset


def build_gate_dataset(
    dataset_name: str,
    data_dir: str | Path,
    transforms: Any | None = None,
    num_classes: int = 101,
    scale_factor=(448, 448),
    crop_size=(224, 224),
    flip_prob=0.5,
    rotation_angles=[0, 90, 180, 270],
    brightness=0.2,
    contrast=0.2,
    jitter_strength=0.1,
) -> dict[str, GATEDataset]:
    datasets = build_dataset(dataset_name=dataset_name, data_dir=data_dir)

    dataset_dict = {}

    if "train" in datasets:
        dataset_dict["train"] = GATEDataset(
            dataset=datasets["train"],
            infinite_sampling=True,
            transforms=[
                TrainVideoTransform(
                    scale_factor=scale_factor,
                    crop_size=crop_size,
                    flip_prob=flip_prob,
                    rotation_angles=rotation_angles,
                    brightness=brightness,
                    contrast=contrast,
                    jitter_strength=jitter_strength,
                ),
                transforms,
            ],
        )

    if "val" in datasets:
        dataset_dict["val"] = GATEDataset(
            dataset=datasets["val"],
            infinite_sampling=False,
            transforms=[
                BaseVideoTransform(scale_factor=crop_size),
                transforms,
            ],
        )

    if "test" in datasets:
        dataset_dict["test"] = GATEDataset(
            dataset=datasets["test"],
            infinite_sampling=False,
            transforms=[
                BaseVideoTransform(scale_factor=crop_size),
                transforms,
            ],
        )

    return dataset_dict
