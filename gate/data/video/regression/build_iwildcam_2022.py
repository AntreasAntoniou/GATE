import os
from pathlib import Path
from typing import Any

from accelerate import Accelerator

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.transforms.video import BaseVideoTransform, TrainVideoTransform
from gate.data.video.regression.iwildcam_2022 import prepare_iwildcam_2022
from gate.data.video.utils.loader.iwildcam2022_dataset import (
    IWildCam2022Dataset,
)


def build_dataset(
    data_dir: str | Path,
    sets_to_include: list[str] | None = None,
    ensure_installed=True,
    accelerator: Accelerator | None = None,
):
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if data_dir.name != "iwildcam2022":
        data_dir = data_dir / "iwildcam2022"

    if sets_to_include is None:
        sets_to_include = ["train", "val", "test"]

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            prepare_iwildcam_2022(data_dir)

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        split_path = data_dir / "splits" / f"{set_name}.txt"

        if set_name not in ["train", "val", "test"]:
            raise ValueError(f"Unknown set_name: {set_name}")

        data = IWildCam2022Dataset(data_dir, split_path=split_path)
        dataset[set_name] = data

    return dataset


def key_selector(input_dict):
    return {"video": input_dict["video"], "labels": input_dict["labels"]}


@configurable(
    group="dataset",
    name="iwildcam_2022",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: str | Path,
    transforms: Any | None = None,
    num_classes: int = 1,
    scale_factor=(448, 448),
    crop_size=(224, 224),
    flip_prob=0.5,
    rotation_angles=[0, 90, 180, 270],
    brightness=0.2,
    contrast=0.2,
    jitter_strength=0.1,
) -> dict[str, GATEDataset]:
    datasets = build_dataset(
        data_dir=data_dir,
        sets_to_include=["train", "val", "test"],
    )

    dataset_dict = {}

    if "train" in datasets:
        dataset_dict["train"] = GATEDataset(
            dataset=datasets["train"],
            infinite_sampling=True,
            transforms=[
                key_selector,
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
                key_selector,
                BaseVideoTransform(scale_factor=crop_size),
                transforms,
            ],
        )

    if "test" in datasets:
        dataset_dict["test"] = GATEDataset(
            dataset=datasets["test"],
            infinite_sampling=False,
            transforms=[
                key_selector,
                BaseVideoTransform(scale_factor=crop_size),
                transforms,
            ],
        )

    return dataset_dict
