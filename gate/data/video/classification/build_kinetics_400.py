import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from huggingface_hub import snapshot_download

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.transforms.video import BaseVideoTransform, TrainVideoTransform
from gate.data.video.classification.build_gulp_sparsesample import DatasetNames
from gate.data.video.classification.kinetics_400 import prepare_kinetics_400
from gate.data.video.utils.loader.decord_sparsesample_dataset import (
    DecordSparsesampleDataset,
)


def build_dataset(
    data_dir: str | Path,
    sets_to_include=None,
    video_height=224,
    video_width=224,
    ensure_installed=True,
    accelerator: Accelerator | None = None,
):
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if data_dir.name != "kinetics-dataset":
        data_dir = data_dir / "kinetics-dataset"

    if sets_to_include is None:
        sets_to_include = ["train", "val", "test"]

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            prepare_kinetics_400(
                download_dataset_rootdir=data_dir / "downloads",
                extract_dataset_rootdir=data_dir,
            )
            snapshot_download(
                repo_id="kiyoonkim/kinetics-400-splits",
                repo_type="dataset",
                resume_download=True,
                local_dir=data_dir / "k400",
                cache_dir=cache_dir,
                max_workers=mp.cpu_count(),
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        videos_dir = data_dir / "kinetics-dataset" / "k400" / "videos"

        if set_name == "train":
            csv_path = data_dir / "k400" / "splits_decord_videos" / "train.csv"
            videos_dir = videos_dir / "train"
        elif set_name == "val":
            csv_path = data_dir / "k400" / "splits_decord_videos" / "val.csv"
            videos_dir = videos_dir / "val"
        elif set_name == "test":
            csv_path = data_dir / "k400" / "splits_decord_videos" / "test.csv"
            videos_dir = videos_dir / "test"
        else:
            raise ValueError(f"Unknown set_name: {set_name}")

        data = DecordSparsesampleDataset(
            csv_path,
            input_frame_length,
            video_height=video_height,
            video_width=video_width,
            sample_index_code="pyvideoai",
            path_prefix=videos_dir,
        )
        dataset[set_name] = data

    return dataset


@configurable(
    group="dataset",
    name=DatasetNames.KINETICS_400.value,
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
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
    datasets = build_dataset(
        data_dir=data_dir, sets_to_include=["train", "val", "test"]
    )

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
