import os
from pathlib import Path

from accelerate import Accelerator
from huggingface_hub import snapshot_download

from .gulp.gulp_sparsesample_skeleton_dataset import GulpSparsesampleSkeletonDataset


def build_gulp_skeleton_dataset(
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
        "hmdb-51",
        "ucf-101",
    ]
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if dataset_name == "hmdb-51":
        assert split_num in [1, 2, 3]
        if data_dir.name != "hmdb51":
            data_dir = data_dir / "hmdb51"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "ucf-101":
        assert split_num in [1, 2, 3]
        if data_dir.name != "ucf101":
            data_dir = data_dir / "ucf101"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    assert sets_to_include is not None

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            if dataset_name == "hmdb-51":
                snapshot_download(
                    repo_id="kiyoonkim/hmdb51-gulprgb",
                    repo_type="dataset",
                    resume_download=True,
                    local_dir=data_dir,
                    cache_dir=cache_dir,
                )
                snapshot_download(
                    repo_id="kiyoonkim/hmdb-51-posec3d",
                    repo_type="dataset",
                    resume_download=True,
                    local_dir=data_dir / "posec3d",
                    cache_dir=cache_dir,
                )
            elif dataset_name == "ucf-101":
                snapshot_download(
                    repo_id="kiyoonkim/ucf-101-gulprgb",
                    repo_type="dataset",
                    resume_download=True,
                    local_dir=data_dir,
                    cache_dir=cache_dir,
                )
                snapshot_download(
                    repo_id="kiyoonkim/ucf-101-posec3d",
                    repo_type="dataset",
                    resume_download=True,
                    local_dir=data_dir / "posec3d",
                    cache_dir=cache_dir,
                )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        gulp_dir_path = data_dir / "gulp_rgb"

        if dataset_name == "hmdb-51":
            skeleton_pkl_path = data_dir / "posec3d" / "hmdb51_2d.pkl"
            if set_name == "train":
                mode = "train"
                csv_path = data_dir / "splits_gulp_rgb" / f"train{split_num}.csv"
            elif set_name == "test":
                mode = "test"
                csv_path = data_dir / "splits_gulp_rgb" / f"test{split_num}.csv"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        elif dataset_name == "ucf-101":
            skeleton_pkl_path = data_dir / "posec3d" / "ucf101_2d.pkl"
            if set_name == "train":
                mode = "train"
                csv_path = (
                    data_dir / "splits_gulp_rgb" / f"trainlist{split_num:02d}.txt"
                )
            elif set_name == "test":
                mode = "test"
                csv_path = data_dir / "splits_gulp_rgb" / f"testlist{split_num:02d}.txt"
            else:
                raise ValueError(f"Unknown set_name: {set_name}")
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        data = GulpSparsesampleSkeletonDataset(
            csv_path,
            mode,
            input_frame_length,
            gulp_dir_path,
            skeleton_pkl_path,
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
            frame_neighbours=1,
        )
        dataset[set_name] = data

    return dataset
