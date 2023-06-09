import os
from pathlib import Path

from accelerate import Accelerator
from huggingface_hub import snapshot_download

from ..loader.gulp_sparsesample_tube_dataset import GulpSparsesampleTubeDataset


def build_gulp_tube_dataset(
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
        "jhmdb",
        "ucf-101-24",
    ]
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    assert cache_dir is not None

    data_dir = Path(data_dir)

    if dataset_name == "jhmdb":
        assert split_num in [1, 2, 3]
        if data_dir.name != "jhmdb":
            data_dir = data_dir / "jhmdb"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    elif dataset_name == "ucf-101-24":
        assert split_num == 1
        if data_dir.name != "ucf101_24":
            data_dir = data_dir / "ucf101_24"
        if sets_to_include is None:
            sets_to_include = ["train", "test"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    assert sets_to_include is not None

    if ensure_installed:
        if accelerator is None or accelerator.is_local_main_process:
            if dataset_name == "jhmdb":
                snapshot_download(
                    repo_id="kiyoonkim/jhmdb-gulprgb",
                    repo_type="dataset",
                    resume_download=True,
                    local_dir=data_dir,
                    cache_dir=cache_dir,
                )
            elif dataset_name == "ucf-101-24":
                snapshot_download(
                    repo_id="kiyoonkim/ucf-101-24-gulprgb",
                    repo_type="dataset",
                    resume_download=True,
                    local_dir=data_dir,
                    cache_dir=cache_dir,
                )

        if accelerator is not None:
            accelerator.wait_for_everyone()

    dataset = {}
    for set_name in sets_to_include:
        input_frame_length = 8
        gulp_dir_path = data_dir / "gulp_rgb"

        if dataset_name == "jhmdb":
            tube_pkl_path = data_dir / "JHMDB-GT.pkl"
        elif dataset_name == "ucf-101-24":
            tube_pkl_path = data_dir / "UCF101v2-GT.pkl"
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        if set_name == "train":
            mode = "train"
            csv_path = data_dir / "splits_gulp_rgb" / "train.csv"
        elif set_name == "test":
            mode = "test"
            csv_path = data_dir / "splits_gulp_rgb" / "test.csv"
        else:
            raise ValueError(f"Unknown set_name: {set_name}")

        data = GulpSparsesampleTubeDataset(
            csv_path,
            mode,
            input_frame_length,
            gulp_dir_path,
            tube_pkl_path,
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
