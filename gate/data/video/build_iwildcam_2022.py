import os
from pathlib import Path

from accelerate import Accelerator
from huggingface_hub import snapshot_download

from .iwildcam_2022 import prepare_iwildcam_2022
from .loader.iwildcam2022_dataset import IWildCam2022Dataset


def build_iwildcam_2022_dataset(
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
        # input_frame_length = 8
        videos_dir = data_dir / "k400" / "videos"
        if set_name == "train":
            mode = "train"
            csv_path = data_dir / "k400" / "splits_decord_videos" / "train.csv"
            videos_dir = videos_dir / "train"
        elif set_name == "val":
            mode = "test"
            csv_path = data_dir / "k400" / "splits_decord_videos" / "val.csv"
            videos_dir = videos_dir / "val"
        elif set_name == "test":
            mode = "test"
            csv_path = data_dir / "k400" / "splits_decord_videos" / "test.csv"
            videos_dir = videos_dir / "test"
        else:
            raise ValueError(f"Unknown set_name: {set_name}")

        data = IWildCam2022Dataset(data_dir)
        dataset[set_name] = data

    return dataset
