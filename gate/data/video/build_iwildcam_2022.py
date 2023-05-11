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
        split_path = data_dir / "splits" / f"{set_name}.txt"

        if set_name not in ["train", "val", "test"]:
            raise ValueError(f"Unknown set_name: {set_name}")

        data = IWildCam2022Dataset(data_dir, split_path=split_path)
        dataset[set_name] = data

    return dataset
