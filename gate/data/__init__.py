import logging
import pathlib
from typing import Optional

from gate.boilerplate.utils import count_files_recursive

logger = logging.getLogger(__name__)


def download_kaggle_dataset(
    dataset_name: str,
    dataset_path: str,
    target_dir_path: pathlib.Path,
    file_count_after_download_and_extract: Optional[int],
):
    # Initialize the Kaggle API client
    from kaggle import KaggleApi

    dataset_download_path = pathlib.Path(target_dir_path) / dataset_name
    if (
        pathlib.Path(dataset_download_path).exists()
        and count_files_recursive(dataset_download_path)
        >= file_count_after_download_and_extract
    ):
        logger.info(f"Dataset directory {target_dir_path} already exists.")
        return {"dataset_download_path": dataset_download_path}

    api = KaggleApi()
    # Ensure the directory for the dataset exists

    api.authenticate()

    # Download the dataset
    api.dataset_download_files(
        dataset_path,
        path=dataset_download_path,
        unzip=True,
        quiet=False,
        force=True,
    )
    return {"dataset_download_path": dataset_download_path}
