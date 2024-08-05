import logging
import pathlib
import zipfile
from typing import Optional

from tqdm import tqdm  # for the progress bar

logger = logging.getLogger(__name__)


def unzip_file(zip_file_path: pathlib.Path, target_dir_path: pathlib.Path):
    try:
        with zipfile.ZipFile(zip_file_path, "r") as z:
            # Fetch the total number of entries (files + directories) in the zip file to initialize the progress bar
            total_files = len(z.infolist())

            with tqdm(total=total_files, desc="Extracting", ncols=80) as pbar:
                for member in z.infolist():
                    # Extract the file/directory with the Path's write_bytes method
                    target_dir_path.joinpath(member.filename).write_bytes(
                        z.read(member)
                    )
                    pbar.update(1)  # update the progress bar
    except Exception as e:
        logger.error(f"Could not extract zip file, got {e}")

    # Delete the zip file after extraction
    try:
        zip_file_path.unlink()  # Using pathlib's unlink method to delete the file
    except OSError as e:
        logger.error(f"Could not delete zip file, got {e}")


def download_kaggle_dataset(
    dataset_name: str,
    dataset_path: str,
    target_dir_path: pathlib.Path,
    file_count_after_download_and_extract: Optional[int],
    is_competition: bool = False,
    unzip: bool = True,
):
    # Initialize the Kaggle API client
    from kaggle import KaggleApi

    from gate.boilerplate.utils import count_files_recursive

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

    if is_competition:
        logger.info(
            "Check if path"
            f" {dataset_download_path / f'{dataset_path}.zip'} exists"
        )
        if not pathlib.Path(
            dataset_download_path / f"{dataset_path}.zip"
        ).exists():
            logger.info(f"Downloading competition {dataset_path}")
            api.competition_download_files(
                competition=dataset_path,
                path=dataset_download_path,
                quiet=False,
                force=True,
            )
        if unzip:
            unzip_file(
                zip_file_path=dataset_download_path / f"{dataset_path}.zip",
                target_dir_path=dataset_download_path,
            )
    else:
        # Download the dataset
        api.dataset_download_files(
            dataset_path,
            path=dataset_download_path,
            unzip=unzip,
            quiet=False,
            force=True,
        )

    return {"dataset_download_path": dataset_download_path}
