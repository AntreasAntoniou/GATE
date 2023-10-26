"""
Download and extract kinetics-400 dataset.
Original code and hosting from https://github.com/cvdfoundation/kinetics-dataset, but re-written in Python.
"""
import logging
import multiprocessing as mp
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from gate.data.video.classes import kinetics_400_classes as CLASSES

logger = logging.getLogger(__name__)
# KINETICS-400
"""     videos      csv         replace
test:   38685       39805       0
train:  240258      246534      1392
val:    19881       19906       0
"""


SPLITS = ["test", "train", "val"]
NUM_TRAIN_PARTS = 242
NUM_VAL_PARTS = 20
NUM_TEST_PARTS = 39

DOWNLOAD_GB = 450  # more precisely, 435 GiB
EXTRACT_GB = 450  # more precisely, 438 GiB


def _fetch_or_resume(url: str, filepath: Path) -> None:
    """Fetches or resumes file download from a given URL.

    Args:
        url (str): URL to fetch the file from.
        filepath (Path): Local path to save the file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "ab") as f:
        headers = {}
        pos = f.tell()
        if pos:
            headers["Range"] = f"bytes={pos}-"
        response = requests.get(url, headers=headers, stream=True)

        total_size = response.headers.get("content-length")
        if total_size is not None:
            total_size = int(total_size)
            for data in tqdm(
                iterable=response.iter_content(chunk_size=1024 * 1024),
                total=total_size // 1024 // 1024,
                unit="MiB",
            ):
                f.write(data)

    logger.info(f"Successfully downloaded {filepath.name}")


def _extract_tar_gz(tar_gz_file: Path, extract_dir: Path) -> None:
    """Extracts a tar.gz file to a given directory.

    Args:
        tar_gz_file (Path): Path to the tar.gz file.
        extract_dir (Path): Directory to extract files into.
    """
    logger.info(f"Extracting {tar_gz_file.name} to {extract_dir}")
    with tarfile.open(tar_gz_file) as tar:
        tar.extractall(extract_dir)
    logger.info(f"Successfully extracted {tar_gz_file.name}")


def _check_disk_space(required_gb: int, path: Path) -> None:
    """Checks if sufficient disk space is available.

    Args:
        required_gb (int): Required disk space in GB.
        path (Path): Directory to check disk space for.
    """
    disk_space = shutil.disk_usage(path).free
    if disk_space < required_gb * 1024**3:
        raise RuntimeError(
            f"Insufficient disk space. At least {required_gb}GB is required, but only {disk_space / 1024 ** 3:.1f} GB available."
        )


def download_kinetics(dataset_rootdir: Path) -> None:
    """Download the Kinetics-400 dataset to the specified root directory.

    Args:
        dataset_rootdir (Path): Root directory to download the dataset.
    """
    dataset_dir = dataset_rootdir / "kinetics-dataset"
    download_dir = dataset_dir / "k400_targz"
    download_dir.mkdir(parents=True, exist_ok=True)

    _check_disk_space(DOWNLOAD_GB, dataset_dir)

    # Define download functions for each part of the dataset
    def download_part(url_base: str, part_dir: Path, part_num: int) -> None:
        _fetch_or_resume(
            f"{url_base}/part_{part_num}.tar.gz",
            part_dir / f"part_{part_num}.tar.gz",
        )

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Download training parts
        executor.map(
            lambda i: download_part(
                "https://s3.amazonaws.com/kinetics/400/train",
                download_dir / "train",
                i,
            ),
            range(NUM_TRAIN_PARTS),
        )
        # Download validation parts
        executor.map(
            lambda i: download_part(
                "https://s3.amazonaws.com/kinetics/400/val",
                download_dir / "val",
                i,
            ),
            range(NUM_VAL_PARTS),
        )
        # Download test parts
        executor.map(
            lambda i: download_part(
                "https://s3.amazonaws.com/kinetics/400/test",
                download_dir / "test",
                i,
            ),
            range(NUM_TEST_PARTS),
        )

    # Download replacement for corrupted files
    _fetch_or_resume(
        "https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz",
        download_dir / "replacement" / "replacement_for_corrupted_k400.tgz",
    )

    logger.info("Successfully downloaded Kinetics-400 dataset.")


# Continuing with the refactoring of remaining functions


def download_kinetics_annotations(dataset_rootdir: Path) -> None:
    """Download the Kinetics-400 annotations to the specified root directory.

    Args:
        dataset_rootdir (Path): Root directory to download the annotations.
    """
    dataset_dir = dataset_rootdir / "kinetics-dataset"
    annotations_dir = dataset_dir / "k400" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    def download_annotation_file(file_name: str) -> None:
        _fetch_or_resume(
            f"https://s3.amazonaws.com/kinetics/400/annotations/{file_name}",
            annotations_dir / file_name,
        )

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        executor.map(
            download_annotation_file, ["train.csv", "val.csv", "test.csv"]
        )

    logger.info("Successfully downloaded Kinetics-400 annotations.")


def extract_kinetics(
    downloaded_dataset_rootdir: Path, extract_dataset_rootdir: Path = None
) -> None:
    """Extract the Kinetics-400 dataset tar files.

    Args:
        downloaded_dataset_rootdir (Path): Directory where the dataset tar files are downloaded.
        extract_dataset_rootdir (Path, optional): Directory where to extract the files. Defaults to None.
    """
    dataset_dir = downloaded_dataset_rootdir / "kinetics-dataset"
    download_dir = dataset_dir / "k400_targz"

    if extract_dataset_rootdir is None:
        extract_dir = dataset_dir / "k400"
    else:
        extract_dir = extract_dataset_rootdir / "kinetics-dataset" / "k400"

    extract_dir.mkdir(parents=True, exist_ok=True)

    # Assuming EXTRACT_GB is a constant that needs to be defined
    _check_disk_space(EXTRACT_GB, extract_dir)

    def extract_part(part_dir: Path, part_num: int, split: str) -> None:
        logger.info(f"Extracting part {part_num}/{split}")
        _extract_tar_gz(
            part_dir / f"part_{part_num}.tar.gz",
            extract_dir / split,
        )

    with ThreadPoolExecutor(max_workers=512) as executor:
        # Extract training parts
        executor.map(
            lambda i: extract_part(
                download_dir / "train",
                i,
                "train",
            ),
            range(NUM_TRAIN_PARTS),
        )
        # Extract validation parts
        executor.map(
            lambda i: extract_part(
                download_dir / "val",
                i,
                "val",
            ),
            range(NUM_VAL_PARTS),
        )
        # Extract test parts
        executor.map(
            lambda i: extract_part(
                download_dir / "test",
                i,
                "test",
            ),
            range(NUM_TEST_PARTS),
        )

    # Extract replacement for corrupted files
    _extract_tar_gz(
        download_dir / "replacement" / "replacement_for_corrupted_k400.tgz",
        extract_dir / "replacement",
    )

    logger.info("Successfully extracted Kinetics-400 dataset.")


# Continuing with the refactoring of the remaining functions


def _load_label(csv: Path) -> dict:
    """Loads labels from a given CSV file.

    Args:
        csv (Path): Path to the CSV file.

    Returns:
        dict: A dictionary mapping video IDs to their labels.
    """
    table = np.loadtxt(csv, skiprows=1, dtype=str, delimiter=",")
    return {k: v.replace('"', "") for k, v in zip(table[:, 1], table[:, 0])}


def _collect_dict(path: Path, split: str, replace_videos: dict) -> dict:
    """Collects video paths and corresponding labels for a given split.

    Args:
        path (Path): Root directory of the dataset.
        split (str): Dataset split ('train', 'val', 'test').
        replace_videos (dict): Dictionary of videos to be replaced.

    Returns:
        dict: Dictionary mapping video paths to their labels.
    """
    split_video_path = path / split
    split_csv = _load_label(path / f"annotations/{split}.csv")
    split_videos = list(split_video_path.glob("*.mp4"))
    split_videos = {str(p.stem)[:11]: p for p in split_videos}

    # Replace paths for corrupted videos
    match_dict = {
        k: replace_videos[k]
        for k in split_videos.keys() & replace_videos.keys()
    }
    split_videos.update(match_dict)

    # Collect videos with labels from csv: dict with {video_path: class}
    return {
        split_videos[k]: split_csv[k]
        for k in split_csv.keys() & split_videos.keys()
    }


def arrange_by_classes(dataset_rootdir: Path) -> None:
    """Arranges videos by classes.

    Args:
        dataset_rootdir (Path): Root directory of the dataset.
    """
    dataset_dir = dataset_rootdir / "kinetics-dataset"
    path = dataset_dir / "k400"

    if not path.exists():
        raise FileNotFoundError(f"Provided path: {path} does not exist")

    # Collect videos in replacement
    replace_videos = {
        str(p.stem)[:11]: p
        for p in (path / "replacement/replacement_for_corrupted_k400").glob(
            "*.mp4"
        )
    }
    video_parent = path / "videos"

    for split in ["train", "val", "test"]:
        split_video_path = video_parent / split
        split_video_path.mkdir(exist_ok=True, parents=True)
        split_final = _collect_dict(path, split, replace_videos)

        logger.info(f"Found {len(split_final)} videos in split: {split}")
        labels = set(split_final.values())

        # Create label directories
        for label in labels:
            (split_video_path / label.replace(" ", "_")).mkdir(
                exist_ok=True, parents=True
            )

        # Symlink videos to respective labels
        for vid_pth, label in tqdm(
            split_final.items(), desc=f"Progress {split}"
        ):
            dst_vid = split_video_path / label.replace(" ", "_") / vid_pth.name
            if dst_vid.is_symlink():
                dst_vid.unlink()
            dst_vid.symlink_to(vid_pth.resolve(), target_is_directory=False)

    logger.info("Successfully arranged videos by classes.")


def check_num_files_each_class(
    dataset_rootdir: Path, expected_file_counts: dict
) -> bool:
    """Checks the number of files in each class for each dataset split.

    Args:
        dataset_rootdir (Path): Root directory of the dataset.
        expected_file_counts (dict): Expected number of files for each class in each split.

    Returns:
        bool: True if the number of files matches the expected counts, False otherwise.
    """
    dataset_dir = dataset_rootdir / "kinetics-dataset"
    videos_dir = dataset_dir / "k400" / "videos"

    for split in ["train", "val", "test"]:
        split_dir = videos_dir / split
        if not split_dir.exists():
            logger.warning(f"{split_dir} does not exist.")
            return False

        num_files_each_class = {}
        for class_dir in split_dir.iterdir():
            num_files_each_class[class_dir.name] = len(
                list(class_dir.iterdir())
            )

        for label, count in num_files_each_class.items():
            if count != expected_file_counts[split].get(label, 0):
                logger.warning(
                    f"Mismatch in file count for label {label} in {split}. Expected {expected_file_counts[split].get(label, 0)}, got {count}."
                )
                return False

    return True


def prepare_kinetics_400(
    download_dataset_rootdir: Path,
    extract_dataset_rootdir: Path,
) -> None:
    """Prepares the Kinetics-400 dataset by downloading, extracting, and arranging the videos.

    Args:
        download_dataset_rootdir (Path): Directory to download the dataset.
        extract_dataset_rootdir (Path, optional): Directory to extract the dataset. Defaults to None.
        expected_file_counts (dict, optional): Expected number of files for each class in each split. Defaults to None.
    """

    # Check if everything is already prepared
    is_prepared = check_num_files_each_class(
        extract_dataset_rootdir
        if extract_dataset_rootdir
        else download_dataset_rootdir,
        CLASSES,
    )

    if is_prepared:
        logger.info("Kinetics-400 dataset is already prepared.")
        return

    # Download and extract dataset
    download_kinetics(download_dataset_rootdir)
    extract_kinetics(download_dataset_rootdir, extract_dataset_rootdir)

    # Download annotations and arrange by classes
    download_kinetics_annotations(
        extract_dataset_rootdir
        if extract_dataset_rootdir
        else download_dataset_rootdir
    )
    arrange_by_classes(
        extract_dataset_rootdir
        if extract_dataset_rootdir
        else download_dataset_rootdir
    )

    # Final verification
    is_prepared = check_num_files_each_class(
        extract_dataset_rootdir
        if extract_dataset_rootdir
        else download_dataset_rootdir,
        CLASSES,
    )
    if not is_prepared:
        raise RuntimeError(
            "Something went wrong while preparing the Kinetics-400 dataset."
        )
    else:
        logger.info("Successfully prepared the Kinetics-400 dataset.")
