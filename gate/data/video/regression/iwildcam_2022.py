import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


DOWNLOAD_GB = 105  # more precisely, 101GiB
EXTRACT_GB = 108  # more precisely, 104GiB


def _read_csv_to_listdict(csv_file: str | Path):
    csv_file = Path(csv_file)
    with open(csv_file) as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")
    data = []
    for line in lines[1:]:
        line = line.strip().split(",")
        data.append(dict(zip(header, line)))
    return data


def read_all_train_metadata(dataset_rootdir: str | Path):
    dataset_rootdir = Path(dataset_rootdir)
    if dataset_rootdir.name != "iwildcam2022":
        dataset_rootdir = dataset_rootdir / "iwildcam2022"

    metadata_dir = dataset_rootdir / "metadata" / "metadata"

    sequence_counts = _read_csv_to_listdict(
        metadata_dir / "train_sequence_counts.csv"
    )

    with open(metadata_dir / "iwildcam2022_mdv4_detections.json") as f:
        detection_data = json.load(f)

    with open(metadata_dir / "gps_locations.json") as f:
        gps_data = json.load(f)

    with open(metadata_dir / "iwildcam2022_train_annotations.json") as f:
        data = json.load(f)

    seq_id_to_counts = {}
    for sequence_count in sequence_counts:
        seq_id_to_counts[sequence_count["seq_id"]] = sequence_count["count"]

    # data.keys(): images, categories, annotations
    image_id_to_detection = {}
    for detection in detection_data["images"]:
        image_id = os.path.splitext(os.path.basename(detection["file"]))[0]
        image_id_to_detection[image_id] = detection

    image_id_to_category_id = {}
    for category in data["annotations"]:
        image_id_to_category_id[category["image_id"]] = category["category_id"]

    seq_id_to_annotations = {}
    for images_data in data["images"]:
        seq_id = images_data["seq_id"]
        id = images_data["id"]
        if seq_id not in seq_id_to_annotations:
            seq_id_to_annotations[seq_id] = {}
        seq_id_to_annotations[seq_id][id] = images_data

        # Category
        category_id = image_id_to_category_id[id]
        seq_id_to_annotations[seq_id][id]["category_id"] = category_id

        # GPS location
        if str(images_data["location"]) not in gps_data:
            gps_location = None
        else:
            gps_location = gps_data[str(images_data["location"])]

        if "sub_location" not in images_data:
            gps_sub_location = None
        else:
            if str(images_data["sub_location"]) not in gps_data:
                gps_sub_location = None
            else:
                gps_sub_location = gps_data[str(images_data["sub_location"])]
            seq_id_to_annotations[seq_id][id]["gps_location"] = gps_location
            seq_id_to_annotations[seq_id][id][
                "gps_sub_location"
            ] = gps_sub_location

        # Detection
        seq_id_to_annotations[seq_id][id]["detection"] = image_id_to_detection[
            id
        ]

    # Rearrange annotations.
    # seq_id_to_annotations[seq_id][id] -> seq_id_to_annotations[seq_id][frame_index]

    seq_id_to_per_image_annotations = {}
    for seq_id, image_id_to_annotations in seq_id_to_annotations.items():
        seq_id_to_per_image_annotations[seq_id] = list(
            sorted(
                image_id_to_annotations.values(),
                key=lambda x: x["seq_frame_num"],
            )
        )

    return seq_id_to_per_image_annotations, seq_id_to_counts


def filter_metadata_with_counts(
    seq_id_to_per_image_annotations: dict, seq_id_to_counts: dict[str, int]
):
    seq_id_to_per_image_annotations_filtered = {}
    for (
        seq_id,
        per_image_annotations,
    ) in seq_id_to_per_image_annotations.items():
        if seq_id in seq_id_to_counts:
            seq_id_to_per_image_annotations_filtered[seq_id] = (
                per_image_annotations
            )
    return seq_id_to_per_image_annotations_filtered


def count_num_files(dataset_rootdir: str | Path):
    dataset_rootdir = Path(dataset_rootdir)
    if dataset_rootdir.name != "iwildcam2022":
        dataset_rootdir = dataset_rootdir / "iwildcam2022"

    metadata_dir = dataset_rootdir / "metadata" / "metadata"
    num_train_files = len(
        list((dataset_rootdir / "train" / "train").glob("*.jpg"))
    )
    # num_test_files = len(list((dataset_rootdir / "test" / "test").glob("*.jpg")))
    num_metadata_files = len(list(metadata_dir.glob("*")))
    num_mask_files = len(
        list(
            (dataset_rootdir / "instance_masks" / "instance_masks").glob(
                "*.png"
            )
        )
    )

    if num_train_files != 201399:
        raise FileNotFoundError(
            f"Expected 201399 train files, but found {num_train_files}"
        )
    if num_metadata_files != 5:
        raise FileNotFoundError(
            f"Expected 5 metadata files, but found {num_metadata_files}"
        )
    if num_mask_files != 150221:
        raise FileNotFoundError(
            f"Expected 150221 mask files, but found {num_mask_files}"
        )
    # assert num_test_files == 60029


def prepare_iwildcam_2022(dataset_rootdir: str | Path):
    logger.info(
        "Preparing iWildCam 2022 dataset. NOTE: you need to have kaggle API"
        " key configured and join the competition."
    )
    dataset_rootdir = Path(dataset_rootdir)
    if dataset_rootdir.name != "iwildcam2022":
        dataset_rootdir = dataset_rootdir / "iwildcam2022"

    dataset_rootdir.mkdir(parents=True, exist_ok=True)
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    if cache_dir is None:
        # use tmp dir
        cache_dir = Path(tempfile.gettempdir()) / "hf"

    snapshot_download(
        "kiyoonkim/iwildcam-2022-splits",
        repo_type="dataset",
        cache_dir=cache_dir,
        local_dir=dataset_rootdir / "splits",
        resume_download=True,
        max_workers=mp.cpu_count(),
    )

    try:
        count_num_files(dataset_rootdir)
    except FileNotFoundError:
        disk_space = shutil.disk_usage(dataset_rootdir).free
        if disk_space < (DOWNLOAD_GB + EXTRACT_GB) * 1024 * 1024 * 1024:
            raise RuntimeError(
                "Insufficient disk space. At least"
                f" {DOWNLOAD_GB + EXTRACT_GB} GB is required, but only"
                f" {disk_space / 1024 / 1024 / 1024:.1f} GB available"
            )

        subprocess.call(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "iwildcam2022-fgvc9",
                "-p",
                str(dataset_rootdir),
            ]
        )
        with zipfile.ZipFile(
            dataset_rootdir / "iwildcam2022-fgvc9.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(dataset_rootdir)


def train_val_test_splits(seq_ids: list[str]):
    """
    Split seq_ids into train (80%), val (10%), test (10%).
    """
    random.seed(0)
    random.shuffle(seq_ids)
    num_train = int(len(seq_ids) * 0.8)
    num_val = int(len(seq_ids) * 0.1)
    train_seq_ids = seq_ids[:num_train]
    val_seq_ids = seq_ids[num_train : num_train + num_val]
    test_seq_ids = seq_ids[num_train + num_val :]
    return train_seq_ids, val_seq_ids, test_seq_ids


def _save_list_to_file(file_path: str | Path, seq_ids: list[str]):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for seq_id in seq_ids:
            f.write(seq_id + "\n")
