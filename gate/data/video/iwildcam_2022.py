import json
import os
from pathlib import Path


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

    sequence_counts = _read_csv_to_listdict(metadata_dir / "train_sequence_counts.csv")

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
            seq_id_to_annotations[seq_id][id]["gps_sub_location"] = gps_sub_location

        # Detection
        seq_id_to_annotations[seq_id][id]["detection"] = image_id_to_detection[id]

        # Count
        count = None
        if seq_id in seq_id_to_counts:
            count = seq_id_to_counts[seq_id]
        seq_id_to_annotations[seq_id][id]["count"] = count

    return seq_id_to_annotations


def count_num_files(dataset_rootdir: str | Path):
    dataset_rootdir = Path(dataset_rootdir)
    if dataset_rootdir.name != "iwildcam2022":
        dataset_rootdir = dataset_rootdir / "iwildcam2022"

    metadata_dir = dataset_rootdir / "metadata" / "metadata"
    num_train_files = len(list((dataset_rootdir / "train" / "train").glob("*.jpg")))
    # num_test_files = len(list((dataset_rootdir / "test" / "test").glob("*.jpg")))
    num_metadata_files = len(list(metadata_dir.glob("*")))
    num_mask_files = len(
        list((dataset_rootdir / "instance_masks" / "instance_masks").glob("*.png"))
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


if __name__ == "__main__":
    count_num_files("/disk/scratch_fast1/datasets/iwildcam2022")
    seq_id_to_annotations = read_all_train_metadata(
        "/disk/scratch_fast1/datasets/iwildcam2022"
    )
    print(seq_id_to_annotations)
    print(len(seq_id_to_annotations))
