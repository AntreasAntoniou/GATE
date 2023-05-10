import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import Resize

from ..iwildcam_2022 import filter_metadata_with_counts, read_all_train_metadata
from . import utils as utils

logger = logging.getLogger(__name__)


PAD_VALUE = -999999


def squeeze_transform_224(video):
    """
    video: (T, C, H, W)
    """

    return Resize((224, 224), antialias=True)(video)


class IWildCam2022Dataset(torch.utils.data.Dataset):
    """
    Note that it will only return sequences with count annotations. (1780 samples)
    """

    def __init__(
        self,
        dataset_rootdir: str | Path,
        transform: Any | None = squeeze_transform_224,
        max_num_frames: int = 10,
        max_num_detections: int = 23,
    ):
        self.dataset_rootdir = Path(dataset_rootdir)
        self.transform = transform
        self.max_num_frames = max_num_frames
        self.max_num_detections = max_num_detections

        (
            seq_id_to_per_image_annotations,
            self.seq_id_to_counts,
        ) = read_all_train_metadata(self.dataset_rootdir)
        self.seq_id_to_per_image_annotations = filter_metadata_with_counts(
            seq_id_to_per_image_annotations, self.seq_id_to_counts
        )
        self.index_to_seq_id = list(sorted(self.seq_id_to_per_image_annotations.keys()))

    def __getitem__(self, index):
        seq_id = self.index_to_seq_id[index]
        per_image_annotations = self.seq_id_to_per_image_annotations[seq_id]
        counts = self.seq_id_to_counts[seq_id]

        num_frames = len(per_image_annotations)
        width = per_image_annotations[0]["width"]
        height = per_image_annotations[0]["height"]

        locations = torch.zeros(self.max_num_frames, dtype=torch.int64)
        sub_locations = torch.zeros(self.max_num_frames, dtype=torch.int64)
        utc_timestamps = torch.zeros(self.max_num_frames, dtype=torch.float32)
        category_ids = torch.zeros(self.max_num_frames, dtype=torch.int64)
        gps_locations = torch.zeros(
            self.max_num_frames, 2, dtype=torch.float32
        )  # (latitude, longitude)
        gps_sub_locations = torch.zeros(
            self.max_num_frames, 2, dtype=torch.float32
        )  # (latitude, longitude)

        instance_masks = torch.zeros(
            self.max_num_frames, height, width, dtype=torch.uint8
        )
        max_detection_confs = torch.zeros(self.max_num_frames, dtype=torch.float32)

        num_detections = torch.zeros(self.max_num_frames, dtype=torch.int64)
        detection_categories = torch.zeros(
            self.max_num_frames, self.max_num_detections, dtype=torch.int64
        )
        detection_confs = torch.zeros(
            self.max_num_frames, self.max_num_detections, dtype=torch.float32
        )
        detection_bboxes = torch.zeros(
            self.max_num_frames, self.max_num_detections, 4, dtype=torch.float32
        )

        video = torch.zeros(self.max_num_frames, 3, height, width, dtype=torch.uint8)

        for frame_idx, image_annotaion in enumerate(per_image_annotations):
            """
            Example of image_annotaion:
                {'seq_num_frames': 9, 'location': 218, 'datetime': '2013-05-25 17:05:41.000', 'id': '97b37728-21bc-11ea-a13a-137349068a90', 'seq_id': '3019fa50-7d42-11eb-8fb5-0242ac1c0002', 'width': 1920, 'height': 1080, 'file_name': '97b37728-21bc-11ea-a13a-137349068a90.jpg', 'sub_location': 0, 'seq_frame_num': 8, 'category_id': 372, 'gps_location': {'latitude': 17.492381819738956, 'longitude': -89.21560449646441}, 'gps_sub_location': {'latitude': -2.6262412503467187, 'longitude': 29.35055652840072}, 'detection': {'file': 'train/97b37728-21bc-11ea-a13a-137349068a90.jpg', 'max_detection_conf': 0.999, 'detections': [{'category': '1', 'conf': 0.999, 'bbox': [0.153, 0.366, 0.186, 0.535]}, {'category': '1', 'conf': 0.999, 'bbox': [0.355, 0.451, 0.288, 0.518]}, {'category': '1', 'conf': 0.971, 'bbox': [0, 0.338, 0.056, 0.343]}, {'category': '1', 'conf': 0.715, 'bbox': [0.978, 0.649, 0.021, 0.078]}, {'category': '1', 'conf': 0.353, 'bbox': [0.002, 0.272, 0.055, 0.177]}, {'category': '2', 'conf': 0.251, 'bbox': [0.003, 0.27, 0.051, 0.172]}, {'category': '1', 'conf': 0.169, 'bbox': [0.002, 0.281, 0.057, 0.282]}]}})]}
            """

            assert image_annotaion["seq_id"] == seq_id
            assert image_annotaion["seq_frame_num"] == frame_idx
            assert image_annotaion["seq_num_frames"] == num_frames
            assert image_annotaion["width"] == width
            assert image_annotaion["height"] == height

            location = image_annotaion["location"]
            utc_timestamp = datetime.strptime(
                image_annotaion["datetime"] + "+0000", "%Y-%m-%d %H:%M:%S.%f%z"
            ).timestamp()
            file_name = image_annotaion["file_name"]
            if "sub_location" in image_annotaion:
                sub_location = image_annotaion["sub_location"]
            else:
                sub_location = PAD_VALUE
            category_id = image_annotaion["category_id"]

            if "gps_location" in image_annotaion:
                gps_location_latitude = image_annotaion["gps_location"]["latitude"]
                gps_location_longitude = image_annotaion["gps_location"]["longitude"]
            else:
                gps_location_latitude = PAD_VALUE
                gps_location_longitude = PAD_VALUE

            if "gps_sub_location" in image_annotaion:
                gps_sub_location_latitude = image_annotaion["gps_sub_location"][
                    "latitude"
                ]
                gps_sub_location_longitude = image_annotaion["gps_sub_location"][
                    "longitude"
                ]
            else:
                gps_sub_location_latitude = PAD_VALUE
                gps_sub_location_longitude = PAD_VALUE

            max_detection_conf = image_annotaion["detection"]["max_detection_conf"]
            detections = image_annotaion["detection"]["detections"]

            # Annotations
            locations[frame_idx] = location
            sub_locations[frame_idx] = sub_location
            utc_timestamps[frame_idx] = utc_timestamp
            category_ids[frame_idx] = category_id
            gps_locations[frame_idx, 0] = gps_location_latitude
            gps_locations[frame_idx, 1] = gps_location_longitude
            gps_sub_locations[frame_idx, 0] = gps_sub_location_latitude
            gps_sub_locations[frame_idx, 1] = gps_sub_location_longitude
            max_detection_confs[frame_idx] = max_detection_conf

            num_detections[frame_idx] = len(detections)
            detection_confs[frame_idx, : len(detections)] = torch.tensor(
                [detection["conf"] for detection in detections]
            )
            detection_categories[frame_idx, : len(detections)] = torch.tensor(
                [int(detection["category"]) for detection in detections]
            )
            detection_bboxes[frame_idx, : len(detections), :] = torch.tensor(
                [detection["bbox"] for detection in detections]
            )

            # Load image (BGR -> RGB)
            image = cv2.imread(
                str(self.dataset_rootdir / "train" / "train" / file_name)
            )[..., ::-1]
            video[frame_idx] = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .contiguous()
            )

            # Load detection masks
            instance_mask = cv2.imread(
                str(
                    (
                        self.dataset_rootdir
                        / "instance_masks"
                        / "instance_masks"
                        / file_name
                    ).with_suffix(".png")
                ),
                cv2.IMREAD_UNCHANGED,
            )
            instance_masks[frame_idx] = torch.from_numpy(instance_mask)

        if self.transform is not None:
            video = self.transform(video)
            instance_masks = self.transform(instance_masks)

        return {
            "index": index,
            "counts": counts,
            "num_frames": num_frames,
            "locations": locations,
            "sub_locations": sub_locations,
            "utc_timestamps": utc_timestamps,
            "category_ids": category_ids,
            "gps_locations": gps_locations,
            "gps_sub_locations": gps_sub_locations,
            "instance_masks": instance_masks,
            "max_detection_confs": max_detection_confs,
            "num_detections": num_detections,
            "detection_categories": detection_categories,
            "detection_confs": detection_confs,
            "detection_bboxes": detection_bboxes,
            "video": video,
        }

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.index_to_seq_id)
