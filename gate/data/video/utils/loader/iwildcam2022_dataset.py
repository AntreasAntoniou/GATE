import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import Resize

from gate.data.video.regression.iwildcam_2022 import (
    filter_metadata_with_counts, read_all_train_metadata)

logger = logging.getLogger(__name__)


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
        split_path: str | Path | None = None,
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

        if split_path is not None:
            split_path = Path(split_path)
            with open(split_path, "r") as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            self.index_to_seq_id = lines
        else:
            self.index_to_seq_id = list(
                sorted(self.seq_id_to_per_image_annotations.keys())
            )

        # Remove unnecessary metadata not included in the split
        self.seq_id_to_per_image_annotations = {
            seq_id: self.seq_id_to_per_image_annotations[seq_id]
            for seq_id in self.index_to_seq_id
        }
        self.seq_id_to_counts = {
            seq_id: self.seq_id_to_counts[seq_id]
            for seq_id in self.index_to_seq_id
        }

    def __getitem__(self, index):
        seq_id = self.index_to_seq_id[index]
        per_image_annotations = self.seq_id_to_per_image_annotations[seq_id]
        counts = self.seq_id_to_counts[seq_id]

        num_frames = len(per_image_annotations)
        width = per_image_annotations[0]["width"]
        height = per_image_annotations[0]["height"]

        video = torch.zeros(
            self.max_num_frames, 3, height, width, dtype=torch.uint8
        )

        for frame_idx, image_annotaion in enumerate(per_image_annotations):
            """
            Example of image_annotaion:
                {'seq_num_frames': 9, 'location': 218, 'datetime': '2013-05-25 17:05:41.000',
                'id': '97b37728-21bc-11ea-a13a-137349068a90', 'seq_id': '3019fa50-7d42-11eb-8fb5-0242ac1c0002',
                'width': 1920, 'height': 1080, 'file_name': '97b37728-21bc-11ea-a13a-137349068a90.jpg',
                'sub_location': 0, 'seq_frame_num': 8, 'category_id': 372, 'gps_location': {'latitude': 17.492381819738956,
                'longitude': -89.21560449646441}, 'gps_sub_location': {'latitude': -2.6262412503467187,
                'longitude': 29.35055652840072}, 'detection': {'file': 'train/97b37728-21bc-11ea-a13a-137349068a90.jpg',
                'max_detection_conf': 0.999, 'detections': [{'category': '1', 'conf': 0.999, 'bbox': [0.153, 0.366, 0.186, 0.535]},
                {'category': '1', 'conf': 0.999, 'bbox': [0.355, 0.451, 0.288, 0.518]}, {'category': '1', 'conf': 0.971, 'bbox': [0, 0.338, 0.056, 0.343]},
                {'category': '1', 'conf': 0.715, 'bbox': [0.978, 0.649, 0.021, 0.078]}, {'category': '1', 'conf': 0.353, 'bbox': [0.002, 0.272, 0.055, 0.177]},
                {'category': '2', 'conf': 0.251, 'bbox': [0.003, 0.27, 0.051, 0.172]}, {'category': '1', 'conf': 0.169, 'bbox': [0.002, 0.281, 0.057, 0.282]}]}})]}
            """

            file_name = image_annotaion["file_name"]

            # Load image (BGR -> RGB)
            image = cv2.imread(
                str(self.dataset_rootdir / "train" / "train" / file_name)
            )[..., ::-1]
            video[frame_idx] = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .contiguous()
            )

        if self.transform is not None:
            video = self.transform(video)

        video = video / 255.0
        return {
            "index": int(index),
            "labels": int(counts),
            "num_frames": int(num_frames),
            "video": video,
        }

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.index_to_seq_id)
