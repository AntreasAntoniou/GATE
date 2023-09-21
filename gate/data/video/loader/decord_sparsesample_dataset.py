# Code inspired from https://github.com/facebookresearch/SlowFast
import functools
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable

import decord
import numpy as np
import torch
import torch.utils.data
from decord import VideoReader

from . import transform as transform
from . import utils as utils

logger = logging.getLogger(__name__)


def get_next_on_error(func: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that catches exceptions in the wrapped function.

    If an exception occurs, it re-runs the function with the next index in sequence.

    Args:
        func: The function to decorate.

    Returns:
        A function with the same signature as `func`, but that catches exceptions and re-runs.
    """

    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.info(
                f"Error occurred at idx {args[1]} {e}, getting the next item instead."
            )
            args = list(args)
            args[1] = args[1] + 1
            args = tuple(args)
            return func(*args, **kwargs)

    return wrapper_collect_metrics


class DecordSparsesampleDataset(torch.utils.data.Dataset):
    """
    Video loader. Construct the video loader, then sample
    clips from the videos. For training, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the center
    and four corners.
    """

    def __init__(
        self,
        csv_file,
        num_frames,
        video_height: int = 224,
        video_width: int = 224,
        path_prefix: str | Path = "",
        sample_index_code="pyvideoai",
        num_decord_threads=1,
    ):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        num_classes     # set it to zero for single label. Only needed for multilabel.
        path_to_video_1.mp4 video_id_1 label_1 start_frame_1 end_frame_1 width_1 height_2
        path_to_video_2.mp4 video_id_2 label_2 start_frame_2 end_frame_2 width_2 height_2
        ...
        path_to_video_3.mp4 video_id_N label_N start_frame_N end_frame_N width_N height_N
        ```
        Args:
            mode (str): Options includes `train`, or `test` mode.
                For the train, the data loader will take data
                from the train set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            sample_index_code (str): Options include `pyvideoai`, `TSN` and `TDN`.
                Slightly different implementation of how video is sampled (pyvideoai and TSN),
                and for the TDN, it is completely different as it samples num_frames*5 frames.
        """

        self._csv_file = csv_file
        self._path_prefix = path_prefix
        self._num_decord_threads = num_decord_threads
        self.sample_index_code = sample_index_code.lower()
        self.video_height = video_height
        self.video_width = video_width

        self.num_frames = num_frames

        self._num_clips = 1

        self._construct_loader()

        decord.bridge.set_bridge("torch")

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(
            self._csv_file
        )

        self._path_to_videos = []
        self._video_ids = []
        self._labels = []
        self._start_frames = []  # number of sample video frames
        self._end_frames = []  # number of sample video frames
        self._widths = []
        self._heights = []
        self._spatial_temporal_idx = []
        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 7
                (
                    path,
                    video_id,
                    label,
                    start_frame,
                    end_frame,
                    width,
                    height,
                ) = path_label.split()

                if self.num_classes > 0:
                    label_list = label.split(",")
                    label = np.zeros(self.num_classes, dtype=np.float32)
                    for label_idx in label_list:
                        label[int(label_idx)] = 1.0  # one hot encoding
                else:
                    label = int(label)

                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self._path_prefix, path)
                    )

                    self._video_ids.append(int(video_id))
                    self._labels.append(label)
                    self._start_frames.append(int(start_frame))
                    self._end_frames.append(int(end_frame))
                    self._widths.append(int(width))
                    self._heights.append(int(height))
                    self._spatial_temporal_idx.append(idx)
        assert (
            len(self._path_to_videos) > 0
        ), f"Failed to load video loader from {self._csv_file}"
        logger.info(
            "Constructing video dataloader (size: {}) from {}".format(
                len(self._path_to_videos), self._csv_file
            )
        )

    def filter_samples(self, video_ids: list):
        """Given a video_ids list, filter the samples.
        Used for visualisation.
        """
        indices_of_video_ids = [
            x for x, v in enumerate(self._video_ids) if v in video_ids
        ]

        self._path_to_videos = [
            self._path_to_videos[x] for x in indices_of_video_ids
        ]
        self._video_ids = [self._video_ids[x] for x in indices_of_video_ids]
        self._labels = [self._labels[x] for x in indices_of_video_ids]
        self._start_frames = [
            self._start_frames[x] for x in indices_of_video_ids
        ]
        self._end_frames = [self._end_frames[x] for x in indices_of_video_ids]
        self._widths = [self._widths[x] for x in indices_of_video_ids]
        self._heights = [self._heights[x] for x in indices_of_video_ids]
        self._spatial_temporal_idx = [
            self._spatial_temporal_idx[x] for x in indices_of_video_ids
        ]

    @get_next_on_error
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            video_id (int): the ID of the current video.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        # -1 indicates random sampling.
        spatial_sample_index = -1
        sample_uniform = False

        num_video_frames = (
            self._end_frames[index] - self._start_frames[index] + 1
        )
        if self.sample_index_code == "pyvideoai":
            frame_indices = utils.sparse_frame_indices(
                num_video_frames, self.num_frames, uniform=sample_uniform
            )
        else:
            raise ValueError(
                f"Wrong self.sample_index_code: {self.sample_index_code}. Should be pyvideoai"
            )
        frame_indices = [
            idx + self._start_frames[index] for idx in frame_indices
        ]  # add offset (frame number start)

        vr = VideoReader(
            self._path_to_videos[index],
            width=self.video_width,
            height=self.video_height,
            num_threads=self._num_decord_threads,
        )
        frames = vr.get_batch(frame_indices)

        # T, H, W, C -> C, T, H, W
        frames = frames.permute(3, 0, 1, 2)

        # Perform data augmentation.
        (
            frames,
            _,
            _,
            x_offset,
            y_offset,
            is_flipped,
        ) = utils.spatial_sampling_5(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=None,  # Already rescaled using decord
            max_scale=None,
            crop_size=self.video_height,
            random_horizontal_flip=False,
        )

        video_id = self._video_ids[index]
        label = self._labels[index]
        frames = frames.permute(1, 0, 2, 3)  # C, T, H, W -> T, C, H, W
        frames = frames.float() / 255.0
        # BGR to RGB
        # frames = frames[:, 0, :, :]
        return {
            "video": frames,
            "video_ids": video_id,
            "labels": label,
            "spatial_sample_indices": spatial_sample_index,
            "indices": index,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "is_flipped": is_flipped,
            "frame_indices": np.array(frame_indices),
        }

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
