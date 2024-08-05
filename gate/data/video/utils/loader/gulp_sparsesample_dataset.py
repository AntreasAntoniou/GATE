# Code inspired from https://github.com/facebookresearch/SlowFast
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from gulpio2 import GulpDirectory

from . import utils as utils

logger = logging.getLogger(__name__)


class GulpSparsesampleDataset(torch.utils.data.Dataset):
    """
    It uses GulpIO2 instead of reading directly from jpg frames to speed up the IO!
    It will ignore the gulp meta data, and read meta from the CSV instead.

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
        mode,
        num_frames,
        gulp_dir_path: str | Path,
        sample_index_code="pyvideoai",
        processing_backend="pil",  # torch, pil
        video_id_to_label: Optional[
            dict
        ] = None,  # Pass a dictionary of mapping video ID to labels, and it will ignore the label in the CSV and get labels from here. Useful when using unsupported label types such as soft labels.
    ):
        """
        Construct the video loader with a given csv file. The format of
        the csv file is:
        ```
        num_classes     # set it to zero for single label. Only needed for multilabel.
        gulp_key_1 video_id_1 label_1 start_frame_1 end_frame_1
        gulp_key_2 video_id_2 label_2 start_frame_2 end_frame_2
        ...
        gulp_key_N video_id_N label_N start_frame_N end_frame_N
        ```

        `gulp_key` are the gulp dictionary key to access the video segment. Must be string.
        It will access something like this.
        ```
        gulpdata = GulpDirectory(gulp_dir_path)
        frames = gulpdata[gulp_key, [0, 1, 2]][0]   # It will ignore the meta data.
        ```

        Note that the `video_id` must be an integer.

        `label` can be separated with commas for multi-label classification. Remember to set the `num_classes` at the beginning of the file.
            - It will produce an array output of zeros and ones.
        `label` can be also separated with semicolons for multi-head classification (e.g. verb and noun output).
            - It will produce an array output of label indices.
        `label` can be separated with both commas and semicolons.
            - It will produce a 2D array output of zeros and ones.

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
        # Only support train, and test mode.
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported".format(mode)
        self._csv_file = csv_file
        self._gulp_dir_path = gulp_dir_path
        self.gulp_dir = GulpDirectory(gulp_dir_path)
        self.mode = mode
        self.sample_index_code = sample_index_code.lower()

        self.processing_backend = processing_backend.lower()
        assert self.processing_backend in ["torch", "pil"]

        self.num_frames = num_frames

        self.video_id_to_label = video_id_to_label
        if video_id_to_label is not None:
            logger.info(
                "video_id_to_label is provided. It will replace the labels in"
                " the CSV file."
            )

        # For training mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        self._num_clips = 1

        logger.info(f"Constructing gulp video dataset {mode=}...")
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(
            self._csv_file
        )

        self._gulp_keys = []
        self._video_ids = []
        self._labels = []
        self._start_frames = []  # number of sample video frames
        self._end_frames = []  # number of sample video frames
        self._spatial_temporal_idx = []

        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())

            for clip_idx, key_label in enumerate(f.read().splitlines()):
                assert len(key_label.split()) == 5
                (
                    gulp_key,
                    video_id,
                    label,
                    start_frame,
                    end_frame,
                ) = key_label.split()
                label_all_heads = []
                if self.video_id_to_label is None:
                    labels = label.split(";")

                    label_all_heads = []
                    for label in labels:
                        if self.num_classes > 0:
                            label_list = label.split(",")
                            label = np.zeros(
                                self.num_classes, dtype=np.float32
                            )
                            for label_idx in label_list:
                                label[int(label_idx)] = 1.0  # one hot encoding
                        else:
                            label = int(label)
                        label_all_heads.append(label)
                else:
                    label = self.video_id_to_label[int(video_id)]

                if len(label_all_heads) == 1:
                    # single head. Just use the element than the array.
                    label = label_all_heads[0]
                else:
                    label = np.array(label_all_heads)

                for idx in range(self._num_clips):
                    self._gulp_keys.append(gulp_key)
                    self._video_ids.append(int(video_id))
                    self._labels.append(label)
                    self._start_frames.append(int(start_frame))
                    self._end_frames.append(int(end_frame))
                    self._spatial_temporal_idx.append(idx)

        assert (
            len(self._gulp_keys) > 0
        ), f"Failed to load gulp video loader from {self._csv_file}"

        logger.info(
            f"Constructing gulp video dataloader (size: {len(self)}) from"
            f" {self._csv_file}"
        )

    def filter_samples(self, video_ids: list):
        """Given a video_ids list, filter the samples.
        Used for visualisation.
        """
        indices_of_video_ids = [
            x for x, v in enumerate(self._video_ids) if v in video_ids
        ]

        self._gulp_keys = [self._gulp_keys[x] for x in indices_of_video_ids]
        self._video_ids = [self._video_ids[x] for x in indices_of_video_ids]
        self._labels = [self._labels[x] for x in indices_of_video_ids]
        self._start_frames = [
            self._start_frames[x] for x in indices_of_video_ids
        ]
        self._end_frames = [self._end_frames[x] for x in indices_of_video_ids]
        self._spatial_temporal_idx = [
            self._spatial_temporal_idx[x] for x in indices_of_video_ids
        ]

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
            index (int): Note that it will change from the index argument if self.train_class_balanced_sampling is True.
        """

        # -1 indicates random sampling.
        spatial_sample_index = -1
        sample_uniform = False

        # Decode video. Meta info is used to perform selective decoding.
        #        frame_indices = utils.TRN_sample_indices(self._num_sample_frames[index], self.num_frames, mode = self.mode)
        num_video_frames = (
            self._end_frames[index] - self._start_frames[index] + 1
        )
        frame_indices = utils.sparse_frame_indices(
            num_video_frames,
            self.num_frames,
            uniform=sample_uniform,
            num_neighbours=1,
        )

        frame_indices = [
            idx + self._start_frames[index] for idx in frame_indices
        ]  # add offset (frame number start)

        frames = np.stack(
            self.gulp_dir[self._gulp_keys[index], frame_indices][0]
        )  # (T, H, W, C=3)
        # or if greyscale images, (T, H, W)
        if frames.ndim == 3:
            # Greyscale images. (T, H, W) -> (T, H, W, 1)
            frames = np.expand_dims(frames, axis=-1)

        frames = torch.from_numpy(frames)

        # Reshape so that neighbouring frames go in the channel dimension.
        _, H, W, _ = frames.shape
        # T*neighbours, H, W, C -> T*neighbours, C, H, W
        frames = frames.permute(0, 3, 1, 2)

        frames = frames.reshape(
            self.num_frames, 3 * 1, H, W
        )  # T, C=3*neighbours, H, W

        # T, C, H, W -> C, T, H, W
        video_id = self._video_ids[index]
        label = self._labels[index]

        return {
            "video": frames / 255.0,
            "video_ids": video_id,
            "labels": label,
            "spatial_sample_indices": spatial_sample_index,
            "indices": index,
            "frame_indices": np.array(frame_indices),
        }

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._gulp_keys)
