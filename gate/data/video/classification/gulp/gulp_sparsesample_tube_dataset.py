import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from gulpio2 import GulpDirectory

from . import utils as utils

logger = logging.getLogger(__name__)


class GulpSparsesampleTubeDataset(torch.utils.data.Dataset):
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
        tube_pkl_path: str | Path,
        train_jitter_min=256,
        train_jitter_max=320,
        train_horizontal_flip=True,
        test_scale=256,
        test_num_spatial_crops=10,
        crop_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        normalise=True,  # divide pixels by 255
        bgr=False,
        greyscale=False,
        sample_index_code="pyvideoai",
        flow=None,  # If "grey", each image is a 2D array of shape (H, W).
        #           Optical flow has to be saved like
        #           (u1, v1, u2, v2, u3, v3, u4, v4, ...)
        #           So when indexing the frame dimension,
        #           it should read [frame*2, frame*2+1] for each frame.
        #           Also, the CSV file has to have the actual number of frames,
        #           instead of doubled number of frames because of the two channels.
        # If "RG", each image is an 3D array of shape (H, W, 3),
        #           and we're using R and G channels for the u and v optical flow channels.
        frame_neighbours=1,  # How many frames to stack.
        video_id_to_label: dict = None,  # Pass a dictionary of mapping video ID to labels, and it will ignore the label in the CSV and get labels from here. Useful when using unsupported label types such as soft labels.
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
            sample_index_code (str): Options include `pyvideoai`, `TSN`.
                Slightly different implementation of how video is sampled
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

        self._tube_pkl_path = tube_pkl_path

        self.sample_index_code = sample_index_code.lower()

        self.train_jitter_min = train_jitter_min
        self.train_jitter_max = train_jitter_max

        self.test_scale = test_scale

        self.train_horizontal_flip = train_horizontal_flip

        self.num_frames = num_frames

        self.crop_size = crop_size

        if greyscale:
            assert len(mean) == 1
            assert len(std) == 1
            assert not bgr, "Greyscale and BGR can't be set at the same time."
        else:
            assert len(mean) in [1, 3]
            assert len(std) in [1, 3]
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        self.normalise = normalise
        self.bgr = bgr
        self.greyscale = greyscale

        if flow is not None:
            self.flow = flow.lower()
            assert self.flow in [
                "grey",
                "rg",
            ], f"Optical flow mode must be either grey or RG but got {flow}"

            self.frame_neighbours = frame_neighbours

            assert len(mean) in [1, 2]
            assert len(std) in [1, 2]
            assert (
                not greyscale
            ), "For optical flow data, it is impossible to use greyscale."
            assert (
                not bgr
            ), "For optical flow data, it is impossible to use BGR channel ordering."
        else:
            self.flow = None
            self.frame_neighbours = frame_neighbours

        self.video_id_to_label = video_id_to_label
        if video_id_to_label is not None:
            logger.info(
                "video_id_to_label is provided. It will replace the labels in the CSV file."
            )

        # For training mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode == "train":
            self._num_clips = 1
        elif self.mode == "test":
            self._num_clips = test_num_spatial_crops

        assert test_num_spatial_crops in [
            1,
            5,
            10,
        ], "1 for centre, 5 for centre and four corners, 10 for their horizontal flips"
        self.test_num_spatial_crops = test_num_spatial_crops

        logger.info(f"Constructing gulp video dataset {mode=}...")
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        assert os.path.exists(self._csv_file), "{} not found".format(self._csv_file)

        self._gulp_keys = []
        self._video_ids = []
        self._labels = []
        self._start_frames = []  # number of sample video frames
        self._end_frames = []  # number of sample video frames
        self._spatial_temporal_idx = []
        # each entry is a dictionary with ['keypoint', 'keypoint_score', 'frame_dir', 'total_frames', 'original_shape', 'img_shape', 'label'] keys.
        self._tubes = []

        with open(self._tube_pkl_path, "rb") as f:
            tube_data = pickle.load(f, encoding="latin1")

        with open(self._csv_file, "r") as f:
            self.num_classes = int(f.readline())

            for clip_idx, key_label in enumerate(f.read().splitlines()):
                assert len(key_label.split()) == 5
                gulp_key, video_id, label, start_frame, end_frame = key_label.split()

                # frame_dir is the gulp key without the class name and a slash
                self._tubes.append(tube_data["gttubes"][gulp_key])

                if self.video_id_to_label is None:
                    labels = label.split(";")

                    label_all_heads = []
                    for label in labels:
                        if self.num_classes > 0:
                            label_list = label.split(",")
                            label = np.zeros(self.num_classes, dtype=np.float32)
                            for label_idx in label_list:
                                label[int(label_idx)] = 1.0  # one hot encoding
                        else:
                            label = int(label)
                        label_all_heads.append(label)
                else:
                    label = self.video_id_to_label[int(video_id)]

                if len(label_all_heads) == 1:
                    # single head. Just use the element than the array.
                    multi_head = False
                    label = label_all_heads[0]
                else:
                    multi_head = True
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
            f"Constructing gulp video dataloader (size: {len(self)}) from {self._csv_file}"
        )

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

        crop_size = self.crop_size
        if self.mode == "train":
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.train_jitter_min
            max_scale = self.train_jitter_max
            sample_uniform = False
        elif self.mode == "test":
            # spatial_sample_index is in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self.test_num_spatial_crops
            )
            min_scale, max_scale = [self.test_scale] * 2
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale are expect to be the same.
            assert len({min_scale, max_scale}) == 1
            sample_uniform = True
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        # Decode video. Meta info is used to perform selective decoding.
        #        frame_indices = utils.TRN_sample_indices(self._num_sample_frames[index], self.num_frames, mode = self.mode)
        num_video_frames = self._end_frames[index] - self._start_frames[index] + 1
        if self.sample_index_code == "pyvideoai":
            frame_indices = utils.sparse_frame_indices(
                num_video_frames,
                self.num_frames,
                uniform=sample_uniform,
                num_neighbours=self.frame_neighbours,
            )
        elif self.sample_index_code == "tsn":
            frame_indices = utils.TSN_sample_indices(
                num_video_frames,
                self.num_frames,
                mode=self.mode,
                new_length=self.frame_neighbours,
            )
        else:
            raise ValueError(
                f"Wrong self.sample_index_code: {self.sample_index_code}. Should be pyvideoai, TSN, TDN"
            )

        frame_indices = [
            idx + self._start_frames[index] for idx in frame_indices
        ]  # add offset (frame number start)

        if self.flow == "grey":
            # Frames are saved as (u0, v0, u1, v1, ...)
            # Read pairs of greyscale images.
            frame_indices = [idx * 2 + uv for idx in frame_indices for uv in range(2)]
            frames = np.stack(
                self.gulp_dir[self._gulp_keys[index], frame_indices][0]
            )  # (T*2, H, W)
            TC, H, W = frames.shape
            frames = np.reshape(frames, (TC // 2, 2, H, W))  # (T, C=2, H, W)
            frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, C=2)
        else:
            frames = np.stack(
                self.gulp_dir[self._gulp_keys[index], frame_indices][0]
            )  # (T, H, W, C=3)
            # or if greyscale images, (T, H, W)
            if frames.ndim == 3:
                # Greyscale images. (T, H, W) -> (T, H, W, 1)
                frames = np.expand_dims(frames, axis=-1)

            if self.flow == "rg":
                frames = frames[
                    ..., 0:2
                ]  # Use R and G as u and v (x,y). Discard B channel.

        if self.bgr:
            frames = frames[..., ::-1]

        if self.greyscale:
            raise NotImplementedError()

        frames = torch.from_numpy(frames)

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.mean, self.std, normalise=self.normalise
        )

        # Reshape so that neighbouring frames go in the channel dimension.
        _, H, W, _ = frames.shape
        # T*neighbours, H, W, C -> T*neighbours, C, H, W
        frames = frames.permute(0, 3, 1, 2)
        if self.flow is not None:
            frames = frames.reshape(
                self.num_frames, 2 * self.frame_neighbours, H, W
            )  # T, C=2*neighbours, H, W
        else:
            frames = frames.reshape(
                self.num_frames, 3 * self.frame_neighbours, H, W
            )  # T, C=3*neighbours, H, W
        # T, C, H, W -> C, T, H, W
        frames = frames.permute(1, 0, 2, 3)

        # Perform data augmentation.
        (
            frames,
            scale_factor_width,
            scale_factor_height,
            x_offset,
            y_offset,
            is_flipped,
        ) = utils.spatial_sampling_5(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.train_horizontal_flip,
        )

        video_id = self._video_ids[index]
        label = self._labels[index]

        return {
            "pixel_values": frames,
            "video_ids": video_id,
            "labels": label,
            "spatial_sample_indices": spatial_sample_index,
            "indices": index,
            "frame_indices": np.array(frame_indices),
            "scale_factor_width": scale_factor_width,
            "scale_factor_height": scale_factor_height,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "is_flipped": is_flipped,
            "tubes": self._tubes[index],
        }

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._gulp_keys)
