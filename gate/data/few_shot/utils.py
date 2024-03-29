import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

import torch.utils.data
from numpy import random
from torch.utils.data import Subset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FewShotSplitSetOptions:
    SUPPORT_SET: str = "train"
    DEV_SET: str = "val"
    QUERY_SET: str = "test"


@dataclass
class FewShotSuperSplitSetOptions:
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"


def collate_resample_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # logging.info(len(batch))
    return torch.utils.data.dataloader.default_collate(batch)


def load_split_datasets(dataset, split_tuple):
    total_length = len(dataset)
    total_idx = [i for i in range(total_length)]

    start_end_index_tuples = [
        (
            int(len(total_idx) * sum(split_tuple[: i - 1])),
            int(len(total_idx) * split_tuple[i]),
        )
        for i in range(len(split_tuple))
    ]

    set_selection_index_lists = [
        total_idx[start_idx:end_idx]
        for (start_idx, end_idx) in start_end_index_tuples
    ]

    return (
        Subset(dataset, set_indices)
        for set_indices in set_selection_index_lists
    )


def get_class_to_idx_dict(
    dataset: Iterator,
):
    class_to_idx_dict = defaultdict(list)
    for sample_idx, sample in tqdm(enumerate(dataset)):
        key = sample["label"]
        class_to_idx_dict[key].append(int(sample_idx))

    temp_class_to_idx_dict = {}
    for key in tqdm(sorted(class_to_idx_dict.keys())):
        temp_class_to_idx_dict[key] = class_to_idx_dict[key]

    return temp_class_to_idx_dict


def get_class_to_image_idx_and_bbox(
    subsets: List[Iterator],
    label_extractor_fn: Optional[Callable] = None,
):
    class_to_image_idx_and_bbox = defaultdict(list)

    for subset_idx, subset in enumerate(subsets):
        for sample_idx, sample in enumerate(subset):
            image = sample["image"]
            objects = sample["objects"]
            object_ids = objects["label"]
            object_bboxes = objects["bbox"]
            for object_idx, object_bbox in zip(object_ids, object_bboxes):
                key = object_idx
                if label_extractor_fn is not None:
                    key = label_extractor_fn(key)

                x_min = int(object_bbox[0] * image.shape[0])
                y_min = int(object_bbox[1] * image.shape[1])
                x_max = int(object_bbox[2] * image.shape[0])
                y_max = int(object_bbox[3] * image.shape[1])

                class_to_image_idx_and_bbox[key].append(
                    dict(
                        subset_idx=int(subset_idx),
                        sample_idx=int(sample_idx),
                        bbox=dict(
                            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
                        ),
                        label=int(object_idx),
                    )
                )

    temp_class_to_image_idx_and_bbox = {}
    for key in sorted(class_to_image_idx_and_bbox.keys()):
        temp_class_to_image_idx_and_bbox[key] = class_to_image_idx_and_bbox[
            key
        ]

    return temp_class_to_image_idx_and_bbox


def collate_fn_replace_corrupted(batch, dataset):
    """Collate function that allows to replace corrupted examples in the
    dataloader. It expect that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are replaced with another examples sampled randomly.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (torch.utils.data.Dataset): dataset which the DataLoader is loading.
            Specify it with functools.partial and pass the resulting partial function that only
            requires 'batch' argument to DataLoader's 'collate_fn' option.

    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783

    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        batch.extend(
            [dataset[random.randint(0, len(dataset))] for _ in range(diff)]
        )
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)
