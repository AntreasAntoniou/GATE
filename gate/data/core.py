import functools
import json
import logging
import traceback
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)


class CustomConcatDataset(Dataset):
    """
    A custom PyTorch Dataset class that concatenates multiple datasets.
    📚 Useful for combining data from different sources while maintaining
    the Dataset interface.
    """

    def __init__(self, datasets):
        """
        Constructor for the CustomConcatDataset class.

        :param datasets: A list of PyTorch Dataset objects to concatenate.
        """
        self.datasets = datasets

    def __len__(self):
        """
        Calculate the total length of the concatenated datasets.
        🔢 Returns the sum of the lengths of all individual datasets.

        :return: The total length of the concatenated datasets.
        """
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        """
        Get an item from the concatenated datasets based on the provided index.
        🔍 It finds the correct dataset and the corresponding index in that
        dataset.

        :param idx: The index of the desired item in the concatenated datasets.
        :return: The item corresponding to the given index.
        """
        dataset_idx = idx % len(self.datasets)
        item_idx = idx // len(self.datasets)
        return self.datasets[dataset_idx][item_idx]


def dict_to_summary(batch: Dict):
    summary_dict = defaultdict(list)

    if not isinstance(batch, dict) and not isinstance(batch, list):
        batch = [batch.__dict__]

    if isinstance(batch, dict):
        batch = [batch]

    for item in batch:
        for key, value in item.items():
            # print(value)
            if hasattr(value, "shape"):
                summary_dict[key].append((str(value.shape), str(value.dtype)))
            elif hasattr(value, "__len__"):
                summary_dict[key].append(len(value))
            elif value is None:
                summary_dict[key].append(None)
            else:
                summary_dict[key].append(value)

    return summary_dict


def dataclass_collate(batch):
    """Collate data from a list of dataclass objects.

    Args:
        batch (list): List of dataclass objects.

    Returns:
        dict: Dictionary of values from the dataclass objects.
    """
    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(lambda x: len(list(x.keys())) != 0, batch))

    try:
        if isinstance(batch[0], dict) or not hasattr(
            batch[0], "__dataclass_fields__"
        ):
            batch = default_collate(batch)
            batch = {key: batch[key][0] for key in batch.keys()}
            return batch
        else:
            batched_dict = {
                key: (
                    default_collate([getattr(sample, key) for sample in batch])
                    if getattr(batch[0], key) is not None
                    else None
                )
                for key in batch[0].__dict__.keys()
            }
            batched_dict = {key: batched_dict[key][0] for key in batched_dict}
            return batch[0].__class__(**batched_dict)
    except Exception as e:
        logger.error(
            f"Current batch we botched up on "
            f"{json.dumps(dict_to_summary(batch), indent=4)}"
        )
        raise e


def pad_and_stack_tensors(tensor_list):
    tensor_list = list(tensor_list)
    for idx, tensor in enumerate(tensor_list):
        if len(tensor.shape) == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            tensor_list[idx] = tensor

    is_ireggular_shape = True if len(tensor_list[0].shape) == 2 else False
    if is_ireggular_shape:
        temp_tensor_list = []
        for tensor in tensor_list:
            temp_tensor_list.extend(tensor.unbind(0))
        tensor_list = temp_tensor_list
    # print(f"total tensor_list: {len(tensor_list)}")
    max_len = max(tensor.size(0) for tensor in tensor_list)
    padded_list = []

    for tensor in tensor_list:
        if tensor.size(0) < max_len:
            padding_size = max_len - tensor.size(0)
            padding = (
                torch.ones(
                    (padding_size),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                * tensor[-1]
            )  # use the last value (eos)
            tensor = torch.cat([tensor, padding], dim=0)
        padded_list.append(tensor)

    if not is_ireggular_shape:
        padded_list = torch.stack(padded_list)
    else:
        padded_list = torch.stack(padded_list).view(-1, 2, max_len)
    return padded_list


def collate_fn_with_token_pad(data):
    batch = defaultdict(lambda: defaultdict(dict))

    def process_value(value):
        if isinstance(value[0], torch.Tensor):
            if value[0].dim() == 0 and value[-1].dim() == 0:
                return torch.stack(value)
            return (
                pad_and_stack_tensors(value)
                if value[0].dtype == torch.long
                else torch.stack(value)
            )
        elif isinstance(value[0], Mapping):
            return collate_fn_with_token_pad(value)
        elif isinstance(value[0], str):
            return value
        else:
            return value

    batch = {}
    for key, values in zip(data[0].keys(), zip(*[d.values() for d in data])):
        batch[key] = process_value(values)

    return batch


def retry_on_exception(func):
    @functools.wraps(func)
    def wrapper(self, index):
        try:
            return func(self, index)
        except Exception as e:
            tb = (
                traceback.format_exc()
            )  # This gets the full traceback as a string
            logger.warning(f"Error at index {index}: {e}\n{tb}")
            print(f"Error at index {index}: {e}\n{tb}")

    return wrapper


class GATEDataset(Dataset):
    """
    The GATEDataset class is a wrapper around another dataset, allowing for key
    remapping and applying a task to the data items.

    📚 Attributes:
        - dataset: The input dataset to wrap around
        - task: An optional task to apply to the data items
        - key_remapper_dict: An optional dictionary for key remapping
    """

    def __init__(
        self,
        dataset: Any,
        infinite_sampling: bool = False,
        transforms: Optional[Any] = None,
        meta_data: Optional[Any] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.infinite_sampling = infinite_sampling
        self.transforms = transforms
        self._meta_data = meta_data

    @property
    def meta_data(self) -> Optional[dict]:
        return self._meta_data

    @meta_data.setter
    def meta_data(self, meta_data: dict) -> None:
        self._meta_data = meta_data
        if meta_data is not None:
            for key, value in meta_data.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)

    def __len__(self) -> int:
        if self.infinite_sampling:
            return int(9 * 10**7)
        return len(self.dataset)

    def _apply_transforms(self, item: Any) -> Any:
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for transform in self.transforms:
                    if transform is not None:
                        item = transform(item)
            else:
                item = self.transforms(item)
        return item

    @retry_on_exception
    def __getitem__(self, index) -> Any:
        if self.infinite_sampling:
            index = index % len(self.dataset)

        item = self.dataset[index]

        item = self._apply_transforms(item)

        # Apply the task to the item if it exists
        return item
