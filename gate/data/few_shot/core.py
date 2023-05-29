import json
import pathlib
from typing import Any, Counter, Dict, List, Optional

import datasets
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from gate.boilerplate.utils import get_logger, load_json, save_json
from gate.data.few_shot.utils import (
    FewShotSuperSplitSetOptions,
    get_class_to_idx_dict,
    get_class_to_image_idx_and_bbox,
)

logger = get_logger(
    __name__,
)


# convert a list of dicts into a dict of lists
def list_of_dicts_to_dict_of_lists(list_of_dicts):
    return {
        key: [x[key] for x in list_of_dicts] for key in list_of_dicts[0].keys()
    }


class FewShotClassificationMetaDataset(Dataset):
    """
    A class to represent a Few-Shot Classification Meta-Dataset using
    TensorFlow Datasets (TFDS).

    The class supports variable number of samples per class, queries per class,
    and classes per set.
    The class also allows for custom transformations to be applied on the s
    upport and query sets.

    Attributes:
    ----------
    dataset_name: str
        Name of the dataset.
    dataset_root: str
        Root directory of the dataset.
    split_name: str
        Name of the split.
    download: bool
        Flag to download the dataset if not available locally.
    num_episodes: int
        Number of episodes.
    min_num_classes_per_set: int
        Minimum number of classes per set.
    min_num_samples_per_class: int
        Minimum number of samples per class.
    min_num_queries_per_class: int
        Minimum number of queries per class.
    num_classes_per_set: int
        Number of classes per set (n-way).
    num_samples_per_class: int
        Number of samples per class (n-shot).
    num_queries_per_class: int
        Number of queries per class.
    variable_num_samples_per_class: bool
        Flag to indicate if the number of samples per class varies.
    variable_num_classes_per_set: bool
        Flag to indicate if the number of classes per set varies.
    input_shape_dict: Dict
        Dictionary representing the input shape.
    input_target_annotation_keys: Dict
        Dictionary representing the input target annotation keys.
    subset_split_name_list: Optional[List[str]]
        List of subset split names.
    split_percentage: Optional[Dict[str, float]]
        Dictionary representing the split percentages.
    split_config: Optional[DictConfig]
        Split configuration, a dict of lists with keys=split_name,
        and values=classes to use for that set.
    support_set_input_transform: Any
        Transformations to be applied on the support set inputs.
    query_set_input_transform: Any
        Transformations to be applied on the query set inputs.
    support_set_target_transform: Any
        Transformations to be applied on the support set targets.
    query_set_target_transform: Any
        Transformations to be applied on the query set targets.
    label_extractor_fn: Optional[Any]
        Function to extract labels.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        dataset_class: Any,
        split_name: str,
        num_episodes: int,
        min_num_classes_per_set: int,
        min_num_samples_per_class: int,
        min_num_queries_per_class: int,
        num_classes_per_set: int,  # n_way
        num_samples_per_class: int,  # n_shot
        num_queries_per_class: int,
        variable_num_samples_per_class: bool,
        variable_num_classes_per_set: bool,
        input_target_annotation_keys: Dict,
        subset_split_name_list: Optional[List[str]] = None,
        split_percentage: Optional[Dict[str, float]] = None,
        split_config: Optional[DictConfig] = None,
        split_as_original: Optional[bool] = False,
        support_set_input_transform: Any = None,
        query_set_input_transform: Any = None,
        support_set_target_transform: Any = None,
        query_set_target_transform: Any = None,
        preprocess_transforms: Optional[Any] = None,
        max_support_set_size: Optional[int] = 250,
    ):
        super(FewShotClassificationMetaDataset, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_root = pathlib.Path(dataset_root)
        self.dataset_class = dataset_class
        self.input_target_annotation_keys = input_target_annotation_keys
        self.num_episodes = num_episodes
        self.split_config = split_config
        self.preprocess_transform = preprocess_transforms
        self.split_as_original = split_as_original

        self._validate_samples_and_classes(
            min_num_samples_per_class,
            num_samples_per_class,
            min_num_classes_per_set,
            num_classes_per_set,
        )

        self.min_num_classes_per_set = min_num_classes_per_set
        self.min_num_samples_per_class = min_num_samples_per_class
        self.min_num_queries_per_class = min_num_queries_per_class
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.num_queries_per_class = num_queries_per_class
        self.max_support_set_size = max_support_set_size
        self.variable_num_samples_per_class = variable_num_samples_per_class
        self.variable_num_classes_per_set = variable_num_classes_per_set

        self.support_set_input_transform = support_set_input_transform
        self.query_set_input_transform = query_set_input_transform
        self.support_set_target_transform = support_set_target_transform
        self.query_set_target_transform = query_set_target_transform

        self.split_name = split_name
        self.split_percentage = split_percentage

        if subset_split_name_list is None:
            subset_split_name_list = ["train", "test"]

        self.dataset = self._load_subsets(subset_split_name_list)

        class_to_address_dict_path = (
            self.dataset_root
            / self.dataset_name
            / "class_to_address_dict.json"
        )
        # if class_to_address_dict_path.exists():
        #     self.class_to_address_dict = load_json(
        #         filepath=class_to_address_dict_path
        #     )
        # else:
        self.class_to_address_dict = get_class_to_idx_dict(
            dataset=self.dataset,
            class_name_key=self.input_target_annotation_keys[
                "target_annotations"
            ],
        )
        # save_json(
        #     filepath=class_to_address_dict_path,
        #     dict_to_store=self.class_to_address_dict,
        #     overwrite=True,
        # )
        print(self.class_to_address_dict)

        self.current_class_to_address_dict = (
            self._get_current_class_to_address_dict()
        )
        # logger.info(
        #     f"Current class to address dict: {self.current_class_to_address_dict}"
        # )

    def _validate_samples_and_classes(
        self,
        min_num_samples_per_class,
        num_samples_per_class,
        min_num_classes_per_set,
        num_classes_per_set,
    ):
        """Validate the number of samples and classes per set."""
        assert min_num_samples_per_class < num_samples_per_class, (
            f"min_num_samples_per_class {min_num_samples_per_class} "
            f"must be less than "
            f"num_samples_per_class {num_samples_per_class}"
        )

        assert min_num_classes_per_set < num_classes_per_set, (
            f"min_num_classes_per_set {min_num_classes_per_set} "
            f"must be less than "
            f"num_classes_per_set {num_classes_per_set}"
        )

    def _load_subsets(self, subset_split_name_list):
        """Load and process the subsets."""
        dataset_path = self.dataset_root / self.dataset_name
        state_path = dataset_path / "dataset_info.json"

        # if state_path.exists():
        #     return datasets.Dataset.load_from_disk(dataset_path)

        subsets = [
            self.dataset_class(
                subset_name,
            )
            for subset_name in subset_split_name_list
        ]
        datapoints = []
        label_set = set()

        for set_name, subset in tqdm(zip(subset_split_name_list, subsets)):
            with tqdm(total=len(subset)) as pbar:
                for sample in subset:
                    sample = self._process_sample(sample)
                    label_set.add(sample["label"])
                    sample["label"] = f"{set_name}-{sample['label']}"

                    datapoints.append(sample)
                    pbar.update(1)
                    pbar.set_description(
                        f"Label {sample['label']}, {set_name}, {len(label_set)}"
                    )

        print(f"Number of classes: {len(label_set)}")
        dataset = datasets.Dataset.from_list(datapoints)

        # Save the dataset to a directory
        # dataset.save_to_disk(self.dataset_root / self.dataset_name)
        return dataset

    def _process_sample(self, sample):
        """Process a sample and return its numpy representation."""
        if self.preprocess_transform is not None:
            return self.preprocess_transform(sample)
        return sample

    def _get_current_class_to_address_dict(self):
        """Get current class to address dict based on split config."""
        if self.split_config is None:
            return self._get_dict_based_on_split_percentage()
        else:
            return {
                label_name: self.class_to_address_dict[label_name]
                for label_name in self.split_config[self.split_name]
            }

    def _get_dict_based_on_split_percentage(self):
        """Get current class to address dict based on split percentage."""
        start_idx, end_idx = self._get_start_end_indices()
        print(
            f"set name: {self.split_name}, start: {start_idx}, end: {end_idx}"
        )
        split_dict = {
            key: value
            for idx, (key, value) in enumerate(
                self.class_to_address_dict.items()
            )
            if start_idx <= idx < end_idx
        }
        split_len_dict = {key: len(value) for key, value in split_dict.items()}
        print(f"set_name {self.split_name} split dict: {split_len_dict}")
        print(f"set_name {self.split_name} split dict len: {len(split_dict)}")
        class_to_address_dict_len = {
            key: len(value)
            for key, value in self.class_to_address_dict.items()
        }
        print(f"total dict: {class_to_address_dict_len}")
        print(f"total dict len: {len(self.class_to_address_dict)}")
        return split_dict

    def _get_start_end_indices(self):
        """Get start and end indices based on split name."""
        train_percentage = self.split_percentage[
            FewShotSuperSplitSetOptions.TRAIN
        ]
        val_percentage = self.split_percentage[FewShotSuperSplitSetOptions.VAL]

        if self.split_name == FewShotSuperSplitSetOptions.TRAIN:
            return 0, train_percentage
        elif self.split_name == FewShotSuperSplitSetOptions.VAL:
            return train_percentage, train_percentage + val_percentage
        elif self.split_name == FewShotSuperSplitSetOptions.TEST:
            return (
                train_percentage + val_percentage,
                100,
            )  # Assuming total to be 100%

    def __len__(self):
        return self.num_episodes

    def _calculate_num_classes_per_set(self, rng):
        """Calculate the number of classes per set."""
        return (
            rng.choice(
                range(self.min_num_classes_per_set, self.num_classes_per_set)
            )
            if self.variable_num_classes_per_set
            else self.num_classes_per_set
        )

    def _prepare_for_sample_selection(self, selected_classes_for_set):
        """Prepare for the sample selection process."""

        class_to_num_available_samples = {
            class_name: len(self.current_class_to_address_dict[class_name])
            for class_name in selected_classes_for_set
        }
        # #logger.debug(
        #     f"Class to num available samples: {class_to_num_available_samples}"
        # )
        return class_to_num_available_samples

    def _calculate_num_query_samples_per_class(
        self, class_to_num_available_samples
    ):
        """Calculate the number of query samples per class."""
        # logger.info(
        #     f"Class to num available samples: {class_to_num_available_samples}"
        # )
        print(class_to_num_available_samples)
        min_available_shots = min(
            [value for value in class_to_num_available_samples.values()]
        )
        num_query_samples_per_class = int(np.floor(min_available_shots * 0.5))
        num_query_samples_per_class = max(num_query_samples_per_class, 1)
        return min(num_query_samples_per_class, self.num_queries_per_class)

    def _prepare_support_and_query_sets(
        self,
        class_name,
        num_query_samples_per_class,
        rng,
        idx,
        support_set_inputs,
    ):
        """Prepare the support and query sets."""

        max_support_set_size = self.max_support_set_size
        max_per_class_support_set_size = 50
        available_support_set_size = (
            max_support_set_size - len(support_set_inputs) - idx
        )
        try:
            if self.variable_num_samples_per_class:
                num_support_samples_per_class = rng.choice(
                    range(
                        self.min_num_samples_per_class,
                        min(
                            self.class_to_num_available_samples[class_name],
                            available_support_set_size,
                            max_per_class_support_set_size,
                        )
                        - num_query_samples_per_class,
                    )
                )
            else:
                num_support_samples_per_class = self.num_samples_per_class
        except Exception as e:
            # #logger.debug(
            #     f"Exception: {e}, {class_name}, min_num_classes_per_set: {self.min_num_classes_per_set}, "
            #     f"class_to_num_available_samples: {self.class_to_num_available_samples[class_name]}, "
            #     f"available_support_set_size: {available_support_set_size}, "
            #     f"max_per_class_support_set_size: {max_per_class_support_set_size}, "
            #     f"num_query_samples_per_class: {num_query_samples_per_class}, "
            # )
            return None, None

        selected_samples_addresses_idx = rng.choice(
            range(
                len(self.current_class_to_address_dict[class_name]),
            ),
            size=min(
                len(self.current_class_to_address_dict[class_name]),
                num_support_samples_per_class + num_query_samples_per_class,
            ),
            replace=False,
        )
        num_selected_samples = min(
            len(self.current_class_to_address_dict[class_name]),
            num_support_samples_per_class + num_query_samples_per_class,
        )

        print(f"Number of samples selected: {num_selected_samples}")

        selected_samples_addresses = [
            self.current_class_to_address_dict[class_name][sample_address_idx]
            for sample_address_idx in selected_samples_addresses_idx
        ]

        return num_support_samples_per_class, selected_samples_addresses

    def _get_data_inputs_and_labels(self, selected_samples_addresses):
        """Get the data inputs and labels."""
        data_inputs = [
            self.dataset[idx][self.input_target_annotation_keys["inputs"]]
            for idx in selected_samples_addresses
        ]

        data_labels = [
            self.dataset[idx][
                self.input_target_annotation_keys["target_annotations"]
            ]
            for idx in selected_samples_addresses
        ]

        return data_inputs, data_labels

    def _shuffle_data(self, data_inputs, data_labels, rng):
        """Shuffle the data."""
        shuffled_idx = rng.permutation(len(data_inputs))
        data_inputs = [data_inputs[i] for i in shuffled_idx]

        if isinstance(data_inputs[0], np.ndarray):
            data_inputs = [
                torch.tensor(sample).permute(2, 0, 1) for sample in data_inputs
            ]

        data_labels = [data_labels[i] for i in shuffled_idx]

        return data_inputs, data_labels

    def _assign_data_to_sets(
        self,
        num_support_samples_per_class,
        num_query_samples_per_class,
        data_inputs,
        data_labels,
        support_set_inputs,
        support_set_labels,
        query_set_inputs,
        query_set_labels,
    ):
        """Assign the data to the support and query sets."""

        logger.debug(
            f"num_support_samples_per_class: {num_support_samples_per_class}, data input length: {len(data_inputs)}"
        )
        print(f"data inputs length: {len(data_inputs)}")

        if len(data_inputs) > num_support_samples_per_class:
            support_set_inputs.extend(
                data_inputs[:num_support_samples_per_class]
            )
            support_set_labels.extend(
                data_labels[:num_support_samples_per_class]
            )
            query_set_inputs.extend(
                data_inputs[
                    num_support_samples_per_class : num_support_samples_per_class
                    + num_query_samples_per_class
                ]
            )
            query_set_labels.extend(
                data_labels[
                    num_support_samples_per_class : num_support_samples_per_class
                    + num_query_samples_per_class
                ]
            )
        else:
            support_set_inputs.extend(data_inputs[:-1])
            support_set_labels.extend(data_labels[:-1])
            query_set_inputs.extend(data_inputs[-1:])
            query_set_labels.extend(data_labels[-1:])
        return (
            support_set_inputs,
            support_set_labels,
            query_set_inputs,
            query_set_labels,
        )

    def _apply_transformations(
        self, inputs, labels, input_transform, target_transform
    ):
        """Apply transformations to the input data and labels."""
        # if input_transform:
        #     inputs = apply_input_transforms(
        #         inputs=inputs, transforms=input_transform
        #     )
        # if target_transform:
        #     labels = apply_target_transforms(
        #         targets=labels, transforms=target_transform
        #     )
        return inputs, labels

    def _convert_to_tensor(self, inputs, labels):
        """Convert input data and labels to tensors."""
        inputs = [
            torch.tensor(input_)
            if isinstance(input_, np.ndarray)
            else T.ToTensor()(Image.open(input_))
            if isinstance(input_, str)
            else input_
            if isinstance(input_, torch.Tensor)
            else T.ToTensor()(input_)
            for input_ in inputs
        ]
        inputs = (
            torch.stack(inputs, dim=0) if isinstance(inputs, list) else inputs
        )
        labels = torch.tensor(labels) if isinstance(labels, list) else labels
        return inputs, labels

    def _log_label_frequency(self, support_set_labels, query_set_labels):
        """Log the frequency of labels in the support and query sets."""
        support_label_frequency_dict = Counter(
            [int(label) for label in support_set_labels]
        )
        query_label_frequency_dict = Counter(
            [int(label) for label in query_set_labels]
        )
        # #logger.debug(
        #     f"Support set label frequency: {support_label_frequency_dict}"
        # )
        # #logger.debug(
        #     f"Query set label frequency: {query_label_frequency_dict}"
        # )

    def _format_input_and_label_data(
        self,
        support_set_inputs,
        support_set_labels,
        query_set_inputs,
        query_set_labels,
    ):
        """Format the input and label data into dictionaries."""
        if not isinstance(support_set_inputs, (Dict, DictConfig)):
            input_dict = dict(
                image=dict(
                    support_set=support_set_inputs, query_set=query_set_inputs
                )
            )

        if not isinstance(support_set_labels, (Dict, DictConfig)):
            label_dict = dict(
                image=dict(
                    support_set=support_set_labels, query_set=query_set_labels
                )
            )

        return input_dict, label_dict

    def __getitem__(self, index):
        """
        Method to get the item at the specified index in the dataset.

        Args:
            index (int): The index of the item to be retrieved.

        Returns:
            input_dict (dict): The dictionary containing the input data.
            label_dict (dict): The dictionary containing the labels for
            the input data.
        """

        rng = np.random.RandomState(
            index
        )  # Initialize a random number generator

        # Initialize the support and query set inputs and labels
        support_set_inputs, support_set_labels = [], []
        query_set_inputs, query_set_labels = [], []

        # Determine the number of classes per set
        num_classes_per_set = self._calculate_num_classes_per_set(rng)
        # logger.debug(f"Number of classes per set: {num_classes_per_set}")

        # Select the classes for the current set
        available_class_labels = list(
            self.current_class_to_address_dict.keys()
        )
        # logger.debug(f"Available class labels: {available_class_labels}")
        selected_classes_for_set = rng.choice(
            available_class_labels,
            size=min(num_classes_per_set, len(available_class_labels)),
        )
        # logger.debug(f"Selected classes for set: {selected_classes_for_set}")

        # Generate mapping from label to local index and prepare for
        # sample selection
        label_idx_to_local_label_idx = {
            label_name: i
            for i, label_name in enumerate(selected_classes_for_set)
        }
        # #logger.debug(
        #     f"Label index to local label index: {label_idx_to_local_label_idx}"
        # )
        self.class_to_num_available_samples = (
            self._prepare_for_sample_selection(selected_classes_for_set)
        )
        # #logger.debug(
        #     f"Class to number of available samples: {self.class_to_num_available_samples}"
        # )

        # Determine the number of query samples per class
        num_query_samples_per_class = (
            self._calculate_num_query_samples_per_class(
                self.class_to_num_available_samples
            )
        )
        # #logger.debug(
        #     f"Number of query samples per class: {num_query_samples_per_class}"
        # )

        # Generate support and query sets for each class
        for idx, class_name in enumerate(selected_classes_for_set):
            # logger.debug(f"Generating set for class: {class_name}")
            (
                num_support_samples_per_class,
                selected_samples_addresses,
            ) = self._prepare_support_and_query_sets(
                class_name=class_name,
                num_query_samples_per_class=num_query_samples_per_class,
                rng=rng,
                idx=len(selected_classes_for_set) - idx,
                support_set_inputs=support_set_inputs,
            )

            if selected_samples_addresses is None:
                continue

            # Get the data inputs and labels
            data_inputs, data_labels = self._get_data_inputs_and_labels(
                selected_samples_addresses
            )

            # Map labels to local index
            data_labels = [
                label_idx_to_local_label_idx[item] for item in data_labels
            ]

            # Shuffle the data
            data_inputs, data_labels = self._shuffle_data(
                data_inputs, data_labels, rng
            )
            # logger.debug(f"num query samples {num_query_samples_per_class}")
            # Assign data to support and query sets
            (
                support_set_inputs,
                support_set_labels,
                query_set_inputs,
                query_set_labels,
            ) = self._assign_data_to_sets(
                num_support_samples_per_class,
                num_query_samples_per_class,
                data_inputs,
                data_labels,
                support_set_inputs,
                support_set_labels,
                query_set_inputs,
                query_set_labels,
            )

        # Apply transformations to the data
        support_set_inputs, support_set_labels = self._apply_transformations(
            support_set_inputs,
            support_set_labels,
            self.support_set_input_transform,
            self.support_set_target_transform,
        )

        query_set_inputs, query_set_labels = self._apply_transformations(
            query_set_inputs,
            query_set_labels,
            self.query_set_input_transform,
            self.query_set_target_transform,
        )

        # Convert data to tensor format
        support_set_inputs, support_set_labels = self._convert_to_tensor(
            support_set_inputs, support_set_labels
        )
        query_set_inputs, query_set_labels = self._convert_to_tensor(
            query_set_inputs, query_set_labels
        )

        # Log the sizes of the support and query sets
        # #logger.debug(
        #     f"Size of support set: {support_set_inputs.shape},"
        #     f"Size of query set: {query_set_inputs.shape}"
        # )

        # Log the frequency of labels in the support and query sets
        self._log_label_frequency(support_set_labels, query_set_labels)

        # Format the input and label data
        input_dict, label_dict = self._format_input_and_label_data(
            support_set_inputs,
            support_set_labels,
            query_set_inputs,
            query_set_labels,
        )

        return input_dict, label_dict
