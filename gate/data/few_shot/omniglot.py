import logging
import multiprocessing as mp
import pathlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import datasets
from torchvision import transforms as T

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.few_shot.core import FewShotClassificationMetaDataset
from gate.data.few_shot.utils import FewShotSuperSplitSetOptions

logger = logging.getLogger(__name__)


def preprocess_transforms(sample: Tuple):
    image_transforms = T.Compose([T.Resize((224, 224))])
    image = image_transforms(sample[0])
    label = sample[1]
    return {"image": image, "label": label}


class OmniglotFewShotClassificationDataset(FewShotClassificationMetaDataset):
    def __init__(
        self,
        dataset_root: Union[str, pathlib.Path],
        split_name: str,
        download: bool,
        num_episodes: int,
        min_num_classes_per_set: int,
        min_num_samples_per_class: int,
        min_num_queries_per_class: int,
        num_classes_per_set: int,  # n_way
        num_samples_per_class: int,  # n_shot
        num_queries_per_class: int,
        variable_num_samples_per_class: bool,
        variable_num_classes_per_set: bool,
        support_set_input_transform: Optional[Any],
        query_set_input_transform: Optional[Any],
        support_set_target_transform: Optional[Any] = None,
        query_set_target_transform: Optional[Any] = None,
    ):
        DATASET_NAME = "metadataset/omniglot"
        super(OmniglotFewShotClassificationDataset, self).__init__(
            dataset_name=DATASET_NAME,
            dataset_root=dataset_root,
            dataset_dict=datasets.load_dataset(
                path="GATE-engine/omniglot",
                cache_dir=dataset_root,
                num_proc=mp.cpu_count(),
            ),
            preprocess_transforms=preprocess_transforms,
            split_name=split_name,
            num_episodes=num_episodes,
            num_classes_per_set=num_classes_per_set,
            num_samples_per_class=num_samples_per_class,
            num_queries_per_class=num_queries_per_class,
            variable_num_samples_per_class=variable_num_samples_per_class,
            variable_num_classes_per_set=variable_num_classes_per_set,
            support_set_input_transform=support_set_input_transform,
            query_set_input_transform=query_set_input_transform,
            support_set_target_transform=support_set_target_transform,
            query_set_target_transform=query_set_target_transform,
            split_percentage={
                FewShotSuperSplitSetOptions.TRAIN: float(1200 / 1623),
                FewShotSuperSplitSetOptions.VAL: float(200 / 1623),
                FewShotSuperSplitSetOptions.TEST: float(223 / 1623),
            },
            min_num_classes_per_set=min_num_classes_per_set,
            min_num_samples_per_class=min_num_samples_per_class,
            min_num_queries_per_class=min_num_queries_per_class,
        )


def build_dataset(set_name: str, num_episodes: int, data_dir: str) -> dict:
    """
    Build a SVHN dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """

    if set_name not in ["train", "val", "test"]:
        raise KeyError(f"Invalid set name: {set_name}")

    dataset_root = pathlib.Path(data_dir)
    data_set = OmniglotFewShotClassificationDataset(
        dataset_root=dataset_root,
        split_name=set_name,
        download=True,
        num_episodes=num_episodes,
        min_num_classes_per_set=2,
        min_num_samples_per_class=2,
        min_num_queries_per_class=2,
        num_classes_per_set=50,
        num_samples_per_class=15,
        num_queries_per_class=5,
        variable_num_samples_per_class=True,
        variable_num_classes_per_set=True,
        support_set_input_transform=None,
        query_set_input_transform=None,
        support_set_target_transform=None,
        query_set_target_transform=None,
    )

    return data_set


single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))
omniglot_transforms = T.Compose([lambda x: 1.0 - x, T.ToPILImage()])


def key_mapper(input_tuple):
    image = input_tuple[0]["image"]
    labels = input_tuple[1]["image"]

    image["support_set"] = [
        omniglot_transforms(item) for item in image["support_set"]
    ]

    image["query_set"] = [
        omniglot_transforms(item) for item in image["query_set"]
    ]

    return {
        "image": image,
        "labels": labels,
    }


@configurable(
    group="dataset",
    name="omniglot-fs-classification",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: str,
    transforms: Optional[Any] = None,
) -> dict:
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir, num_episodes=10000),
        infinite_sampling=True,
        transforms=[key_mapper, transforms],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir, num_episodes=600),
        infinite_sampling=False,
        transforms=[key_mapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir, num_episodes=600),
        infinite_sampling=False,
        transforms=[key_mapper, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 101
