import os
import pathlib

import pytest
from accelerate.utils import set_seed

from gate.data.few_shot.aircraft import AircraftFewShotClassificationDataset
from gate.data.few_shot.aircraft import \
    build_gate_dataset as build_gate_dataset_aircraft
from gate.data.few_shot.cubirds200 import CUB200FewShotClassificationDataset
from gate.data.few_shot.cubirds200 import \
    build_gate_dataset as build_gate_dataset_cubirds200
from gate.data.few_shot.describable_textures import \
    DescribableTexturesFewShotClassificationDataset
from gate.data.few_shot.describable_textures import \
    build_gate_dataset as build_gate_dataset_describable_textures
from gate.data.few_shot.fungi import FungiFewShotClassificationDataset
from gate.data.few_shot.fungi import \
    build_gate_dataset as build_gate_dataset_fungi
from gate.data.few_shot.mini_imagenet import \
    MiniImageNetFewShotClassificationDataset
from gate.data.few_shot.mini_imagenet import \
    build_gate_dataset as build_gate_dataset_mini_imagenet
from gate.data.few_shot.omniglot import OmniglotFewShotClassificationDataset
from gate.data.few_shot.omniglot import \
    build_gate_dataset as build_gate_dataset_omniglot

set_seed(42)

classes_to_test = [
    AircraftFewShotClassificationDataset,
    CUB200FewShotClassificationDataset,
    DescribableTexturesFewShotClassificationDataset,
    FungiFewShotClassificationDataset,
    MiniImageNetFewShotClassificationDataset,
    OmniglotFewShotClassificationDataset,
]

gate_classes_to_test = [
    build_gate_dataset_aircraft,
    build_gate_dataset_cubirds200,
    build_gate_dataset_describable_textures,
    build_gate_dataset_fungi,
    build_gate_dataset_mini_imagenet,
    build_gate_dataset_omniglot,
]


def dataset_init(DATASET_CLASS):
    dataset_root = pathlib.Path(os.environ["PYTEST_DIR"])
    return DATASET_CLASS(
        dataset_root=dataset_root,
        split_name="train",
        download=True,
        num_episodes=100000,
        min_num_classes_per_set=2,
        min_num_samples_per_class=1,
        min_num_queries_per_class=1,
        num_classes_per_set=50,
        num_samples_per_class=5,
        num_queries_per_class=15,
        variable_num_samples_per_class=True,
        variable_num_classes_per_set=True,
        support_set_input_transform=None,
        query_set_input_transform=None,
        support_set_target_transform=None,
        query_set_target_transform=None,
    )


def gate_dataset_init(DATASET_CLASS):
    dataset_root = pathlib.Path(os.environ["PYTEST_DIR"])
    return DATASET_CLASS(data_dir=dataset_root, transforms=None)


@pytest.mark.parametrize("dataset_item", classes_to_test)
def test_dataset_creation(
    dataset_item,
):
    dataset_instance = dataset_init(dataset_item)
    assert isinstance(dataset_instance, dataset_item)


@pytest.mark.parametrize("dataset_item", classes_to_test)
def test_dataset_length(
    dataset_item,
):
    dataset_item = dataset_init(dataset_item)
    assert len(dataset_item) == 100000


@pytest.mark.parametrize("dataset_item", classes_to_test)
def test_dataset_sample(
    dataset_item,
):
    dataset_item = dataset_init(dataset_item)
    input_dict, label_dict = dataset_item[0]
    for key, value in input_dict.items():
        for subkey, subvalue in value.items():
            print(f"{key}.{subkey}: {subvalue.shape}")

    for key, value in label_dict.items():
        for subkey, subvalue in value.items():
            print(f"{key}.{subkey}: {subvalue.shape}")


@pytest.mark.parametrize("dataset_item", gate_classes_to_test)
def test_gate_dataset_creation(
    dataset_item,
):
    dataset_instance = gate_dataset_init(dataset_item)
    assert isinstance(dataset_instance, dict)
