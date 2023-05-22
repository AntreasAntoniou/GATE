import os
import pathlib

import pytest
import torch
from accelerate.utils import set_seed

from gate.data.few_shot import (
    AircraftFewShotClassificationDataset,
    CIFARFewShotClassificationDataset,
    CUB200FewShotClassificationDataset,
    DescribableTexturesFewShotClassificationDataset,
    FC100FewShotClassificationDataset,
    FungiFewShotClassificationDataset,
    MiniImageNetFewShotClassificationDataset,
    OmniglotFewShotClassificationDataset,
    QuickDrawFewShotClassificationDataset,
    TieredImageNetFewShotClassificationDataset,
    VGGFlowersFewShotClassificationDataset,
)

set_seed(42)

classes_to_test = [
    AircraftFewShotClassificationDataset,
    CUB200FewShotClassificationDataset,
    CIFARFewShotClassificationDataset,
    CUB200FewShotClassificationDataset,
    DescribableTexturesFewShotClassificationDataset,
    FC100FewShotClassificationDataset,
    FungiFewShotClassificationDataset,
    MiniImageNetFewShotClassificationDataset,
    OmniglotFewShotClassificationDataset,
    QuickDrawFewShotClassificationDataset,
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
        variable_num_queries_per_class=True,
        variable_num_classes_per_set=True,
        support_set_input_transform=None,
        query_set_input_transform=None,
        support_set_target_transform=None,
        query_set_target_transform=None,
    )


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

    # assert input_dict["image"]["support_set"].shape == (488, 3, 224, 224)
    # assert input_dict["image"]["query_set"].shape == (460, 3, 224, 224)
    # assert label_dict["image"]["support_set"].shape == torch.Size([488])
    # assert label_dict["image"]["query_set"] == torch.Size([460])
