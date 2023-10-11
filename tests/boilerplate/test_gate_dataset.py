import os
from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from gate.data.core import GATEDataset
from gate.data.image.classification.cifar100 import build_dataset
from gate.data.transforms.tiny_image_transforms import pad_image
from gate.models.task_specific_models.classification.clip import build_model


def test_GATEDataset():
    data_dir = os.path.expanduser("~/.cache/huggingface/")
    set_name = "train"
    cifar100_dataset = build_dataset(
        set_name,
        data_dir,
    )
    model_and_transforms = build_model()
    transforms = model_and_transforms.transform

    def transform_wrapper(inputs: Tuple, target_size=224):
        return {
            "image": pad_image(inputs[0], target_size=target_size),
            "labels": inputs[1],
        }

    gate_dataset = GATEDataset(
        dataset=cifar100_dataset,
        transforms=[transform_wrapper, transforms],
    )
    assert isinstance(gate_dataset, Dataset)

    dataloader = DataLoader(gate_dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        # Ensure the keys were remapped
        assert "image" in batch
        assert "labels" in batch
        break
