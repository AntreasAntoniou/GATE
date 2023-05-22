import os

from torch.utils.data import DataLoader, Dataset

from gate.data.core import GATEDataset
from gate.data.image.classification.cifar100 import build_cifar100_dataset
from gate.models.task_specific_models.classification.clip import build_model


def test_GATEDataset():
    data_dir = os.path.expanduser("~/.cache/huggingface/")
    set_name = "train"
    beans_dataset = build_cifar100_dataset(
        set_name,
        data_dir,
    )
    model_and_transforms = build_model()
    transforms = model_and_transforms.transform

    gate_dataset = GATEDataset(
        dataset=beans_dataset,
        key_remapper_dict={"pixel_values": "image"},
        transforms=transforms,
    )
    assert isinstance(gate_dataset, Dataset)

    dataloader = DataLoader(gate_dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        # Ensure the keys were remapped
        assert "image" in batch
        assert "img" not in batch
