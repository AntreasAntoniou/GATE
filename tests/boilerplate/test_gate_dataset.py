import os

from torch.utils.data import DataLoader, Dataset

from gate.data import GATEDataset
from gate.data.data import build_dataset
from gate.models.task_specific_models.classification.clip import build_model


def test_GATEDataset():
    dataset_name = "beans"
    data_dir = os.path.expanduser("~/.cache/huggingface/")
    set_name = "train"
    beans_dataset = build_dataset(dataset_name, data_dir, set_name)
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
