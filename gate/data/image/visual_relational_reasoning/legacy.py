import pathlib

import tensorflow_datasets as tfds
import torch.utils.data as data
from tqdm.auto import tqdm
from transformers.tf_utils import tf

import datasets


class TFDatasetToHFDatasetAdapter(data.Dataset):
    def __init__(
        self, tf_dataset_path, hf_dataset_name, dataset_dir, set_name
    ):
        self.tf_dataset = lambda split_name: tfds.load(
            tf_dataset_path,
            split=split_name,
            download=True,
            data_dir=pathlib.Path(dataset_dir) / "tfds",
        )
        self.hf_dataset_dir = pathlib.Path(dataset_dir) / hf_dataset_name
        self.hf_dataset_name = hf_dataset_name
        self.set_name = set_name

        if self.hf_dataset_dir.exists():
            self.dataset = datasets.load_from_disk(self.hf_dataset_name)
            print(self.dataset)
        else:
            self.convert_dataset(self.tf_dataset)
            self.dataset = datasets.load_from_disk(self.hf_dataset_name)
            print(self.dataset)

    def convert_dataset(self, tf_dataset):
        sample_list = []
        tf_dataset = self.tf_dataset(self.set_name)
        for tf_example in tqdm(tf_dataset):
            sample_list.append(tf_example)

        dataset = datasets.Dataset.from_list(sample_list, split=self.set_name)
        dataset.save_to_disk(self.hf_dataset_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    import os

    dataset = TFDatasetToHFDatasetAdapter(
        tf_dataset_path="clevr",
        hf_dataset_name="clevr",
        dataset_dir=os.environ.get("PYTEST_DIR"),
        set_name="train",
    )
