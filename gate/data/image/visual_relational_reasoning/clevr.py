import os
import pathlib
from typing import Callable, Optional
from PIL import Image

import torch.utils.data as data
import pandas as pd

from gate.data import download_kaggle_dataset


FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 70236


class DiabeticRetinopathyClassification(data.Dataset):
    def __init__(
        self, dataset_path: pathlib.Path, transform: Optional[Callable] = None
    ):
        super().__init__()
        self.dataset_path = dataset_path
        dataset_path_dict = self.download_and_extract(dataset_path)
        self.labels_frame = pd.read_csv(
            dataset_path_dict["dataset_download_path"] / "trainLabels.csv"
        )
        self.img_dir = (
            dataset_path_dict["dataset_download_path"] / "resized_train"
        )
        self.transform = transform

    def download_and_extract(self, dataset_path: pathlib.Path):
        return download_kaggle_dataset(
            dataset_name="clevr",
            dataset_path="timoboz/clevr-dataset",
            target_dir_path=dataset_path,
            count_after_download_and_extract=FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT,
        )

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_dir, self.labels_frame.iloc[idx, 0] + ".jpeg"
        )
        image = Image.open(img_name)
        label = self.labels_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}


if __name__ == "__main__":
    pass
