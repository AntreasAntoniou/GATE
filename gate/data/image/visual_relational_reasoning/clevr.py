import os
import pathlib
from typing import Callable, Optional

import pandas as pd
import torch.utils.data as data
from PIL import Image

from gate.data import download_kaggle_dataset

FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 70236


class CLEVRClassification(data.Dataset):
    def __init__(
        self, dataset_path: pathlib.Path, transform: Optional[Callable] = None
    ):
        super().__init__()
        self.dataset_path = dataset_path
        dataset_path_dict = self.download_and_extract(dataset_path)
        self.labels_frame = pd.read_csv(
            dataset_path_dict["dataset_download_path"] / "trainLabels.csv"
        )
<<<<<<< HEAD
        self.img_dir = dataset_path_dict["dataset_download_path"] / "resized_train"
=======
        self.img_dir = (
            dataset_path_dict["dataset_download_path"] / "resized_train"
        )
>>>>>>> 054bb2ecac9c7df61b38b0fdc337c174fbbd1fdd
        self.transform = transform

    def download_and_extract(self, dataset_path: pathlib.Path):
        return download_kaggle_dataset(
            dataset_name="clevr",
            dataset_path="timoboz/clevr-dataset",
            target_dir_path=dataset_path,
            file_count_after_download_and_extract=FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT,
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
    from rich import print as rprint
    from rich.traceback import install

    install()
    dataset = (
        CLEVRClassification(
            dataset_path=pathlib.Path(os.environ["PYTEST_DIR"]), transform=None
        ),
    )

    # print(len(dataset))
    # print(dataset[0]["image"].shape)
    # print(dataset[0]["label"])

    # plt.imshow(dataset[0]["image"].permute(1, 2, 0))
    # plt.show()
