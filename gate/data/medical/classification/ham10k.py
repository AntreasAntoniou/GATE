import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset, Subset

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import count_files_recursive, get_logger
from gate.config.variables import DATASET_DIR
from gate.data import download_kaggle_dataset
from gate.data.core import GATEDataset
from gate.data.tasks.classification import ClassificationTask

logger = get_logger(name=__name__, set_rich=True)

FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 20035


class HAM10KClassification(Dataset):
    def __init__(
        self, dataset_path: pathlib.Path, transform: Optional[Callable] = None
    ):
        """https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

        Args:
            dataset_path (pathlib.Path): _description_
            transform (Optional[Callable], optional): _description_. Defaults to None.
        """
        super().__init__()
        dataset_path_dict = self.download_and_extract(dataset_path)

        dataset_path = dataset_path_dict["dataset_download_path"]
        self.dataset_path = dataset_path
        self.meta_data_path = dataset_path / "HAM10000_metadata.csv"
        self.image_paths = [
            dataset_path / "HAM10000_images_part_1",
            dataset_path / "HAM10000_images_part_2",
        ]
        self.image_name_to_path = self.collect_image_paths()
        self.meta_data = pd.read_csv(self.meta_data_path)

        # get labels and map to ints
        self.label_names = self.meta_data["dx"].unique()
        self.label_names_to_int = {
            label: i for i, label in enumerate(self.label_names)
        }
        self.image_ids = self.meta_data["image_id"]
        self.labels = self.meta_data["dx"].map(self.label_names_to_int)

        self.transform = transform

    def collect_image_paths(self):
        paths = self.image_paths

        image_path_dict = {}

        for path in paths:
            for image_path in path.glob("*.jpg"):
                image_name = pathlib.Path(
                    image_path
                ).stem  # remove the '.jpg' extension
                image_path_dict[image_name] = image_path

        return image_path_dict

    def download_and_extract(self, dataset_path: pathlib.Path):
        return download_kaggle_dataset(
            dataset_name="ham10k",
            dataset_path="kmader/skin-cancer-mnist-ham10000",
            target_dir_path=dataset_path,
            file_count_after_download_and_extract=FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT,
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.image_ids[idx]
        label = self.labels[idx]
        image_path = self.image_name_to_path[image_name]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "labels": label}


def build_dataset(
    train_ratio: float = 0.8,
    val_ratio: float = 0.05,
    data_dir: Optional[str] = None,
) -> dict:
    """
    Build a HAM10K dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    torch.manual_seed(42)

    logger.info(
        f"Loading Diabetic retinopathy dataset, will download to {data_dir} if necessary."
    )

    dataset = HAM10KClassification(dataset_path=data_dir)

    train_length = int(len(dataset) * train_ratio)
    val_length = int(len(dataset) * val_ratio)

    train_set = Subset(dataset, list(range(train_length)))
    val_set = Subset(
        dataset, list(range(train_length, train_length + val_length))
    )
    test_set = Subset(
        dataset, list(range(train_length + val_length, len(dataset)))
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict


def dataset_format_transform(sample: Dict) -> Dict:
    # Example of sample:
    #

    input_dict = {}
    input_dict["image"] = sample["image"]
    input_dict["labels"] = sample["labels"]
    return input_dict


@configurable(
    group="dataset",
    name="ham10k",
    defaults=dict(data_dir=DATASET_DIR),
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=4,
) -> dict:
    dataset_dict = build_dataset(data_dir=data_dir)
    train_set = GATEDataset(
        dataset=dataset_dict["train"],
        infinite_sampling=True,
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=dataset_dict["val"],
        infinite_sampling=False,
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=dataset_dict["test"],
        infinite_sampling=False,
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_dataset(transforms: Optional[Any] = None) -> dict:
    # Create a dummy dataset that emulates food-101's shape and modality
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_classes: int = 4


# Details on classes https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6091241/

# akiec
# Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma
# (Bowen’s disease) are common non-invasive, variants of squamous
# cell carcinoma that can be treated locally without surgery.
# The dermatoscopic criteria of pigmented actinic keratoses and Bowen’s disease.

# bcc
# Basal cell carcinoma is a common variant of epithelial skin cancer that rarely
# metastasizes but grows destructively if untreated.

# bkl
# "Benign keratosis" is a generic class that includes seborrheic keratoses ("senile wart"),
# solar lentigo - which can be regarded a flat variant of seborrheic keratosis - and lichen-planus
# like keratoses (LPLK), which corresponds to a seborrheic keratosis or a solar lentigo with
# inflammation and regression25. The three subgroups may look different dermatoscopically,
# but we grouped them together because they are similar biologically and often reported under
# the same generic term histopathologically.


# df
# Dermatofibroma is a benign skin lesion regarded as either a benign proliferation or an
# inflammatory reaction to minimal trauma.

# nv
# Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants,
# which all are included in our series. The variants may differ significantly from a
# dermatoscopic point of view.

# mel
# Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants.
# If excised in an early stage it can be cured by simple surgical excision.

# vasc
# Vascular skin lesions in the dataset range from cherry angiomas to angiokeratomas31 and pyogenic granulomas32. Hemorrhage is also included in this category.
# Angiomas are dermatoscopically characterized by red or purple color and solid, well circumscribed structures known as red clods or lacunes.
# The number of images in the datasets does not correspond to the number of unique lesions, because we also provide images of the same lesion taken at different magnifications or angles (Fig. 4), or with different cameras. This should serve as a natural data-augmentation as it shows random transformations and visualizes both general and local features.
