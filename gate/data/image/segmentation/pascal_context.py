# pascal_context.py
import os
import pathlib
import tarfile
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import (
    pascal_context_classes as CLASSES,
)
from gate.data.transforms.segmentation_transforms import (
    BaseDatasetTransforms,
    DualImageRandomCrop,
    KeySelectorTransforms,
)

logger = get_logger(name=__name__)


class PascalContextDataset(Dataset):
    def __init__(
        self,
        root_dir: str | pathlib.Path,
        subset: str = "train",
        transform: Optional[List[Callable] | Callable] = None,
    ):
        """
        Initializes a PascalContextDataset object.

        Args:
            root_dir (str): Root directory of the dataset.
            subset (str, optional): Dataset subset to use. Defaults to "train".
            transform (callable, optional): Optional transform to be applied on a sample.

        Returns:
            None
        """
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.download_dataset()
        self.image_dir = os.path.join(
            self.root_dir, "VOCdevkit/VOC2010/JPEGImages"
        )
        self.annotation_dir = os.path.join(
            self.root_dir, "VOCdevkit/VOC2010/SegmentationClass"
        )
        self.split_file = os.path.join(
            self.root_dir,
            f"VOCdevkit/VOC2010/ImageSets/Segmentation/{self.subset}.txt",
        )

        with open(self.split_file, "r") as f:
            self.ids = f.read().splitlines()

    def download_dataset(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        dataset_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
        tar_path = os.path.join(self.root_dir, "VOCtrainval_03-May-2010.tar")

        if not os.path.exists(tar_path):
            logger.info("Downloading dataset...")

            response = requests.get(dataset_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

            with open(tar_path, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                logger.error("ERROR, something went wrong")

            print("Extracting dataset...")
            with tarfile.open(tar_path, "r") as tar_ref:
                tar_ref.extractall(path=self.root_dir)

        logger.info("Dataset is ready.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.ids[idx]}.jpg")
        img = Image.open(img_name).convert("RGB")

        annotation_name = os.path.join(
            self.annotation_dir, f"{self.ids[idx]}.png"
        )
        annotation = Image.open(annotation_name).convert("L")

        sample = {"image": img, "labels": annotation}

        if self.transform:
            if isinstance(self.transform, list):
                for transform in self.transform:
                    sample["image"] = transform(sample["image"])
                    sample["labels"] = transform(sample["labels"])
            else:
                sample["image"] = self.transform(sample["image"])
                sample["labels"] = self.transform(sample["labels"])

        return sample


def build_dataset(
    set_name: str, data_dir: Optional[str] = None, download: bool = False
):
    """
    Build the Pascal Context dataset.

    Args:
        set_name (str): The name of the dataset split to return
        ("train", "val" or "test").
        data_dir (Optional[str]): The directory where the dataset is stored.
        Default: None.
        download (bool): Whether to download the dataset if
        not already present. Default: False.

    Returns:
        A Dataset object containing the specified dataset split.
    """
    if data_dir is None:
        data_dir = "data/pascal_context"

    if set_name not in ["train", "val", "test"]:
        raise KeyError("❌ Invalid set_name, choose 'train', 'val' or 'test'")

    # 🛠️ Create the Pascal Context dataset using the torchvision
    # VOCSegmentation class
    data_dir = pathlib.Path(data_dir) / "pascal_context"

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    train_dataset = PascalContextDataset(
        root_dir=data_dir,
        subset="train",
    )
    test_dataset = PascalContextDataset(
        root_dir=data_dir,
        subset="val",
    )
    # 💥 Split the train set into training and validation sets
    train_len = int(0.9 * len(train_dataset))
    val_len = len(train_dataset) - train_len

    train_dataset, val_dataset = random_split(
        train_dataset, [train_len, val_len]
    )

    if set_name == "train":
        return train_dataset
    elif set_name == "val":
        return val_dataset
    else:
        return test_dataset


def label_replacement(annotation):
    annotation = torch.from_numpy(np.array(annotation))
    annotation[annotation != 255] += 1
    annotation[annotation == 255] = 0

    annotation = T.ToPILImage()(annotation / 255)

    return annotation


@configurable(
    group="dataset", name="pascal_context", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes=len(CLASSES),
    image_size=1024,
    target_image_size=256,
) -> dict:
    input_transforms = KeySelectorTransforms(
        initial_size=2048,
        image_label="image",
        label_label="labels",
        # label_transforms=[label_replacement],
    )

    train_transforms = BaseDatasetTransforms(
        input_size=image_size,
        target_size=target_image_size,
        crop_size=image_size,
        flip_probability=0.5,
        use_photo_metric_distortion=True,
    )

    eval_transforms = BaseDatasetTransforms(
        input_size=image_size,
        target_size=target_image_size,
        crop_size=None,
        flip_probability=None,
        use_photo_metric_distortion=False,
    )

    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            input_transforms,
            train_transforms,
            transforms,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            input_transforms,
            eval_transforms,
            transforms,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[
            input_transforms,
            eval_transforms,
            transforms,
        ],
        meta_data={"class_names": CLASSES, "num_classes": num_classes},
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
