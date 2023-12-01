# pascal_context.py
import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import scipy
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.datasets.utils import download_and_extract_archive

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data.core import GATEDataset
from gate.data.image.segmentation.classes import (
    pascal_context_classes as CLASSES,
)
from gate.data.transforms.segmentation import (
    BaseDatasetTransforms,
    KeySelectorTransforms,
)

logger = logging.getLogger(__name__)


def get_matching_ids(dir1, dir2):
    # Get list of file names (without extensions) in each directory
    print(f"dir1 {dir1}, dir2 {dir2}")
    files_dir1 = [f.stem for f in Path(dir1).glob("*")]
    files_dir2 = [f.stem for f in Path(dir2).glob("*")]

    print(
        f"len files_dir1 {len(files_dir1)}, len files_dir2 {len(files_dir2)}"
    )

    # Find common elements
    matching_ids = list(set(files_dir1) & set(files_dir2))

    return matching_ids


class PascalContextDataset(Dataset):
    def __init__(
        self,
        root_dir: str | pathlib.Path,
        subset: str = "train",
        transform: Optional[List[Callable] | Callable] = None,
    ):
        super().__init__()
        self.root = root_dir
        exists = os.path.isdir(root_dir)

        if not exists:
            os.makedirs(root_dir)
            download_and_extract_archive(
                "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
                download_root=root_dir,
            )
            download_and_extract_archive(
                "https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz",
                download_root=root_dir,
            )

        # Images
        self.image_dir = Path(self.root, "VOCdevkit", "VOC2010", "JPEGImages")

        # Annotations
        self.annotation_dir = Path(self.root, "trainval/")

        self.ids = get_matching_ids(self.image_dir, self.annotation_dir)

        print(f"len ids {len(self.ids)}")

        # Define the classes you care about
        self.selected_classes = CLASSES
        self.selected_class_indices = [
            i for i, class_name in enumerate(self.selected_classes)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.ids[idx]}.jpg")
        img = Image.open(img_name).convert("RGB")

        annotation_name = os.path.join(
            self.annotation_dir, f"{self.ids[idx]}.mat"
        )
        annotation = scipy.io.loadmat(annotation_name)["LabelMap"]

        # Convert to NumPy array to manipulate the labels
        annotation_np = np.array(annotation)

        # Create a new array filled with zeros (background class)
        new_annotation_np = np.zeros_like(annotation_np)

        # Loop through the selected classes and set the corresponding labels
        for new_label, original_label in enumerate(
            self.selected_class_indices
        ):
            new_annotation_np[annotation_np == original_label] = new_label

        # Convert back to PIL Image
        new_annotation = Image.fromarray(
            new_annotation_np.astype("uint8"), "L"
        )

        sample = {"image": img, "labels": new_annotation}

        if self.transform:
            if isinstance(self.transform, list):
                for transform in self.transform:
                    sample["image"] = transform(sample["image"])
                    sample["labels"] = transform(sample["labels"])
            else:
                sample["image"] = self.transform(sample["image"])
                sample["labels"] = self.transform(sample["labels"])

        return sample


from torch.utils.data import random_split


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

    dataset = PascalContextDataset(
        root_dir=data_dir,
    )

    # 💥 Split the dataset into training, validation, and test sets
    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len]
    )

    if set_name == "train":
        return train_dataset
    elif set_name == "val":
        return val_dataset
    else:
        return test_dataset


def label_replacement(annotation):
    annotation = torch.from_numpy(np.array(annotation))
    remapped_labels = annotation.clone()

    remapped_labels[annotation != 255] += 1
    remapped_labels[annotation == 255] = 0

    remapped_labels = T.ToPILImage()(remapped_labels / 255)

    return remapped_labels


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
