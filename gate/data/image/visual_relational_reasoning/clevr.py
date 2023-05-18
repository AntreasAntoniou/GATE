from dataclasses import dataclass
import os
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import orjson as json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from rich import print
from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data import download_kaggle_dataset
from gate.data.core import GATEDataset

FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 100008


class CLEVRClassificationDataset(Dataset):
    """
    A PyTorch Dataset for the CLEVR dataset.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        split: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            root_dir (Union[str, Path]): Root directory of the dataset.
            transform (Optional[Callable[[Image.Image], torch.Tensor]], optional): Optional transformation to apply on the images. Defaults to None.
            split (str, optional): Split of the dataset to load. One of "train", "val", or "test". Defaults to 'train'.
        """
        super().__init__()
        self.dataset_path = Path(root_dir)
        dataset_path_dict = self.download_and_extract(self.dataset_path)
        dataset_path_dict["dataset_download_path"] = (
            dataset_path_dict["dataset_download_path"] / "CLEVR_v1.0"
        )

        self.transform = transform
        self.split = split

        # Load the questions
        questions_file = (
            dataset_path_dict["dataset_download_path"]
            / "questions"
            / f"CLEVR_{split}_questions.json"
        )
        if not questions_file.is_file():
            raise FileNotFoundError(f"{questions_file} does not exist.")
        with questions_file.open() as f:
            self.questions = json.loads(f.read())["questions"]

        # for idx, question in enumerate(self.questions):
        #     print(question)
        #     if idx == 10:
        #         break

        # Set the image directory
        self.images_dir = (
            dataset_path_dict["dataset_download_path"] / "images" / split
        )

    def download_and_extract(self, dataset_path: Path) -> dict:
        """
        Download and extract the dataset.

        Args:
            dataset_path (Path): Path to download and extract the dataset.

        Returns:
            dict: Dictionary containing the paths of the dataset.
        """
        return download_kaggle_dataset(
            dataset_name="clevr",
            dataset_path="timoboz/clevr-dataset",
            target_dir_path=dataset_path,
            file_count_after_download_and_extract=FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT,
        )

    def __len__(self) -> int:
        """
        Determine the length of the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.questions)

    def __getitem__(
        self, idx: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, str]:
        """
        Fetch an item from the dataset.

        Args:
            idx (Union[int, torch.Tensor]): Index of the item.

        Returns:
            Tuple[torch.Tensor, str]: A tuple containing the image and the corresponding question.
        """

        # "image_index": 0,
        # "split": "test",
        # "image_filename": "CLEVR_test_000000.png",
        # "question_index": 0,
        # "question": "Is there anything else that is the same shape as the small brown matte object?"

        # Load image
        img_name = self.images_dir / self.questions[idx]["image_filename"]
        if not img_name.is_file():
            raise FileNotFoundError(f"{img_name} does not exist.")

        image = Image.open(img_name).convert("RGB")
        question = self.questions[idx]["question"]
        question_idx = self.questions[idx]["question_index"]
        image_idx = self.questions[idx]["image_index"]
        split = self.questions[idx]["split"]
        image_filename = self.questions[idx]["image_filename"]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "text": question,
            "question_idx": question_idx,
            "image_idx": image_idx,
            "split": split,
            "image_filename": image_filename,
        }


def build_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a OK VQA dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    torch.manual_seed(42)

    if set_name not in ["train", "val", "test"]:
        raise KeyError(f"Invalid set name: {set_name}")

    train_set = CLEVRClassificationDataset(root_dir=data_dir, split="train")

    val_set = CLEVRClassificationDataset(root_dir=data_dir, split="val")

    test_set = CLEVRClassificationDataset(root_dir=data_dir, split="test")

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


@configurable(
    group="dataset", name="clevr", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
):
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=transforms,
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=transforms,
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=transforms,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict


def build_dummy_dataset(transforms: Optional[Any] = None):
    pass


@dataclass
class DefaultHyperparameters:
    train_batch_size: int = 256
    eval_batch_size: int = 512
