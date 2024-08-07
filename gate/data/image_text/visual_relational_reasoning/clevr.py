from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import datasets
import orjson as json
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split

from gate.boilerplate.decorators import configurable
from gate.config.variables import DATASET_DIR
from gate.data import download_kaggle_dataset
from gate.data.core import GATEDataset
from gate.data.image.classification.imagenet1k import StandardAugmentations

FILE_COUNT_AFTER_DOWNLOAD_AND_EXTRACT = 100008


colour_dict = {
    "blue": 0,
    "brown": 1,
    "cyan": 2,
    "gray": 3,
    "green": 4,
    "purple": 5,
    "red": 6,
    "yellow": 7,
}

shape_dict = {
    "cube": 0,
    "cylinder": 1,
    "sphere": 2,
}

count_dict = {
    "0": 0,
    "1": 1,
    "10": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
}

size_dict = {
    "large": 0,
    "small": 1,
}

yes_no_dict = {"no": 0, "yes": 1}

material_dict = {"metal": 0, "rubber": 1}

num_classes: Dict = {
    "colour": len(colour_dict),
    "shape": len(shape_dict),
    "count": len(count_dict),
    "size": len(size_dict),
    "yes_no": len(yes_no_dict),
    "material": len(material_dict),
}


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
        super().__init__()
        self.dataset_path = dataset_path
        dataset_path_dict = self.download_and_extract(dataset_path)
        self.labels_frame = pd.read_csv(
            dataset_path_dict["dataset_download_path"] / "trainLabels.csv"
        )
        self.img_dir = dataset_path_dict["dataset_download_path"] / "resized_train"
        self.transform = transform

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
        self.hf_dataset_dir = (
            dataset_path_dict["dataset_download_path"]
            / "huggingface_dataset"
            / split
        )

        self.transform = transform
        self.split = split

        if not self.hf_dataset_dir.exists():
            # Load the questions
            questions_file = (
                dataset_path_dict["dataset_download_path"]
                / "questions"
                / f"CLEVR_{split}_questions.json"
            )
            if not questions_file.is_file():
                raise FileNotFoundError(f"{questions_file} does not exist.")
            with questions_file.open() as f:
                questions = json.loads(f.read())["questions"]

            self.questions = datasets.Dataset.from_list(questions)
            self.questions.save_to_disk(self.hf_dataset_dir)
        else:
            self.questions = datasets.load_from_disk(self.hf_dataset_dir)

        # Set the image directory
        self.images_dir = (
            dataset_path_dict["dataset_download_path"] / "images" / split
        )

    def create_answer_mapping(self) -> dict:
        """
        Create a mapping from answers to indices.

        Returns:
            dict: A dictionary mapping answers to indices.
        """
        # Collect all unique answers
        answers = set(question["answer"] for question in self.questions)

        # Map each answer to a unique index
        return {answer: idx for idx, answer in enumerate(sorted(answers))}

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
        answer = self.questions[idx]["answer"]
        if answer in yes_no_dict.keys():
            labels = torch.tensor(yes_no_dict[answer])
            answer_type = "yes_no"
        elif answer in colour_dict.keys():
            labels = torch.tensor(colour_dict[answer])
            answer_type = "colour"
        elif answer in shape_dict.keys():
            labels = torch.tensor(shape_dict[answer])
            answer_type = "shape"
        elif answer in count_dict.keys():
            labels = torch.tensor(count_dict[answer])
            answer_type = "count"
        elif answer in size_dict.keys():
            labels = torch.tensor(size_dict[answer])
            answer_type = "size"
        elif answer in material_dict.keys():
            labels = torch.tensor(material_dict[answer])
            answer_type = "material"

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "text": question,
            "question_idx": question_idx,
            "question_family_idx": self.questions[idx][
                "question_family_index"
            ],
            "image_idx": image_idx,
            "split": split,
            "image_filename": image_filename,
            "answer": answer,
            "answer_type": answer_type,
            "labels": labels,
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

    if set_name not in ["train", "val", "test"]:
        raise KeyError(f"Invalid set name: {set_name}")

    train_val_set = CLEVRClassificationDataset(
        root_dir=data_dir, split="train"
    )
    dataset_length = len(train_val_set)
    val_split = 0.1  # Fraction for the validation set (e.g., 10%)

    # Calculate the number of samples for train and validation sets
    val_length = int(dataset_length * val_split)
    train_length = dataset_length - val_length

    train_set, val_set = random_split(
        dataset=train_val_set, lengths=[train_length, val_length]
    )

    test_set = CLEVRClassificationDataset(root_dir=data_dir, split="val")

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]


def transform_wrapper(inputs: Dict, target_size=224):
    return {
        "image": T.Resize(size=(target_size, target_size), antialias=True)(
            inputs["image"]
        ),
        "text": inputs["text"],
        "labels": torch.tensor(int(inputs["labels"])).long(),
        "answer_type": inputs["answer_type"],
        "question_family_idx": inputs["question_family_idx"],
    }


@configurable(
    group="dataset", name="clevr", defaults=dict(data_dir=DATASET_DIR)
)
def build_gate_dataset(
    data_dir: Optional[str] = None,
    transforms: Optional[Any] = None,
    num_classes: Dict = num_classes,
):
    train_set = GATEDataset(
        dataset=build_dataset("train", data_dir=data_dir),
        infinite_sampling=True,
        transforms=[
            transform_wrapper,
            StandardAugmentations(image_key="image"),
            transforms,
        ],
    )

    val_set = GATEDataset(
        dataset=build_dataset("val", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    test_set = GATEDataset(
        dataset=build_dataset("test", data_dir=data_dir),
        infinite_sampling=False,
        transforms=[transform_wrapper, transforms],
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}
    return dataset_dict
