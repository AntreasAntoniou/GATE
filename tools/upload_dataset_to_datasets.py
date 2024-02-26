import logging
import os
import pathlib

import datasets
import fire
from datasets import Dataset
from tqdm.auto import tqdm

from gate.data.image.segmentation.cityscapes_pytorch import build_dataset


def convert_dataset_to_list(dataset):
    dataset_as_list = []
    for item in tqdm(dataset):
        dataset_as_list.append(
            {
                "image": item[0],
                "semantic_segmentation": item[1],
            }
        )
    return dataset_as_list


def upload_dataset(
    repo_id: str = "Antreas/Cityscapes",
    dataset_path: str = os.environ.get("DATASET_DIR"),
):
    train_data = build_dataset("train", data_dir=dataset_path)

    train_data_list = convert_dataset_to_list(train_data)

    val_data = build_dataset("val", data_dir=dataset_path)

    val_data_list = convert_dataset_to_list(val_data)

    test_data = build_dataset("test", data_dir=dataset_path)

    test_data_list = convert_dataset_to_list(test_data)

    dataset = datasets.DatasetDict(
        train=Dataset.from_list(train_data_list),
        val=Dataset.from_list(val_data_list),
        test=Dataset.from_list(test_data_list),
    )
    # dataset.save_to_disk(pathlib.Path(dataset_path) / "cityscapes")

    dataset.push_to_hub(
        repo_id=repo_id,
        commit_message="Initial commit",
    )


if __name__ == "__main__":
    fire.Fire(upload_dataset)
