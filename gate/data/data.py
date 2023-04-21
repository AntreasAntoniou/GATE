from typing import List, Optional

from datasets import load_dataset

from gate.boilerplate.decorators import configurable


@configurable
def build_dataset(
    dataset_name: str,
    data_dir: str,
    sets_to_include: Optional[List[str]] = None,
) -> dict:
    """
    🏗️ Build a dataset using the Hugging Face datasets library.

    :param dataset_name: The name of the dataset to load.
    :param data_dir: The directory where the dataset cache is stored.
    :param sets_to_include: A list of dataset splits to include
    (default: ["train", "validation"]).
    :return: A dictionary containing the dataset splits.
    """
    if sets_to_include is None:
        sets_to_include = ["train", "validation"]

    dataset = {}
    for set_name in sets_to_include:
        data = load_dataset(
            path=dataset_name,
            split=set_name,
            cache_dir=data_dir,
            task="image-classification",
        )
        dataset[set_name] = data

    return dataset
