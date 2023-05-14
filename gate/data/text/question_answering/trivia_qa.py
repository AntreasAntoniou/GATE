# trivia_qa.py
from typing import Optional

import numpy as np
from datasets import load_dataset


def build_trivia_qa_dataset(set_name: str, data_dir: Optional[str] = None) -> dict:
    """
    Build a trivia_qa dataset using the Hugging Face datasets library.
    https://huggingface.co/datasets/trivia_qa

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    rng = np.random.RandomState(42)

    train_set = load_dataset(
        'trivia_qa', 'rc',
        split="train",
        cache_dir=data_dir,
    )

    val_set = load_dataset(
        'trivia_qa', 'rc',
        split="validation",
        cache_dir=data_dir,
    )

    test_set = load_dataset(
        'trivia_qa', 'rc',
        split="test",
        cache_dir=data_dir,
    )

    dataset_dict = {"train": train_set, "val": val_set, "test": test_set}

    return dataset_dict[set_name]
