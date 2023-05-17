# cityscapes.py
from typing import Optional

import torch

from datasets import load_dataset


def build_nyu_depth_v2_dataset(
    set_name: str, data_dir: Optional[str] = None
) -> dict:
    """
    Build a Food-101 dataset using the Hugging Face datasets library.

    Args:
        data_dir: The directory where the dataset cache is stored.
        set_name: The name of the dataset split to return
        ("train", "val", or "test").

    Returns:
        A dictionary containing the dataset split.
    """
    # Create a generator with the specified seed
    rng = torch.Generator().manual_seed(42)

    train_val_data = load_dataset(
        path="sayakpaul/nyu_depth_v2",
        split="train",
        cache_dir=data_dir,
    )

    test_data = load_dataset(
        path="sayakpaul/nyu_depth_v2",
        split="validation",
        cache_dir=data_dir,
    )

    train_val_data = train_val_data.train_test_split(test_size=0.1)
    train_data = train_val_data["train"]
    val_data = train_val_data["test"]

    dataset_dict = {"train": train_data, "val": val_data, "test": test_data}

    return dataset_dict[set_name]


# from datasets import load_dataset
# import numpy as np
# import matplotlib.pyplot as plt


# cmap = plt.cm.viridis

# ds = load_dataset("sayakpaul/nyu_depth_v2")


# def colored_depthmap(depth, d_min=None, d_max=None):
#     if d_min is None:
#         d_min = np.min(depth)
#     if d_max is None:
#         d_max = np.max(depth)
#     depth_relative = (depth - d_min) / (d_max - d_min)
#     return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


# def merge_into_row(input, depth_target):
#     input = np.array(input)
#     depth_target = np.squeeze(np.array(depth_target))

#     d_min = np.min(depth_target)
#     d_max = np.max(depth_target)
#     depth_target_col = colored_depthmap(depth_target, d_min, d_max)
#     img_merge = np.hstack([input, depth_target_col])

#     return img_merge


# random_indices = np.random.choice(len(ds["train"]), 9).tolist()
# train_set = ds["train"]

# plt.figure(figsize=(15, 6))

# for i, idx in enumerate(random_indices):
#     ax = plt.subplot(3, 3, i + 1)
#     image_viz = merge_into_row(
#         train_set[idx]["image"], train_set[idx]["depth_map"]
#     )
#     plt.imshow(image_viz.astype("uint8"))
#     plt.axis("off")
