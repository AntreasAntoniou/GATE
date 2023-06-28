import torch
import os
import datasets
import numpy as np
from rich import print

data = datasets.load_dataset(
    "GATE-engine/medical_decathlon",
    cache_dir=os.environ.get("DATASET_DIR", "/data/tmp/datasets"),
)

keys = list(data.keys())
print(keys)

# ['image', 'label', 'image_meta_dict', 'label_meta_dict', 'task_name']


for item in data["training.task01braintumour"]:
    image = item["image"]
    label = item["label"]
    image_meta_dict = item["image_meta_dict"]
    label_meta_dict = item["label_meta_dict"]
    task_name = item["task_name"]
    print(
        f"task_name: {task_name}, image.shape: {image.shape}, label.shape: {label.shape}, image_meta_dict: {image_meta_dict}, label_meta_dict: {label_meta_dict}"
    )
    break
