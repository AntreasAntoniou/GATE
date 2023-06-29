import os

import datasets
import numpy as np
import torch
import yaml
from rich import print

data = datasets.load_dataset(
    "GATE-engine/medical_decathlon",
    cache_dir=os.environ.get("DATASET_DIR", "/data/tmp/datasets"),
)

task_list = list(data.keys())
print(task_list)

# ['image', 'label', 'image_meta_dict', 'label_meta_dict', 'task_name']


for task in task_list:
    for item in data[task]:
        image = (
            torch.stack([torch.tensor(i) for i in item["image"]])
            if isinstance(item["image"], list)
            else item["image"]
        )
        label = (
            torch.stack([torch.tensor(i) for i in item["label"]])
            if isinstance(item["label"], list)
            else item["label"]
        )
        image_meta_dict = yaml.dump(item["image_meta_dict"])
        label_meta_dict = yaml.dump(item["label_meta_dict"])
        task_name = item["task_name"]
        print(
            f"task_name: {task_name}, image.shape: {image.shape}, label.shape: {label.shape}, image_meta_dict: {image_meta_dict}, label_meta_dict: {label_meta_dict}"
        )
        break
