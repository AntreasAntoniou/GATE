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

for item in data["training.task01braintumour"]:
    print(item)
    break
