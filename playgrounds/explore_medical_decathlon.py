import torch
import os
import datasets
import numpy as np

data = datasets.load_dataset(
    "GATE-engine/medical_decathlon",
    cache_dir=os.environ.get("DATASET_DIR", "/data/tmp/datasets"),
)
