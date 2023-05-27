import os
import time
from datasets import load_dataset
import numpy as np
import multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gate.data.image.classification.food101 import build_gate_dataset
from gate.models.task_specific_models.classification.clip import (
    build_gate_clip_model,
)

clip_classifier = build_gate_clip_model()
# Build the dataset
dataset_dict = build_gate_dataset(
    data_dir=os.environ["DATASET_DIR"], transforms=clip_classifier.transform
)

train_val_data = load_dataset(
    path="food101",
    split="train",
    cache_dir=os.environ["DATASET_DIR"],
    task="image-classification",
    num_proc=mp.cpu_count(),
)
train_val_data = train_val_data.set_transform(clip_classifier.transform)
# Create a DataLoader with batch size 1 to load one sample at a time
data_loader = DataLoader(
    train_val_data,
    batch_size=256,
    shuffle=True,
    num_workers=mp.cpu_count(),
    pin_memory=True,
    persistent_workers=True,
)

# Array to store loading times
loading_times = []

# Measure the loading speed for 100 data points
start_time = time.time()
for i, data in tqdm(enumerate(data_loader)):
    if i >= 100:  # We only measure the first 100 data points
        break
    end_time = time.time()
    start_time = time.time()

    loading_times.append(end_time - start_time)

# Convert to numpy array for easier manipulation
loading_times = np.array(loading_times)

# Calculate mean and standard deviation
mean_time = np.mean(loading_times)
std_time = np.std(loading_times)

print(f"Mean loading time per sample: {mean_time} seconds")
print(f"Standard deviation of loading time per sample: {std_time} seconds")
