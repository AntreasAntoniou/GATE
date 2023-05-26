import os
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gate.data.image.classification.imagenet1k import build_gate_dataset
from gate.models.task_specific_models.classification.clip import (
    build_gate_clip_model,
)

clip_classifier = build_gate_clip_model()
# Build the dataset
dataset_dict = build_gate_dataset(
    data_dir=os.environ["DATASET_DIR"], transforms=clip_classifier.transform
)

# Create a DataLoader with batch size 1 to load one sample at a time
data_loader = DataLoader(
    dataset_dict["train"], batch_size=1, shuffle=True, num_workers=4
)

# Array to store loading times
loading_times = []

# Measure the loading speed for 100 data points
for i, data in tqdm(enumerate(data_loader)):
    if i >= 100:  # We only measure the first 100 data points
        break

    start_time = time.time()

    # Here, data is a single data point from the dataset
    # You can replace the pass statement with your processing code
    pass

    end_time = time.time()

    loading_times.append(end_time - start_time)

# Convert to numpy array for easier manipulation
loading_times = np.array(loading_times)

# Calculate mean and standard deviation
mean_time = np.mean(loading_times)
std_time = np.std(loading_times)

print(f"Mean loading time per sample: {mean_time} seconds")
print(f"Standard deviation of loading time per sample: {std_time} seconds")
