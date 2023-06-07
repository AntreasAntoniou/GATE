from gate.data.image_text.zero_shot.flickr30k import build_dataset
from gate.models.task_specific_models.zero_shot_classification.clip import (
    build_gate_model,
)

dataset = build_dataset("train", data_dir="/data1/")
model = build_gate_model()
