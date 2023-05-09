import os
from rich import print
from gate.data.image.visual_question_answering.ok_vqa import (
    build_ok_vqa_dataset,
)


train_set = build_ok_vqa_dataset(
    "train", data_dir=os.environ.get("PYTEST_DIR")
)

for item in train_set:
    print(item)
    break
