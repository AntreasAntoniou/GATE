import os
from rich import print
from gate.data.image.visual_question_answering.ok_vqa import (
    build_ok_vqa_dataset,
)
from gate.data.image.visual_question_answering.vqa_v2 import (
    build_vqa_v2_dataset,
)


ok_train_set = build_ok_vqa_dataset(
    "train", data_dir=os.environ.get("PYTEST_DIR")
)

vqa_train_vqa = build_vqa_v2_dataset(
    "train", data_dir=os.environ.get("PYTEST_DIR")
)

for item in ok_train_set:
    print(item)
    break

print(
    "-------------------------------------------------------------------------------------------"
)

for item in vqa_train_vqa:
    print(item)
    break
