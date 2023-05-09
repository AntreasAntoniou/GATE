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

# for item in ok_train_set:
#     print(item)
#     break

# print(
#     "-------------------------------------------------------------------------------------------"
# )

for idx, item in enumerate(vqa_train_vqa):
    print(item)
    if idx == 5:
        break


def select_okvqa_items(data_dict):
    """
    Selects the necessary keys from an OK-VQA dataset entry.

    :param data_dict: A dictionary containing an entry from the OK-VQA dataset.
    :type data_dict: dict

    :return: A dictionary containing only the selected keys and their corresponding values.
    :rtype: dict
    """
    return {
        "image_id": data_dict["id_image"],
        "image": data_dict["image"],
        "question_id": data_dict["id_question"],
        "question": data_dict["question"],
        "answers": data_dict["answers"],
    }
