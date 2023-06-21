import os

from rich import print

from gate.data.image.task_specific_models.visual_question_answering.ok_vqa import (
    build_ok_vqa_dataset,
)
from gate.data.image.task_specific_models.visual_question_answering.vqa_v2 import (
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
    if idx == 20:
        break


def select_okvqa_items(data_dict):
    """
    Selects the necessary keys from an OK-VQA dataset entry.

    :param data_dict: A dictionary containing an entry from the OK-VQA dataset.
    :type data_dict: dict

    :return: A dictionary containing only the selected keys and their corresponding values.
    :rtype: dict

    .. code-block:: python
        Example item from OK-VQA dataset:
        {
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x427 at 0x7F0BC07E5480>,
        'question_type': 'other',
        'confidence': 2,
        'answers': ['traffic', 'traffic', 'traffic', 'traffic', 'cars', 'cars', 'people', 'people', 'people advancing', 'people advancing'],
        'answers_original': [
            {'answer': 'traffic', 'raw_answer': 'traffic', 'answer_confidence': 'yes', 'answer_id': 1},
            {'answer': 'traffic', 'raw_answer': 'traffic', 'answer_confidence': 'yes', 'answer_id': 2},
            {'answer': 'traffic', 'raw_answer': 'traffic', 'answer_confidence': 'yes', 'answer_id': 3},
            {'answer': 'traffic', 'raw_answer': 'traffic', 'answer_confidence': 'yes', 'answer_id': 4},
            {'answer': 'car', 'raw_answer': 'cars', 'answer_confidence': 'yes', 'answer_id': 5},
            {'answer': 'car', 'raw_answer': 'cars', 'answer_confidence': 'yes', 'answer_id': 6},
            {'answer': 'people', 'raw_answer': 'people', 'answer_confidence': 'yes', 'answer_id': 7},
            {'answer': 'people', 'raw_answer': 'people', 'answer_confidence': 'yes', 'answer_id': 8},
            {'answer': 'people advance', 'raw_answer': 'people advancing', 'answer_confidence': 'yes', 'answer_id': 9},
            {'answer': 'people advance', 'raw_answer': 'people advancing', 'answer_confidence': 'yes', 'answer_id': 10}
        ],
        'id_image': 187611,
        'answer_type': 'other',
        'question_id': 1876115,
        'question': 'What are these traffic lights preventing?',
        'id': 8412,

    """
    return {
        "image_id": data_dict["id_image"],
        "image": data_dict["image"],
        "question_id": data_dict["id_question"],
        "question": data_dict["question"],
        "answers": data_dict["answers"],
    }


def select_vqa_v2_items(data_dict):
    """
    Selects the necessary keys from an OK-VQA dataset entry.

    :param data_dict: A dictionary containing an entry from the OK-VQA dataset.
    :type data_dict: dict

    :return: A dictionary containing only the selected keys and their corresponding values.
    :rtype: dict


    .. code-block:: python
        Example data pooint from VQA v2 dataset:
        {
        'question_type': 'is the',
        'multiple_choice_answer': 'yes',
        'answers': [
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 1},
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 2},
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 3},
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 4},
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 5},
            {'answer': 'yes', 'answer_confidence': 'maybe', 'answer_id': 6},
            {'answer': 'no', 'answer_confidence': 'maybe', 'answer_id': 7},
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 8},
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 9},
            {'answer': 'yes', 'answer_confidence': 'yes', 'answer_id': 10}
        ],
        'image_id': 140655,
        'answer_type': 'yes/no',
        'question_id': 140655004,
        'question': 'Is the food eaten?',
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=352x288 at 0x7FA3785436D0>
        }
    """
    return {
        "image_id": data_dict["id_image"],
        "image": data_dict["image"],
        "question_id": data_dict["id_question"],
        "question": data_dict["question"],
        "answers": data_dict["answers"],
        "answer_type": data_dict["answer_type"],
    }
