from typing import Any


class VQAV2Task:
    def __init__(self):
        super().__init__()

    def __call__(self, inputs) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            Any: _description_

        .. code-block:: python
        An input looks like this before processing
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
            "image": inputs["image"],
            "text": {
                "question": inputs["question"],
                "answers": [entry["answer"] for entry in inputs["answers"]],
            },
        }


class OKVQATask:
    def __init__(self):
        super().__init__()

    def __call__(self, inputs) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            Any: _description_

        .. code-block:: python
        An input looks like this before processing
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
        }
        """
        return {
            "image": inputs["image"],
            "text": {
                "question": inputs["question"],
                "answers": inputs["answers"],
            },
        }
