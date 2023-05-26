from typing import Any


class YahooAnswersTask:
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
        'id': 209438,

        'topic: 9,

        'question_title': 'After a child is conceived, should the father have a say in whether he should be forced to pay child support..',

        'question_content': '. or not...even if he legally doesn't want to have any emotional responsibility for the child...',

        'best_answer': 'The father must be made to take care of his child, unless he is mentally retarded and therefore can't work or attain a fixed income to support his child.'
        }

        """
        return {
            "text": {
                "question_content": inputs["question_content"],
                "best_answer": inputs["best_answer"],
            },
            "labels": {
                "labels": inputs["topic"],
                "labels_names": inputs["question_title"],
            },
        }
