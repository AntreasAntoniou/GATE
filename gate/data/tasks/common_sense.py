from typing import Any


class PIQATask:
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
        'goal': "When boiling butter, when it's ready, you can",
        'sol1': 'Pour it onto a plate',
        'sol2': 'Pour it into a jar',
        'label': 1
        }
        """
        return {
            "text": {
                "prompt": inputs["goal"],
                "choices": [inputs["sol1"], inputs["sol2"]],
            },
            "labels": inputs["label"],
        }


class WinograndeTask:
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
        'sentence': 'John moved the couch from the garage to the backyard to create space. The _ is small.',
        'option1': 'garage',
        'option2': 'backyard',
        'answer': '1'
        }
        """
        return {
            "text": {
                "prompt": inputs["sentence"],
                "choices": [inputs["option1"], inputs["option2"]],
            },
            "labels": inputs["answer"],
        }


class HellaSwagTask:
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
        'ind': 4,
        'activity_label': 'Removing ice from car',
        'ctx_a': 'Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.',
        'ctx_b': 'then',
        'ctx': 'Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then',
        'endings': [
            ', the man adds wax to the windshield and cuts it.',
            ', a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.',
            ', the man puts on a christmas coat, knitted with netting.',
            ', the man continues removing the snow on his car.'
            ],
        'source_id': 'activitynet~v_-1IBHYS3L-Y',
        'split': 'train',
        'split_type': 'indomain',
        'label': '3'
        }
        """
        return {
            "text": {"prompt": inputs["ctx"], "choices": inputs["endings"]},
            "labels": inputs["label"],
        }
