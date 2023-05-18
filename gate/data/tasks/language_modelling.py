from typing import Any


class LanguageModellingTask:
    def __init__(self):
        super().__init__()
        self.labels = {

        }

    def __call__(self, inputs) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            Any: _description_

        .. code-block:: python
        An input looks like this before processing
        {
            "text": "While athletes in different professions dealt with doping scandals and other controversies , Woods 
            continued to do what he did best : dominate the field of professional golf and rake in endorsements ."
        }
        """
        return inputs