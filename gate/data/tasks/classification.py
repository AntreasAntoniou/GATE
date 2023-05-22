from typing import Any


class ClassificationTask:
    def __init__(
        self, key_remapper_dict: dict = None, tuple_idx_to_key: dict = None
    ):
        super().__init__()
        self.key_remapper_dict = key_remapper_dict
        self.tuple_idx_to_key = tuple_idx_to_key

    def __call__(self, inputs) -> Any:
        output = inputs
        if self.key_remapper_dict is not None:
            output = {}
            for key, value in self.key_remapper_dict.items():
                output[value] = inputs[key]

        if self.tuple_idx_to_key is not None:
            output = {}
            for tuple_idx, value in self.tuple_idx_to_key.items():
                output[value] = inputs[tuple_idx]

        return output
