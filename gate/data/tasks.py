from typing import Any

from gate.boilerplate.decorators import configurable


class ClassificationTask:
    def __call__(self, inputs) -> Any:
        return inputs
