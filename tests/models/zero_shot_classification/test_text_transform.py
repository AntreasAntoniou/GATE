from typing import List, Union

import pytest
import torch
from transformers import CLIPProcessor

clip_model_name = "openai/clip-vit-base-patch16"


class TextProcessor:
    def __init__(self):
        self.preprocessor = CLIPProcessor.from_pretrained(clip_model_name)

    def text_transforms(self, x: Union[List[str], List[List[str]]]):
        if isinstance(x[0], list):
            x = [item for sublist in x for item in sublist]
        return self.preprocessor(
            text=x, return_tensors="pt", padding=True, truncation=True
        ).input_ids.squeeze(0)

    def apply_transform(
        self, text: Union[List[str], List[List[str]]], transform_dict: dict
    ):
        if not all(
            isinstance(i, list) for i in text
        ):  # if text is list of strings
            text = [text]
        transformed_text = transform_dict["text"](text)
        return transformed_text


def test_apply_transform():
    p = TextProcessor()
    transform_dict = {"text": p.text_transforms}

    text_input_1 = ["I am John", "I am Johnny"]
    text_input_2 = [
        ["I am John", "I am Johnny"],
        ["I am Andreas", "I am Antreas"],
    ]

    result_1 = p.apply_transform(text_input_1, transform_dict)
    result_2 = p.apply_transform(text_input_2, transform_dict)

    assert isinstance(
        result_1, torch.Tensor
    ), "Expected Tensor, got: {type(result_1).__name__}"
    assert isinstance(
        result_2, torch.Tensor
    ), "Expected Tensor, got: {type(result_2).__name__}"
