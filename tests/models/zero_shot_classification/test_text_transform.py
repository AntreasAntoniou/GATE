
import torch
from transformers import CLIPProcessor

from gate.models.backbones.clip_image import TextProcessor

clip_model_name = "openai/clip-vit-base-patch16"


def test_apply_transform():
    preprocessor = CLIPProcessor.from_pretrained(clip_model_name)
    p = TextProcessor(preprocessor)

    text_input_1 = ["I am John", "I am Johnny"]
    text_input_2 = [
        ["I am John", "I am Johnny"],
        ["I am Andreas", "I am Andreas"],
    ]

    result_1 = p.apply_transform(text_input_1)
    result_2 = p.apply_transform(text_input_2)

    assert isinstance(
        result_1, torch.Tensor
    ), "Expected Tensor, got: {type(result_1).__name__}"
    assert isinstance(
        result_2, torch.Tensor
    ), "Expected Tensor, got: {type(result_2).__name__}"
