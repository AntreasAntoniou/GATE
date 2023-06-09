from urllib.request import urlopen

import PIL.Image as Image
import torch
from rich.traceback import install
from tests.models.test_clip_vqa_model import pad_tokens

from gate.models.task_specific_models.visual_question_answering.timm import (
    ModelAndTransform,
    build_model,
)

install()


def test_build_model():
    model_and_transform = build_model()

    assert isinstance(model_and_transform, ModelAndTransform)
    assert model_and_transform.model is not None
    assert model_and_transform.transform is not None


def test_model_forward():
    model_and_transform = build_model()
    model = model_and_transform.model
    transforms_dict = model.get_transforms()

    img = Image.open(
        urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
    images = [img, img, img]
    questions = [
        "What is in the image?",
        "Who is in the image?",
        "What is the weather like?",
    ]
    answers = ["beignets", "a cat", "sunny"]

    encoder_images = torch.stack(
        [transforms_dict["image_encoder"](image) for image in images]
    )

    encoder_questions = pad_tokens(
        [transforms_dict["text_encoder"](question) for question in questions]
    )

    decoder_questions = pad_tokens(
        [transforms_dict["text_decoder"](question) for question in questions]
    )

    decoder_answers = pad_tokens(
        [transforms_dict["text_decoder"](answer) for answer in answers]
    )

    input_dict = {
        "image_encoder_tokens": encoder_images,
        "question_encoder_tokens": encoder_questions,
        "question_decoder_tokens": decoder_questions,
        "answer_decoder_tokens": decoder_answers,
    }

    output = model.forward(input_dict)

    assert output["logits"].shape == (3, 8, 50257)

    assert output["loss"] > 0


def test_model_forward_loss():
    model_and_transform = build_model()
    model = model_and_transform.model
    transforms_dict = model.get_transforms()

    img = Image.open(
        urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
    images = [img, img, img]
    questions = [
        "What is in the image?",
        "Who is in the image?",
        "What is the weather like?",
    ]
    answers = ["beignets", "a cat", "sunny"]

    encoder_images = torch.stack(
        [transforms_dict["image_encoder"](image) for image in images]
    )

    encoder_questions = pad_tokens(
        [transforms_dict["text_encoder"](question) for question in questions]
    )

    decoder_questions = pad_tokens(
        [transforms_dict["text_decoder"](question) for question in questions]
    )

    decoder_answers = pad_tokens(
        [transforms_dict["text_decoder"](answer) for answer in answers]
    )

    input_dict = {
        "image_encoder_tokens": encoder_images,
        "question_encoder_tokens": encoder_questions,
        "question_decoder_tokens": decoder_questions,
        "answer_decoder_tokens": decoder_answers,
    }

    output = model.forward(input_dict)

    assert output["logits"].shape == (3, 8, 50257)

    assert output["loss"] > 0

    output["loss"].backward()


if __name__ == "__main__":
    test_build_model()
    test_model_forward()
    test_model_forward_loss()
