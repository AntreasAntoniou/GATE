from typing import List
from urllib.request import urlopen
import torch
import torch.nn.functional as F

from gate.models.visual_question_answering.clip import (
    ModelAndTransform,
    build_model,
)
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
import PIL.Image as Image


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


def pad_tokens(token_list: List):
    token_list = [torch.tensor(token).squeeze() for token in token_list]
    max_len = max([len(question) for question in token_list])

    encoder_questions_tensor = torch.zeros(
        (len(token_list), max_len), dtype=torch.long
    )
    print(f"encoder_questions_tensor: {encoder_questions_tensor.shape}")

    for i, question in enumerate(token_list):
        print("question: ", question.shape)
        encoder_questions_tensor[i, : len(question)] = question

    return encoder_questions_tensor


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
