from urllib.request import urlopen

import accelerate
import PIL.Image as Image
import pytest
import torch
from tests.models.test_clip_vqa_model import pad_tokens

from gate.models.backbones.clip import CLIPAdapter
from gate.models.task_adapters.simple_vqa_transformer import (
    SimpleVQATransformer,
)


def test_generate():
    accelerator = accelerate.Accelerator()

    backbone_model = CLIPAdapter(
        model_name="openai/clip-vit-base-patch16", pretrained=True
    )
    clip_transforms = backbone_model.get_transforms()
    simple_vqa_transformer = SimpleVQATransformer(
        image_encoder=backbone_model,
        image_encoder_transforms=clip_transforms["image"],
        image_encoder_num_features=768,
        text_encoder=backbone_model,
        text_encoder_transforms=clip_transforms["text"],
        text_encoder_num_features=512,
    )
    simple_vqa_transformer = accelerator.prepare(simple_vqa_transformer)
    transforms_dict = simple_vqa_transformer.get_transforms()

    # Prepare the input data
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

    encoder_images = [
        transforms_dict["image_encoder"](image) for image in images
    ]
    encoder_questions = pad_tokens(
        [transforms_dict["text_encoder"](question) for question in questions]
    )

    decoder_questions = pad_tokens(
        [transforms_dict["text_decoder"](question) for question in questions]
    )

    decoder_answers = pad_tokens(
        [transforms_dict["text_decoder"](answer) for answer in answers]
    )

    encoder_images = torch.stack(encoder_images).to(accelerator.device)
    encoder_questions = encoder_questions.to(accelerator.device)
    decoder_questions = decoder_questions.to(accelerator.device)
    decoder_answers = decoder_answers.to(accelerator.device)

    # Call the generate function
    answer = simple_vqa_transformer.generate_text(
        image_encoder_tokens=encoder_images,
        question_encoder_tokens=encoder_questions,
        question_decoder_tokens=decoder_questions,
        max_length=10,
    )
    print(answer)
    # Check that the function returned a string
    assert isinstance(
        answer[0], str
    ), "The generate function should return a string"
