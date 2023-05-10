from urllib.request import urlopen
import pytest
import accelerate
import PIL.Image as Image
import torch
from gate.models.backbones.clip import CLIPAdapter
from gate.models.task_adapters.vqa.simple_vqa_transformer import (
    SimpleVQATransformer,
)


def test_generate():
    accelerator = accelerate.Accelerator()

    backbone_model = CLIPAdapter(
        model_name="openai/clip-vit-base-patch16", pretrained=True
    )
    clip_transforms = backbone_model.get_transforms()
    simple_vqa_transformer = SimpleVQATransformer(
        image_encoder=backbone_model.vision_model,
        image_encoder_transforms=clip_transforms["image"],
        image_encoder_num_features=768,
        text_encoder=backbone_model.text_model,
        text_encoder_transforms=clip_transforms["text"],
        text_encoder_num_features=512,
    )
    simple_vqa_transformer = accelerator.prepare(simple_vqa_transformer)
    vqa_transforms = simple_vqa_transformer.get_transforms()

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
        vqa_transforms["image_encoder"](image) for image in images
    ]
    encoder_questions = vqa_transforms["text_encoder"](questions)
    decoder_questions = vqa_transforms["text_decoder"](questions)
    decoder_answers = vqa_transforms["text_decoder"](answers)

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
        answer, str
    ), "The generate function should return a string"
