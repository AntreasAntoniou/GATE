from collections import defaultdict
from copy import copy
import random
from typing import Any, Dict, Union

from gate.boilerplate.decorators import configurable
from gate.models import ModelAndTransform
from gate.models.backbones.clip import CLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.simple_vqa_transformer import (
    SimpleVQATransformer,
)


def build_model(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model
    and transform function.
    """
    backbone_model = CLIPAdapter(model_name=model_name, pretrained=pretrained)

    clip_transforms = backbone_model.get_transforms()

    model = SimpleVQATransformer(
        image_encoder=backbone_model,
        image_encoder_transforms=clip_transforms["image"],
        image_encoder_num_features=backbone_model.image_num_features,
        text_encoder=backbone_model,
        text_encoder_transforms=clip_transforms["text"],
        text_encoder_num_features=backbone_model.text_num_features,
    )

    if not pretrained:
        model.init_weights()

    transform_dict = model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        output_dict = defaultdict(dict)

        if "image" in inputs:
            output_dict["image"]["image_encoder_tokens"] = transform_dict[
                "image_encoder"
            ](inputs["image"])

        if "text" in inputs and "question" in inputs["text"]:
            output_dict["text"]["encoder_question_tokens"] = transform_dict[
                "text_encoder"
            ](copy(inputs["text"]["question"]))

            output_dict["text"]["decoder_question_tokens"] = transform_dict[
                "text_decoder"
            ](copy(inputs["text"]["question"]))

        if "text" in inputs and "answers" in inputs["text"]:
            random_idx = random.randint(0, len(inputs["text"]["answers"]) - 1)
            output_dict["text"]["decoder_answer_tokens"] = transform_dict[
                "text_decoder"
            ](copy(inputs["text"]["answers"])[random_idx])

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="clip-vqa",
)
def build_gate_clip_model(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
):
    model_and_transform = build_model(
        model_name=model_name,
        pretrained=pretrained,
    )

    vqa_text_generation = TargetModalityConfig(
        text=[SourceModalityConfig(image=True, text=True)]
    )

    gate_model = GATEModel(
        config=vqa_text_generation, model=model_and_transform.model
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )
