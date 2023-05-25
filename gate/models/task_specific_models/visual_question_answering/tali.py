import random
from collections import defaultdict
from copy import copy
from typing import Any, Dict, Union

from gate.boilerplate.decorators import configurable
from gate.models import ModelAndTransform
from gate.models.backbones.clip import CLIPAdapter
from gate.models.backbones.tali import TALINet
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.simple_vqa_transformer import (
    SimpleVQATransformer,
)
from gate.models.task_specific_models.visual_question_answering import (
    transform_wrapper,
)

SUPPORTED_MODALITIES = ["image", "text", "audio", "video"]


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
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
    backbone_model = TALINet(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
    )

    tali_transforms = backbone_model.get_transforms()

    model = SimpleVQATransformer(
        image_encoder=backbone_model,
        image_encoder_transforms=tali_transforms["image"],
        image_encoder_num_features=backbone_model.image_num_features,
        text_encoder=backbone_model,
        text_encoder_transforms=tali_transforms["text"],
        text_encoder_num_features=backbone_model.text_num_features,
    )

    if not pretrained:
        model.init_weights()

    transform_dict = model.get_transforms()

    return ModelAndTransform(
        model=model,
        transform=lambda x: transform_wrapper(
            x, transform_dict=transform_dict
        ),
    )


@configurable(
    group="model",
    name="tali-vqa",
)
def build_gate_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    pretrained: bool = True,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
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
