from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from numpy import isin

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.clip import CLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.few_shot_classification.protonet import (
    PrototypicalNetwork,
)


def build_model(
    model_name: str = "openai/clip-vit-base-patch16",
    modality: str = "image",
    pretrained: bool = True,
    num_output_features: int = 100,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = CLIPAdapter(model_name=model_name, pretrained=pretrained)
    if modality in ["image", "text"]:
        model = PrototypicalNetwork(
            model=backbone_model,
            num_clip_features={
                "text": backbone_model.text_num_features,
                "image": backbone_model.image_num_features,
            }[modality],
            num_output_features=num_output_features,
            modality=modality,
        )
    else:
        raise ValueError(f"Modality {modality} not supported for CLIP.")

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        output_dict = {}

        for modality in inputs.keys():
            temp_shape = inputs[modality]["support_set"].shape
            inputs[modality]["support_set"] = transform_dict[modality](
                inputs[modality]["support_set"].view(-1, *temp_shape[-3:])
            ).view(temp_shape)

            temp_shape = inputs[modality]["query_set"].shape
            inputs[modality]["query_set"] = transform_dict[modality](
                inputs[modality]["query_set"].view(-1, *temp_shape[-3:])
            ).view(temp_shape)

            output_dict[modality] = inputs[modality]

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="clip-protonet-few-shot-classification",
)
def build_gate_model(
    model_name: str = "openai/clip-vit-base-patch16",
    modality: str = "image",
    pretrained: bool = True,
    num_output_features: int = 512,
):
    model_and_transform = build_model(
        model_name=model_name,
        pretrained=pretrained,
        num_output_features=num_output_features,
        modality=modality,
    )
    if modality == "image":
        model_modality_config_image_classification = TargetModalityConfig(
            image=[SourceModalityConfig(image=True)]
        )
    elif modality == "text":
        model_modality_config_image_classification = TargetModalityConfig(
            text=[SourceModalityConfig(text=True)]
        )

    model_key_remapper_dict_config = {
        "image": "image",
        "text": "image",
    }

    gate_model = GATEModel(
        config=model_modality_config_image_classification,
        model=model_and_transform.model,
        key_remapper_dict=model_key_remapper_dict_config,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )
