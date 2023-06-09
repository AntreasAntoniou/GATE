from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from numpy import isin

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.timm import TimmCLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.classification import BackboneWithLinear


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    timm_model_name: str = "resnet50.a1_in1k",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
    allow_on_model_metric_computation: bool = True,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = TimmCLIPAdapter(
        clip_model_name=clip_model_name,
        timm_model_name=timm_model_name,
        pretrained=pretrained,
    )
    if modality in ["image", "text"]:
        model = BackboneWithLinear(
            backbone_model,
            {
                "text": backbone_model.text_num_features,
                "image": backbone_model.image_num_features,
            }[modality],
            num_classes,
            modality=modality,
            allow_on_model_metric_computation=allow_on_model_metric_computation,
        )

    else:
        raise ValueError(f"Modality {modality} not supported for CLIP.")

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        output_dict = {}

        if "image" in inputs:
            output_dict["image"] = transform_dict["image"](inputs["image"])

        if "text" in inputs:
            output_dict["text"] = transform_dict["text"](inputs["text"])

        if "labels" in inputs:
            output_dict["labels"] = inputs["labels"]

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="timm-classification",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_timm_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    timm_model_name: str = "resnet50.a1_in1k",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
    allow_on_model_metric_computation: bool = True,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        timm_model_name=timm_model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        modality=modality,
        allow_on_model_metric_computation=allow_on_model_metric_computation,
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
