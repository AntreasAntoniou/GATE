from gate.models import ModelAndTransform
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.clip import CLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.classification import (
    BackboneWithLinear,
)


def build_model(
    model_name: str = "openai/clip-vit-base-patch16",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = CLIPAdapter(model_name=model_name, pretrained=pretrained)
    if modality == "image":
        model = BackboneWithLinear(
            backbone_model, backbone_model.clip.vision_embed_dim, num_classes
        )
    elif modality == "text":
        model = BackboneWithLinear(
            backbone_model, backbone_model.clip.text_embed_dim, num_classes
        )
    else:
        raise ValueError(f"Modality {modality} not supported for CLIP.")

    if not pretrained:
        model.init_weights()

    transform = lambda image: backbone_model.preprocessor(
        images=image, return_tensors="pt"
    )

    def transform_wrapper(inputs: Union[Dict, Any]):
        output_dict = {}

        if "image" in inputs:
            output_dict["image"] = transform(inputs["image"])["pixel_values"][
                0
            ]
        if "text" in inputs:
            output_dict["text"] = transform(inputs["text"])["input_ids"][0]

        if "labels" in inputs:
            output_dict["labels"] = inputs["labels"]

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="clip-classification",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_clip_model(
    model_name: str = "openai/clip-vit-base-patch16",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
):
    model_and_transform = build_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
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
