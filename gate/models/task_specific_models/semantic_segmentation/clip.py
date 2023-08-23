from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_IMAGE_SIZE, HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.clip import CLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.semantic_segmentation import (
    SimpleSegmentationDecoder,
)

# modality_a_model: nn.Module,
# modality_b_model: nn.Module,
# modality_a_identifier: str,
# modality_b_identifier: str,
# modality_a_num_features: int,
# modality_b_num_features: int,
# projection_num_features: Optional[int] = None,


def build_model(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    mlp_ratio: float = 4.0,
    num_classes: int = 10,
    image_size: int = 512,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = CLIPAdapter(
        model_name=model_name, pretrained=pretrained, image_size=image_size
    )

    model = SimpleSegmentationDecoder(
        encoder_model=backbone_model,
        embed_dim=backbone_model.image_num_features,
        decoder_embed_dim=backbone_model.image_num_features,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        num_patches=backbone_model.vision_model.embeddings.num_patches,
    )
    x = torch.randn(2, 3, image_size, image_size)
    dummy_out = model.forward(x)

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms(image_size=image_size)

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
    name="clip-segmentation-transformer",
    defaults=dict(
        num_classes=HYDRATED_NUM_CLASSES, image_size=HYDRATED_IMAGE_SIZE
    ),
)
def build_gate_model(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    mlp_ratio: float = 4.0,
    num_classes: int = 10,
    image_size: int = 512,
):
    model_and_transform = build_model(
        model_name=model_name,
        pretrained=pretrained,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        image_size=image_size,
    )

    model_modality_config_image_classification = TargetModalityConfig(
        image=[SourceModalityConfig(image=True)]
    )

    model_key_remapper_dict_config = {
        "image": "image",
        "text": "text",
    }

    gate_model = GATEModel(
        config=model_modality_config_image_classification,
        model=model_and_transform.model,
        key_remapper_dict=model_key_remapper_dict_config,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )
