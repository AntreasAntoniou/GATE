from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.tali import TALINet
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.duo_modal_zero_shot_classification import (
    DuoModalZeroShotModel,
)
from gate.models.task_adapters.semantic_segmentation import SegmentationViT

# modality_a_model: nn.Module,
# modality_b_model: nn.Module,
# modality_a_identifier: str,
# modality_b_identifier: str,
# modality_a_num_features: int,
# modality_b_num_features: int,
# projection_num_features: Optional[int] = None,


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    pretrained: bool = True,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    mlp_ratio: float = 4.0,
    num_classes: int = HYDRATED_NUM_CLASSES,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = TALINet(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
    )
    model = SegmentationViT(
        encoder_model=backbone_model,
        embed_dim=backbone_model.image_num_features,
        decoder_embed_dim=backbone_model.image_num_features,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
    )

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        output_dict = {}

        if "image" in inputs:
            output_dict["image"] = transform_dict["image"](inputs["image"])

        if "text" in inputs:
            output_dict["text"] = transform_dict["text"](inputs["text"])

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="tali-segmentation-transformer",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    pretrained: bool = True,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    mlp_ratio: float = 4.0,
    num_classes: int = HYDRATED_NUM_CLASSES,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
    )

    model_modality_config_image_classification = TargetModalityConfig(
        image_text=[SourceModalityConfig(image=True, text=True)]
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
