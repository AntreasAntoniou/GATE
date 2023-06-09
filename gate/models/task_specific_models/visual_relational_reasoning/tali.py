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
from gate.models.task_adapters.visual_relational_reasoning_classification import (
    DuoModalFusionModel,
)


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    pretrained: bool = True,
    modality_a_identifier: str = "image",
    modality_b_identifier: str = "text",
    num_projection_features: int = 512,
    dropout_fusion_prob: float = 0.0,
    num_classes: int = 10,
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
    num_feature_dict = {
        "text": backbone_model.text_num_features,
        "image": backbone_model.image_num_features,
        "audio": backbone_model.audio_num_features,
        "video": backbone_model.video_num_features,
    }
    if modality_a_identifier in [
        "image",
        "text",
        "audio",
        "video",
    ] and modality_b_identifier in ["image", "text", "audio", "video"]:
        model = DuoModalFusionModel(
            modality_a_model=backbone_model,
            modality_b_model=backbone_model,
            modality_a_identifier=modality_a_identifier,
            modality_b_identifier=modality_b_identifier,
            modality_a_num_features=num_feature_dict[modality_a_identifier],
            modality_b_num_features=num_feature_dict[modality_b_identifier],
            projection_num_features=num_projection_features,
            dropout_fusion_prob=dropout_fusion_prob,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Modality combination of {modality_a_identifier} {modality_b_identifier} not supported for TALI."
        )

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        if "image" in inputs:
            inputs["image"] = transform_dict["image"](
                inputs["image"]
            ).squeeze()

        if "text" in inputs:
            inputs["text"] = transform_dict["text"](inputs["text"])

        if "labels" in inputs:
            inputs["labels"] = inputs["labels"]

        return inputs

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="tali-relational-reasoning",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    pretrained: bool = True,
    modality_a_identifier: str = "image",
    modality_b_identifier: str = "text",
    num_projection_features: int = 512,
    dropout_fusion_prob: float = 0.0,
    num_classes: int = 10,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
        modality_a_identifier=modality_a_identifier,
        modality_b_identifier=modality_b_identifier,
        num_projection_features=num_projection_features,
        dropout_fusion_prob=dropout_fusion_prob,
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


colour_dict = {
    "blue": 0,
    "brown": 1,
    "cyan": 2,
    "gray": 3,
    "green": 4,
    "purple": 5,
    "red": 6,
    "yellow": 7,
}

shape_dict = {
    "cube": 0,
    "cylinder": 1,
    "sphere": 2,
}

count_dict = {
    "0": 0,
    "1": 1,
    "10": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
}

size_dict = {
    "large": 0,
    "small": 1,
}

yes_no_dict = {"no": 0, "yes": 1}

material_dict = {"metal": 0, "rubber": 1}


@configurable(
    group="model",
    name="tali-relational-reasoning-multi-task",
)
def build_gate_model_multi_task(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    pretrained: bool = True,
    modality_a_identifier: str = "image",
    modality_b_identifier: str = "text",
    num_projection_features: int = 512,
    dropout_fusion_prob: float = 0.0,
    num_classes: Dict = {
        "colour": len(colour_dict),
        "shape": len(shape_dict),
        "count": len(count_dict),
        "size": len(size_dict),
        "yes_no": len(yes_no_dict),
        "material": len(material_dict),
    },
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
        modality_a_identifier=modality_a_identifier,
        modality_b_identifier=modality_b_identifier,
        num_projection_features=num_projection_features,
        dropout_fusion_prob=dropout_fusion_prob,
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
