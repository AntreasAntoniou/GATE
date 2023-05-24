from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.data.image_text.zero_shot.imagenet1k import (
    generate_per_class_prompts,
)
from gate.models import ModelAndTransform
from gate.models.backbones.clip import CLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.duo_modal_zero_shot_classification import (
    DuoModalZeroShotModel,
    DuoModalZeroShotModelWithPresetClasses,
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
    modality_a_identifier: str = "image",
    modality_b_identifier: str = "text",
    num_projection_features: Optional[int] = None,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = CLIPAdapter(model_name=model_name, pretrained=pretrained)
    num_feature_dict = {
        "text": backbone_model.text_num_features,
        "image": backbone_model.image_num_features,
    }
    if modality_a_identifier in [
        "image",
        "text",
    ] and modality_b_identifier in ["image", "text"]:
        model = DuoModalZeroShotModel(
            modality_a_model=backbone_model,
            modality_b_model=backbone_model,
            modality_a_identifier=modality_a_identifier,
            modality_b_identifier=modality_b_identifier,
            modality_a_num_features=num_feature_dict[modality_a_identifier],
            modality_b_num_features=num_feature_dict[modality_b_identifier],
            projection_num_features=num_projection_features,
        )
    else:
        raise ValueError(
            f"Modality combination of {modality_a_identifier} {modality_b_identifier} not supported for CLIP."
        )

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        if "image" in inputs:
            image = inputs["image"]
            if isinstance(image, List):
                image = torch.stack(
                    [transform_dict["image"](sample) for sample in image]
                )
            else:
                image = transform_dict["image"](image)
            inputs["image"] = image

        if "text" in inputs:
            text = inputs["text"]
            if isinstance(text, List):
                text = transform_dict["text"](text)
                max_length = max([t.shape[0] for t in text])
                temp_text = (
                    torch.ones((2, max_length), dtype=torch.long) * text[0, -1]
                )
                for i, t in enumerate(text):
                    temp_text[i, : t.shape[0]] = t
                text = temp_text

            else:
                text = transform_dict["text"](text)

            inputs["text"] = text

        return inputs

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="clip-zero-shot-classification",
)
def build_gate_model(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    modality_a_identifier: str = "image",
    modality_b_identifier: str = "text",
    num_projection_features: Optional[int] = None,
):
    model_and_transform = build_model(
        model_name=model_name,
        pretrained=pretrained,
        modality_a_identifier=modality_a_identifier,
        modality_b_identifier=modality_b_identifier,
        num_projection_features=num_projection_features,
    )

    model_modality_config_image_classification = TargetModalityConfig(
        image_text=[SourceModalityConfig(image=True, text=True)]
    )

    gate_model = GATEModel(
        config=model_modality_config_image_classification,
        model=model_and_transform.model,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )


def build_model_with_presets(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    modality_a_identifier: str = "image",
    modality_b_identifier: str = "text",
    num_projection_features: Optional[int] = None,
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = CLIPAdapter(model_name=model_name, pretrained=pretrained)
    num_feature_dict = {
        "text": backbone_model.text_num_features,
        "image": backbone_model.image_num_features,
    }
    if modality_a_identifier in [
        "image",
        "text",
    ] and modality_b_identifier in ["image", "text"]:
        model = DuoModalZeroShotModelWithPresetClasses(
            image_modality_model=backbone_model,
            text_modality_model=backbone_model,
            image_modality_num_features=num_feature_dict["image"],
            projection_num_features=num_projection_features,
            class_prompts=generate_per_class_prompts(),
        )
    else:
        raise ValueError(
            f"Modality combination of {modality_a_identifier} {modality_b_identifier} not supported for CLIP."
        )

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        if "image" in inputs:
            image = inputs["image"]
            if isinstance(image, List):
                image = torch.stack(
                    [transform_dict["image"](sample) for sample in image]
                )
            else:
                image = transform_dict["image"](image)
            inputs["image"] = image

        if "text" in inputs:
            text = inputs["text"]
            if isinstance(text, List):
                text = torch.stack(
                    [transform_dict["text"](sample) for sample in text]
                )
            else:
                text = transform_dict["text"](text)

            inputs["text"] = text

        return inputs

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="clip-zero-shot-classification-with-preset-classes",
)
def build_gate_model_with_presets(
    model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    modality_a_identifier: str = "image",
    modality_b_identifier: str = "text",
    num_projection_features: Optional[int] = None,
):
    model_and_transform = build_model_with_presets(
        model_name=model_name,
        pretrained=pretrained,
        modality_a_identifier=modality_a_identifier,
        modality_b_identifier=modality_b_identifier,
        num_projection_features=num_projection_features,
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
