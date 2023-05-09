from dataclasses import dataclass
from typing import Any, Dict, Union

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.talinet import TALINet
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.classification import BackboneWithLinear


SUPPORTED_MODALITIES = ["image", "text", "audio", "video"]


def create_model_with_linear(backbone_model, num_features, num_classes):
    """
    Helper function to create a model with linear layer.
    """
    return BackboneWithLinear(backbone_model, num_features, num_classes)


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
) -> ModelAndTransform:
    backbone_model = TALINet(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        pretrained=pretrained,
    )

    # Check if modality is supported, if not raise a ValueError.
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Modality {modality} not supported for CLIP.")

    num_features_attr = f"{modality}_num_features"
    num_features = getattr(backbone_model, num_features_attr, None)
    model = create_model_with_linear(backbone_model, num_features, num_classes)

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        return {
            key: transform_dict[key](value)
            if key in SUPPORTED_MODALITIES
            else inputs["labels"]
            for key, value in inputs.items()
        }

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="tali-classification",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_tali_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        modality=modality,
    )

    # Create Modality Config based on input modality
    model_modality_config = TargetModalityConfig(
        **{modality: [SourceModalityConfig(**{modality: True})]}
    )

    model_key_remapper_dict_config = {
        "image": "image",
        "text": "image",
        "audio": "audio",
        "video": "video",
    }

    gate_model = GATEModel(
        config=model_modality_config,
        model=model_and_transform.model,
        key_remapper_dict=model_key_remapper_dict_config,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )
