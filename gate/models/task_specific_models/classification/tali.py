from typing import Any, Dict, Union

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.tali import TALINet
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.classification import BackboneWithLinear

SUPPORTED_MODALITIES = ["image", "text", "audio", "video"]


def create_model_with_linear(
    backbone_model,
    num_features,
    num_classes,
    modality,
    allow_on_model_metric_computation,
):
    """
    Helper function to create a model with linear layer.
    """
    model = BackboneWithLinear(
        model=backbone_model,
        num_clip_features=num_features,
        num_classes=num_classes,
        modality=modality,
    )
    if not allow_on_model_metric_computation:
        del model.compute_loss_and_metrics

    return model


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
    allow_on_model_metric_computation: bool = True,
) -> ModelAndTransform:
    backbone_model = TALINet(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
    )

    # Check if modality is supported, if not raise a ValueError.
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Modality {modality} not supported for CLIP.")

    num_features_attr = f"{modality}_num_features"
    num_features = getattr(backbone_model, num_features_attr, None)
    model = create_model_with_linear(
        backbone_model,
        num_features,
        num_classes,
        modality,
        allow_on_model_metric_computation,
    )

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
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
    allow_on_model_metric_computation: bool = True,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
        num_classes=num_classes,
        modality=modality,
        allow_on_model_metric_computation=allow_on_model_metric_computation,
    )

    # Create Modality Config based on input modality
    if modality == "image":
        model_modality_config = TargetModalityConfig(
            image=[SourceModalityConfig(image=True)]
        )
    elif modality == "text":
        model_modality_config = TargetModalityConfig(
            text=[SourceModalityConfig(text=True)]
        )
    elif modality == "audio":
        model_modality_config = TargetModalityConfig(
            audio=[SourceModalityConfig(audio=True)]
        )
    elif modality == "video":
        model_modality_config = TargetModalityConfig(
            video=[SourceModalityConfig(video=True)]
        )

    model_key_remapper_dict_config = {
        "image": "image",
        "text": "image",
        "audio": "audio",
        "video": "video",
        "pixel_values": "image",
    }

    gate_model = GATEModel(
        config=model_modality_config,
        model=model_and_transform.model,
        key_remapper_dict=model_key_remapper_dict_config,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )
