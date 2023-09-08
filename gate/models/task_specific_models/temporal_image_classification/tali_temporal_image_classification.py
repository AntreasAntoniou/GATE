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
from gate.models.task_adapters.temporal_image_classification import (
    BackboneWithTemporalTransformerAndLinear,
)


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
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
    backbone_model = TALINet(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
        pretrained=pretrained,
    )
    if modality in ["image", "text"]:
        model = BackboneWithTemporalTransformerAndLinear(
            model=backbone_model,
            num_backbone_features={
                "text": backbone_model.text_num_features,
                "image": backbone_model.image_num_features,
            }[modality],
            num_classes=num_classes,
            modality=modality,
        )
    else:
        raise ValueError(f"Modality {modality} not supported for TALI.")

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        output_dict = {}

        for modality in inputs.keys():
            temp_shape = inputs[modality]["support_set_inputs"].shape
            inputs[modality]["support_set_inputs"] = transform_dict[modality](
                inputs[modality]["support_set_inputs"].view(
                    -1, *temp_shape[-3:]
                )
            ).view(temp_shape)

            temp_shape = inputs[modality]["support_set_inputs"].shape
            inputs[modality]["query_set_inputs"] = transform_dict[modality](
                inputs[modality]["query_set_inputs"].view(-1, *temp_shape[-3:])
            ).view(temp_shape)

            output_dict[modality] = inputs[modality]

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="tali-temporal-classification",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    whisper_model_name: str = "openai/whisper-small",
    model_repo_path: str = "Antreas/tali-2-tali_omni_base_patch16_224-wit_tali_image_text_audio_video_dataset-2306",
    checkpoint_identifier: str = "latest",
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 512,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        model_repo_path=model_repo_path,
        checkpoint_identifier=checkpoint_identifier,
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
