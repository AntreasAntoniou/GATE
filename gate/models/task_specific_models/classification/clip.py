from typing import Any, Dict, Union

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.clip_image import (
    CLIPModelPaths,
    VisionTextGATEAdapter,
)
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.classification import BackboneWithLinear


def build_model(
    model_name: str = "openai/clip-vit-base-patch16",
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
    backbone_model = VisionTextGATEAdapter(
        model_name=model_name, pretrained=pretrained
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

    backbone_transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        output_dict = {}

        if "image" in inputs:
            output_dict["image"] = backbone_transform_dict["image"](
                inputs["image"]
            )

        if "text" in inputs:
            output_dict["text"] = backbone_transform_dict["text"](
                inputs["text"]
            )

        if "labels" in inputs:
            output_dict["labels"] = inputs["labels"]

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="clip-classification",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_model(
    model_name: str = CLIPModelPaths.openai_b_16,
    modality: str = "image",
    pretrained: bool = True,
    num_classes: int = 100,
    allow_on_model_metric_computation: bool = True,
):
    model_and_transform = build_model(
        model_name=model_name,
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

    gate_model = GATEModel(
        config=model_modality_config_image_classification,
        model=model_and_transform.model,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )
