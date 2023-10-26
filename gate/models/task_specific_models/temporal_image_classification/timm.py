from typing import Any, Dict, Union

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models import ModelAndTransform
from gate.models.backbones.timm import TimmCLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.temporal_image_classification import (
    BackboneWithTemporalTransformerAndLinear,
    get_classification_metrics_fn_dict,
)


def build_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    timm_model_name: str = "resnet50.a1_in1k",
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
    backbone_model = TimmCLIPAdapter(
        clip_model_name=clip_model_name,
        timm_model_name=timm_model_name,
        pretrained=pretrained,
    )

    model = BackboneWithTemporalTransformerAndLinear(
        model=backbone_model,
        num_backbone_features=backbone_model.image_num_features,
        num_classes=num_classes,
        metric_fn_dict=get_classification_metrics_fn_dict(
            num_classes=num_classes
        ),
    )

    if not pretrained:
        model.init_weights()

    transform_dict = backbone_model.get_transforms()

    def transform_wrapper(inputs: Union[Dict, Any]):
        if isinstance(inputs, dict):
            output_dict = {
                "video": transform_dict["video"](inputs["video"]),
            }
        else:
            output_dict = {"video": transform_dict["video"](inputs)}

        inputs.update(output_dict)
        output_dict = inputs

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


@configurable(
    group="model",
    name="timm-temporal-classification",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
def build_gate_model(
    clip_model_name: str = "openai/clip-vit-base-patch16",
    timm_model_name: str = "resnet50.a1_in1k",
    pretrained: bool = True,
    num_classes: int = 512,
):
    model_and_transform = build_model(
        clip_model_name=clip_model_name,
        timm_model_name=timm_model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    model_modality_config_video_classification = TargetModalityConfig(
        video=[SourceModalityConfig(video=True)]
    )

    gate_model = GATEModel(
        config=model_modality_config_video_classification,
        model=model_and_transform.model,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )