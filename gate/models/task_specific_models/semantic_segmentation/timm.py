import logging
from typing import Any, Dict, Union

import torch
from omegaconf import DictConfig

from gate.boilerplate.decorators import configurable
from gate.config.variables import (
    HYDRATED_IMAGE_SIZE,
    HYDRATED_NUM_CLASSES,
    HYDRATED_TASK_NAME,
)
from gate.models import ModelAndTransform
from gate.models.backbones.timm import TimmCLIPAdapter
from gate.models.core import (
    GATEModel,
    SourceModalityConfig,
    TargetModalityConfig,
)
from gate.models.task_adapters.semantic_segmentation import (
    SegmentationAdapter,
    SegmentationAdapterOptions,
    SegmentationLossOptions,
)

# modality_a_model: nn.Module,
# modality_b_model: nn.Module,
# modality_a_identifier: str,
# modality_b_identifier: str,
# modality_a_num_features: int,
# modality_b_num_features: int,
# projection_num_features: Optional[int] = None,

logger = logging.getLogger(__name__)


def build_model(
    timm_model_name: str = "resnet50.a1_in1k",
    clip_model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    decoder_num_blocks: int = 2,
    decoder_num_heads: int = 8,
    num_classes: int = 10,
    image_size: int = 512,
    decoder_layer_type: str = "transformer",
    ignore_index: int = 0,
    background_loss_weight: float = 0.01,
    loss_type_id: str = "default",
) -> ModelAndTransform:
    """
    🏗️ Build the model using the Hugging Face transformers library.

    :param model_name: The name of the model to load.
    :param pretrained: Whether to use a pretrained model.
    :param num_classes: The number of classes for the linear layer.
    :return: A ModelAndTransform instance containing the model and transform function.
    """
    backbone_model = TimmCLIPAdapter(
        timm_model_name=timm_model_name,
        clip_model_name=clip_model_name,
        pretrained=pretrained,
        image_size=image_size,
    )

    model = SegmentationAdapter(
        encoder=backbone_model,
        decoder_embed_dim=backbone_model.image_num_features,
        num_classes=num_classes,
        decoder_layer_type=decoder_layer_type,
        decoder_num_blocks=decoder_num_blocks,
        decoder_num_heads=decoder_num_heads,
        ignore_index=ignore_index,
        decoder_target_image_size=(64, 64),
        background_loss_weight=background_loss_weight,
        loss_type_id=loss_type_id,
    )

    x = torch.randn(1, 3, image_size, image_size)
    logger.info(f"x build shape: {x.shape}")
    _ = model.forward(x)

    # forward features for conv nets, and get the patches for the transformer manually
    # do the same for all others? sounds like the most general way to do this

    if not pretrained:
        backbone_model.init_weights()

    transform_dict = backbone_model.get_transforms()

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
    name="timm-md-segmentation-transformer",
    defaults=dict(
        num_classes=HYDRATED_NUM_CLASSES,
        image_size=HYDRATED_IMAGE_SIZE,
        task_name=HYDRATED_TASK_NAME,
    ),
)
def build_gate_md_model(
    timm_model_name: str = "resnet50.a1_in1k",
    clip_model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    num_classes: int = 10,
    image_size: int = 512,
    decoder_layer_type: str = SegmentationAdapterOptions.TRANSFORMER.value,
    ignore_index: int = 0,
    background_loss_weight: float = 0.01,
    task_name: str = "task01braintumour",
):
    if isinstance(num_classes, dict) or isinstance(num_classes, DictConfig):
        num_classes = len(num_classes[task_name])

    model_and_transform = build_model(
        timm_model_name=timm_model_name,
        clip_model_name=clip_model_name,
        pretrained=pretrained,
        decoder_num_blocks=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        num_classes=num_classes,
        image_size=image_size,
        decoder_layer_type=decoder_layer_type,
        ignore_index=ignore_index,
        background_loss_weight=background_loss_weight,
        loss_type_id=SegmentationLossOptions.MD.value,
    )

    model_modality_config_image_classification = TargetModalityConfig(
        image=[SourceModalityConfig(image=True)]
    )

    gate_model = GATEModel(
        config=model_modality_config_image_classification,
        model=model_and_transform.model,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )


@configurable(
    group="model",
    name="timm-segmentation-transformer",
    defaults=dict(
        num_classes=HYDRATED_NUM_CLASSES,
        image_size=HYDRATED_IMAGE_SIZE,
    ),
)
def build_gate_model(
    timm_model_name: str = "resnet50.a1_in1k",
    clip_model_name: str = "openai/clip-vit-base-patch16",
    pretrained: bool = True,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    num_classes: int = 10,
    image_size: int = 512,
    decoder_layer_type: str = SegmentationAdapterOptions.TRANSFORMER.value,
    ignore_index: int = 0,
    background_loss_weight: float = 0.1,
):
    model_and_transform = build_model(
        timm_model_name=timm_model_name,
        clip_model_name=clip_model_name,
        pretrained=pretrained,
        decoder_num_blocks=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        num_classes=num_classes,
        image_size=image_size,
        decoder_layer_type=decoder_layer_type,
        ignore_index=ignore_index,
        background_loss_weight=background_loss_weight,
        loss_type_id=SegmentationLossOptions.DEFAULT.value,
    )

    model_modality_config_image_classification = TargetModalityConfig(
        image=[SourceModalityConfig(image=True)]
    )

    gate_model = GATEModel(
        config=model_modality_config_image_classification,
        model=model_and_transform.model,
    )

    return ModelAndTransform(
        model=gate_model, transform=model_and_transform.transform
    )
