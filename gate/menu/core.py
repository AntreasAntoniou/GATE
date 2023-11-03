from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from gate.data.few_shot import DatasetName as few_shot_dataset_name
from gate.data.image.classification import (
    DatasetName as image_class_dataset_name,
)
from gate.data.image.segmentation import DatasetName as image_seg_dataset_name
from gate.data.image_text.visual_relational_reasoning import (
    DatasetName as image_rr_dataset_name,
)
from gate.data.image_text.zero_shot import (
    DatasetName as image_text_dataset_name,
)
from gate.data.medical.classification import (
    DatasetName as med_classification_dataset_name,
)
from gate.data.medical.segmentation import ACDCDatasetName as acdc_dataset_name
from gate.data.medical.segmentation import MD_DatasetName as md_options
from gate.data.video import DatasetName as video_dataset_name
from gate.data.video import (
    RegressionDatasetName as video_regression_dataset_name,
)


class DatasetName(Enum):
    IMAGE_CLASSIFICATION = image_class_dataset_name
    IMAGE_SEGMENTATION = image_seg_dataset_name
    IMAGE_TEXT_ZERO_SHOT_CLASSIFICATION = image_text_dataset_name
    MEDICAL_DECATHLON_SEGMENTATION = md_options
    MEDICAL_ACDC_SEGMENTATION = acdc_dataset_name
    MEDICAL_CLASSIFICATION = med_classification_dataset_name
    VISUAL_RELATIONAL_REASONING = image_rr_dataset_name
    FEW_SHOT_PROTONET_CLASSIFICATION = few_shot_dataset_name
    VIDEO_CLASSIFICATION = video_dataset_name
    VIDEO_REGRESSION = video_regression_dataset_name


class AdapterTypeNames(Enum):
    TIMM_IMAGE_CLASSIFICATION = "timm-classification"
    TIMM_FEW_SHOT_PROTONET = "timm-protonet-few-shot-classification"
    TIMM_SEGMENTATION = "timm-segmentation-transformer"
    TIMM_MD_SEGMENTATION = "timm-md-segmentation-transformer"
    TIMM_TEMPORAL_CLASSIFICATION = "timm-temporal-classification"
    TIMM_TEMPORAL_REGRESSION = "timm-temporal-regression"
    TIMM_RELATIONAL_REASONING = "timm-relational-reasoning"
    TIMM_RELATIONAL_REASONING_MULTI_TASK = (
        "timm-relational-reasoning-multi-task"
    )

    TIMM_ZERO_SHOT_IMAGE_TEXT = "timm-zero-shot-classification"


class TrainerName(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"
    VISUAL_RELATIONAL_REASONING = "visual_relational_reasoning"
    VIDEO_CLASSIFICATION = "video_classification"
    VIDEO_REGRESSION = "video_regression"
    IMAGE_SEMANTIC_SEGMENTATION = "image_semantic_segmentation"
    MEDICAL_SEMANTIC_SEGMENTATION = "medical_semantic_segmentation"
    IMAGE_TO_TEXT_ZERO_SHOT_CLASSIFICATION = (
        "image_to_text_zero_shot_classification"
    )


class EvaluatorName(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"
    VISUAL_RELATIONAL_REASONING = "visual_relational_reasoning"
    VIDEO_CLASSIFICATION = "video_classification"
    VIDEO_REGRESSION = "video_regression"
    IMAGE_SEMANTIC_SEGMENTATION = "image_semantic_segmentation"
    MEDICAL_SEMANTIC_SEGMENTATION = "medical_semantic_segmentation"
    IMAGE_TO_TEXT_ZERO_SHOT_CLASSIFICATION = (
        "image_to_text_zero_shot_classification"
    )


@dataclass
class LearningRateConfig:
    default: List[float]
    dataset_specific: Dict[DatasetName, List[float]] = field(
        default_factory=dict
    )

    def get_lr(self, dataset_name: Optional[DatasetName] = None):
        if dataset_name in self.dataset_specific and dataset_name is not None:
            return self.dataset_specific[dataset_name]
        else:
            return self.default


@dataclass
class EncoderConfig:
    pretty_name: str
    timm_model_name: Optional[str] = None


# Create an Enum to store EncoderConfigs
class EncoderNames(Enum):
    ResNet50A1 = EncoderConfig(
        pretty_name="R50A1", timm_model_name="resnet50.a1_in1k"
    )
    WideResNet50_2TV = EncoderConfig(
        pretty_name="WR50_2TV", timm_model_name="wide_resnet50_2.tv_in1k"
    )
    ResNeXt50_32x4dA1 = EncoderConfig(
        pretty_name="RNX50_32x4A1", timm_model_name="resnext50_32x4d.a1_in1k"
    )
    SamViTBase16_224 = EncoderConfig(
        pretty_name="SViT_B16_224",
        timm_model_name="vit_base_patch16_224.sam_in1k",
    )
    AugRegViTBase16_224 = EncoderConfig(
        pretty_name="AR_ViT_B16_224",
        timm_model_name="vit_base_patch16_224.augreg_in1k",
    )
    DiNoViTBase16_224 = EncoderConfig(
        pretty_name="DINO_B16_224", timm_model_name="vit_base_patch16_224.dino"
    )
    CLIPViTBase16_224 = EncoderConfig(
        pretty_name="CLIP_B16_224", timm_model_name=None
    )
    LaionViTBase16_224 = EncoderConfig(
        pretty_name="Laion_B16_224",
        timm_model_name="vit_base_patch16_clip_224.laion2b",
    )
    EfficientFormer_s0 = EncoderConfig(
        pretty_name="EffFormer_s0",
        timm_model_name="efficientformerv2_s0",
    )
    EffNetV2_RW_S_RA2 = EncoderConfig(
        pretty_name="EffV2_RW_S",
        timm_model_name="efficientnetv2_rw_s.ra2_in1k",
    )
    ConvNextV2_Base = EncoderConfig(
        pretty_name="ConvNextV2_Base",
        timm_model_name="convnextv2_base",
    )
    DeiT3BasePatch16_224 = EncoderConfig(
        pretty_name="DeiT3_B16_224",
        timm_model_name="deit3_base_patch16_224.fb_in1k",
    )
    FlexViTBase_1200EP = EncoderConfig(
        pretty_name="Flex_B_1200EP",
        timm_model_name="flexivit_base.1200ep_in1k",
    )
    IJEPAViTHugePatch14_224 = EncoderConfig(
        pretty_name="IJEPA_Huge_P14_224",
        timm_model_name="vit_huge_patch14_gap_224.in22k_ijepa",
    )
    SIGLIPPathch16_224 = EncoderConfig(
        pretty_name="SIGLIP_P16_224",
        timm_model_name="vit_base_patch16_siglip_224",
    )


@dataclass
class ModelConfig:
    learning_rate_config: LearningRateConfig
    model_type: str
    encoder_config: EncoderConfig
    eval_batch_size: int = 128
    train_batch_size: int = 128


def get_model_selection(model_type, batch_size, resnet_lr, vit_lr):
    return {
        EncoderNames.CLIPViTBase16_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.CLIPViTBase16_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.LaionViTBase16_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.LaionViTBase16_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.SamViTBase16_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.SamViTBase16_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.AugRegViTBase16_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.AugRegViTBase16_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.DiNoViTBase16_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.DiNoViTBase16_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.DeiT3BasePatch16_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.DeiT3BasePatch16_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.FlexViTBase_1200EP.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.FlexViTBase_1200EP,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.IJEPAViTHugePatch14_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.IJEPAViTHugePatch14_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.SIGLIPPathch16_224.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.SIGLIPPathch16_224,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.EfficientFormer_s0.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.EfficientFormer_s0,
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.ResNet50A1.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.ResNet50A1,
            learning_rate_config=LearningRateConfig(
                default=[resnet_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.EffNetV2_RW_S_RA2.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.EffNetV2_RW_S_RA2,
            learning_rate_config=LearningRateConfig(
                default=[resnet_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.ResNeXt50_32x4dA1.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.ResNeXt50_32x4dA1,
            learning_rate_config=LearningRateConfig(
                default=[resnet_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        EncoderNames.ConvNextV2_Base.value.pretty_name: ModelConfig(
            model_type=model_type,
            encoder_config=EncoderNames.ConvNextV2_Base,
            learning_rate_config=LearningRateConfig(
                default=[resnet_lr], dataset_specific={}
            ),
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
    }
