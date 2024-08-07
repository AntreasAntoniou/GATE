from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from rich import print

from gate.data import image
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
from gate.models.backbones.bart_text import BartModelPaths
from gate.models.backbones.bert_text import BertModelPaths
from gate.models.backbones.mpnet_text import MPNetModelPaths
from gate.models.backbones.timm import CLIPModelPaths
from gate.models.backbones.wave2vec_audio import Wav2Vec2ModelPaths
from gate.models.backbones.whisper_audio import WhisperModelPaths
from gate.models.task_adapters.semantic_segmentation import (
    SegmentationLossOptions,
)
from gate.models.task_adapters.temporal_image_classification import Metrics


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


class EncoderConfig:
    def __init__(
        self,
        pretty_name: str,
        image_size: int = 224,
        pretrained: bool = True,
        encoder_name: Optional[str] = None,
        model_name: Optional[str] = None,
        timm_model_name: Optional[str] = None,
        clip_model_name: Optional[str] = None,
        bart_model_name: Optional[str] = None,
        bert_model_name: Optional[str] = None,
        wav2vec2_model_name: Optional[str] = None,
        whisper_model_name: Optional[str] = None,
        mpnet_model_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        num_projection_features: Optional[int] = 768,
        freeze_encoder: bool = False,
    ):
        self.pretty_name = pretty_name
        self.image_size = image_size
        self.pretrained = pretrained
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.timm_model_name = timm_model_name
        self.clip_model_name = clip_model_name
        self.bart_model_name = bart_model_name
        self.bert_model_name = bert_model_name
        self.wav2vec2_model_name = wav2vec2_model_name
        self.whisper_model_name = whisper_model_name
        self.mpnet_model_name = mpnet_model_name
        self.embedding_dim = embedding_dim
        self.num_projection_features = num_projection_features
        self.freeze_encoder = freeze_encoder

    def __call__(self, image_size: int):
        self.image_size = image_size
        return self


@dataclass
class AdapterConfig:
    adapter_name: str
    use_stem_instance_norm: bool = True
    metric_type: Optional[str] = None
    loss_type_id: Optional[str] = None
    background_loss_weight: Optional[float] = None
    dice_loss_weight: Optional[float] = None
    focal_loss_weight: Optional[float] = None
    ce_loss_weight: Optional[float] = None
    freeze_encoder: bool = False


class Adapters(Enum):
    IMAGE_CLASSIFICATION = AdapterConfig(
        adapter_name="backbone-with-linear-single-classifier"
    )
    MULTI_CLASS_CLASSIFICATION = AdapterConfig(
        adapter_name="backbone-with-linear-multi-classifier"
    )
    FEW_SHOT_PROTONET = AdapterConfig(adapter_name="fs-protonet")
    SEGMENTATION = AdapterConfig(
        adapter_name="segmentation-adapter",
        loss_type_id=SegmentationLossOptions.DEFAULT.value,
        background_loss_weight=0.01,
        dice_loss_weight=1.0,
        focal_loss_weight=1.0,
        ce_loss_weight=1.0,
    )
    MD_SEGMENTATION = AdapterConfig(
        adapter_name="segmentation-adapter",
        loss_type_id=SegmentationLossOptions.MD.value,
        background_loss_weight=0.01,
    )
    TEMPORAL_CLASSIFICATION = AdapterConfig(
        adapter_name="temporal-classification",
        metric_type=Metrics.CLASSIFICATION,
    )
    TEMPORAL_REGRESSION = AdapterConfig(
        adapter_name="temporal-classification", metric_type=Metrics.REGRESSION
    )
    RELATIONAL_REASONING = AdapterConfig(
        adapter_name="relational-reasoning",
    )
    RELATIONAL_REASONING_MULTI_TASK = AdapterConfig(
        adapter_name="relational-reasoning",
    )

    ZERO_SHOT_IMAGE_TEXT = AdapterConfig(
        adapter_name="duo-modal-zero-shot-classifier",
    )


# Create an Enum to store EncoderConfigs
class Encoders(Enum):
    # CLIPViTBase16_224HF_IMAGE = EncoderConfig(
    #     pretty_name="CLIP_B16_224HF_image",
    #     model_name=CLIPModelPaths.openai_b_16,
    #     encoder_name="clip-image",
    #     num_projection_features=768,
    # )
    # CLIPViTBase16_224HF_TEXT = EncoderConfig(
    #     pretty_name="CLIP_B16_224HF_text",
    #     model_name=CLIPModelPaths.openai_b_16,
    #     encoder_name="clip-text",
    #     num_projection_features=768,
    #     freeze_encoder=True,
    # )
    BART_TEXT = EncoderConfig(
        pretty_name="BART",
        bart_model_name=BartModelPaths.base,
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="bart",
        num_projection_features=768,
        freeze_encoder=True,
    )
    BERT_TEXT = EncoderConfig(
        pretty_name="BERT",
        bert_model_name=BertModelPaths.base_uncased,
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="bert",
        num_projection_features=768,
        freeze_encoder=True,
    )
    MPNet = EncoderConfig(
        pretty_name="MPNET",
        mpnet_model_name=MPNetModelPaths.base,
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="mpnet",
        num_projection_features=768,
        freeze_encoder=True,
    )
    # Wave2VecV2Base = EncoderConfig(
    #     pretty_name="W2V2",
    #     encoder_name="wav2vecv2",
    #     wav2vec2_model_name=Wav2Vec2ModelPaths.base,
    #     clip_model_name=CLIPModelPaths.openai_b_16,
    #     num_projection_features=768,
    #     freeze_encoder=True,
    # )
    WhisperBase = EncoderConfig(
        pretty_name="Whisper",
        encoder_name="whisper",
        whisper_model_name=WhisperModelPaths.base,
        clip_model_name=CLIPModelPaths.openai_b_16,
        num_projection_features=768,
        freeze_encoder=True,
    )
    ResNet50A1 = EncoderConfig(
        pretty_name="R50A1",
        timm_model_name="resnet50.a1_in1k",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    ResNeXt50_32x4dA1 = EncoderConfig(
        pretty_name="RNX50_32x4A1",
        timm_model_name="resnext50_32x4d.a1_in1k",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    SamViTBase16_224 = EncoderConfig(
        pretty_name="SViT_B16_224",
        timm_model_name="vit_base_patch16_224.sam_in1k",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    AugRegViTBase16_224 = EncoderConfig(
        pretty_name="AR_ViT_B16_224",
        timm_model_name="vit_base_patch16_224.augreg_in1k",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    DiNoViTBase16_224 = EncoderConfig(
        pretty_name="DINO_B16_224",
        timm_model_name="vit_base_patch16_224.dino",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    CLIPViTBase16_224 = EncoderConfig(
        pretty_name="CLIP_B16_224",
        timm_model_name="vit_base_patch16_clip_224",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    LaionViTBase16_224 = EncoderConfig(
        pretty_name="Laion_B16_224",
        timm_model_name="vit_base_patch16_clip_224.laion2b",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    EfficientFormer_s0 = EncoderConfig(
        pretty_name="EffFormer_s0",
        timm_model_name="efficientformerv2_s0",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    EffNetV2_RW_S_RA2 = EncoderConfig(
        pretty_name="EffV2_RW_S",
        timm_model_name="efficientnetv2_rw_s.ra2_in1k",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    ConvNextV2_Base = EncoderConfig(
        pretty_name="ConvNextV2_Base",
        timm_model_name="convnextv2_base",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    DeiT3BasePatch16_224 = EncoderConfig(
        pretty_name="DeiT3_B16_224",
        timm_model_name="deit3_base_patch16_224.fb_in1k",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    FlexViTBase_1200EP = EncoderConfig(
        pretty_name="Flex_B_1200EP",
        timm_model_name="flexivit_base.1200ep_in1k",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )
    # IJEPAViTHugePatch14_224 = EncoderConfig(
    #     pretty_name="IJEPA_Huge_P14_224",
    #     timm_model_name="vit_huge_patch14_gap_224.in22k_ijepa",
    #     clip_model_name=CLIPModelPaths.openai_b_16,
    #     encoder_name="timm",
    #     num_projection_features=768,
    # )
    SIGLIPPathch16_224 = EncoderConfig(
        pretty_name="SIGLIP_P16_224",
        timm_model_name="vit_base_patch16_siglip_224",
        clip_model_name=CLIPModelPaths.openai_b_16,
        encoder_name="timm",
        num_projection_features=768,
    )


@dataclass
class MixedPrecisionMode:
    BF16 = "bf16"
    FP16 = "fp16"


@dataclass
class ModelConfig:
    learning_rate_config: LearningRateConfig
    adapter_config: AdapterConfig
    encoder_config: EncoderConfig
    eval_batch_size: int = 128
    train_batch_size: int = 128
    weight_decay: float = 0.01
    mixed_precision_mode: str = MixedPrecisionMode.BF16


def get_model_selection(
    adapter_config,
    batch_size,
    resnet_lr,
    vit_lr,
    wd: float,
    image_size: int,
    mixed_precision_mode: str = MixedPrecisionMode.BF16,
):
    encoder_menu = deepcopy(Encoders)

    output_dict = {
        # encoder_menu.Wave2VecV2Base.value.pretty_name: ModelConfig(
        #     adapter_config=adapter_config,
        #     encoder_config=deepcopy(encoder_menu.Wave2VecV2Base.value)(
        #         image_size=image_size
        #     ),
        #     learning_rate_config=LearningRateConfig(
        #         default=[vit_lr], dataset_specific={}
        #     ),
        #     weight_decay=wd,
        #     train_batch_size=batch_size,
        #     eval_batch_size=batch_size,
        # ),
        encoder_menu.WhisperBase.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.WhisperBase.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.MPNet.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.MPNet.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.BERT_TEXT.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.BERT_TEXT.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.BART_TEXT.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.BART_TEXT.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        # encoder_menu.CLIPViTBase16_224HF_IMAGE.value.pretty_name: ModelConfig(
        #     adapter_config=adapter_config,
        #     encoder_config=deepcopy(
        #         encoder_menu.CLIPViTBase16_224HF_IMAGE.value
        #     )(image_size=image_size),
        #     learning_rate_config=LearningRateConfig(
        #         default=[vit_lr], dataset_specific={}
        #     ),
        #     weight_decay=wd,
        #     train_batch_size=batch_size,
        #     eval_batch_size=batch_size,
        # ),
        # encoder_menu.CLIPViTBase16_224HF_TEXT.value.pretty_name: ModelConfig(
        #     adapter_config=adapter_config,
        #     encoder_config=deepcopy(
        #         encoder_menu.CLIPViTBase16_224HF_TEXT.value
        #     )(image_size=image_size),
        #     learning_rate_config=LearningRateConfig(
        #         default=[vit_lr], dataset_specific={}
        #     ),
        #     weight_decay=wd,
        #     train_batch_size=batch_size,
        #     eval_batch_size=batch_size,
        # ),
        encoder_menu.AugRegViTBase16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.AugRegViTBase16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.LaionViTBase16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.LaionViTBase16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.CLIPViTBase16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.CLIPViTBase16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.SamViTBase16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.SamViTBase16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.AugRegViTBase16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.AugRegViTBase16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.DiNoViTBase16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.DiNoViTBase16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.DeiT3BasePatch16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.DeiT3BasePatch16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.FlexViTBase_1200EP.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.FlexViTBase_1200EP.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        # encoder_menu.IJEPAViTHugePatch14_224.value.pretty_name: ModelConfig(
        #     adapter_config=adapter_config,
        #     encoder_config=deepcopy(
        #         encoder_menu.IJEPAViTHugePatch14_224.value
        #     )(image_size=image_size),
        #     learning_rate_config=LearningRateConfig(
        #         default=[vit_lr], dataset_specific={}
        #     ),
        #     weight_decay=wd,
        #     train_batch_size=batch_size,
        #     eval_batch_size=batch_size,
        # ),
        encoder_menu.SIGLIPPathch16_224.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.SIGLIPPathch16_224.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            mixed_precision_mode=mixed_precision_mode,
        ),
        encoder_menu.EfficientFormer_s0.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.EfficientFormer_s0.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.EffNetV2_RW_S_RA2.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.EffNetV2_RW_S_RA2.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[resnet_lr], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.ResNeXt50_32x4dA1.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.ResNeXt50_32x4dA1.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr * 2], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
        encoder_menu.ConvNextV2_Base.value.pretty_name: ModelConfig(
            adapter_config=adapter_config,
            encoder_config=deepcopy(encoder_menu.ConvNextV2_Base.value)(
                image_size=image_size
            ),
            learning_rate_config=LearningRateConfig(
                default=[vit_lr * 2], dataset_specific={}
            ),
            weight_decay=wd,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
        ),
    }
    return output_dict
