import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2PreTrainedModel,
)

from gate.boilerplate.decorators import configurable
from gate.models.backbones import (
    GATEncoder,
    Modality,
    TextProcessor,
    VisionTextGATEAdapter,
    forward_dict,
)
from gate.models.core import reinit
from gate.models.task_adapters.utils.modality_transfer import (
    VisionRootReplacedBackbone,
)

logger = logging.getLogger(__name__)


class ModifiedWav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(
                torch.FloatTensor(config.hidden_size).uniform_()
            )

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output_attentions = self.config.output_attentions

        return_dict = self.config.use_return_dict

        encoder_outputs = self.encoder(
            image,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        return {
            "features": hidden_states.mean(dim=1),
            "raw_features": hidden_states,
            "per_layer_raw_features": encoder_outputs.hidden_states,
        }


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


class Wav2Vec2ModelPaths:
    base: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


@configurable(
    group="encoder",
    name="wav2vecv2",
)
class Wav2VecV2Adapter(VisionTextGATEAdapter, GATEncoder):
    def __init__(
        self,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        wav2vec2_model_name: str = Wav2Vec2ModelPaths.base,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        num_projection_features: Optional[int] = None,
    ):
        VisionTextGATEAdapter.__init__(self)
        nn.Module.__init__(self)
        self.image_size = image_size
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            clip_model_name
        )

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.text_transforms = TextProcessor(self.preprocessor)

        if not pretrained:
            self.clip.init_weights()

        vision_embedding = ModifiedWav2Vec2Model.from_pretrained(
            wav2vec2_model_name
        )

        self.vision_model = VisionRootReplacedBackbone(
            model=vision_embedding,
            num_root_features=1024,
            backbone_root_layers_to_remove=["embeddings"],
            image_size=image_size,
            num_channels=3,
            patch_size=16,
            source_modality=Modality.image,
            target_modality=Modality.image,
        )

        self.visual_projection = (
            nn.Linear(
                vision_embedding.config.hidden_size,
                num_projection_features,
                bias=False,
            )
            if num_projection_features is not None
            else nn.Identity()
        )

        self.text_model = self.clip.text_model
        self.text_projection = (
            nn.Linear(
                self.text_model.config.hidden_size,
                num_projection_features,
            )
            if num_projection_features is not None
            else nn.Identity()
        )

        # setattr signature: setattr(object, name, value)

        setattr(self.vision_model, "legacy_forward", self.text_model.forward)
        setattr(self.text_model, "legacy_forward", self.text_model.forward)

        setattr(
            self.text_model, "forward", forward_dict.__get__(self.text_model)
        )

        self.image_num_features = (
            vision_embedding.config.hidden_size
            if num_projection_features is None
            else num_projection_features
        )
        self.text_num_features = (
            self.clip.text_embed_dim
            if num_projection_features is None
            else num_projection_features
        )

        self.text_num_raw_features = self.text_model.config.hidden_size
        self.image_num_raw_features = vision_embedding.config.hidden_size

    @property
    def image_shape(self):
        return (self.image_size, self.image_size)

    def init_weights(self):
        reinit(self)

    @property
    def num_in_features_image(self):
        return self.image_num_features

    @property
    def num_in_features_text(self):
        return self.text_num_features

    @property
    def num_raw_features_image(self):
        return self.image_num_raw_features

    @property
    def num_raw_features_text(self):
        return self.text_num_raw_features

    @property
    def num_in_features_video(self):
        raise NotImplementedError("BART does not have a video backbone")

    def init_weights(self):
        return super().init_weights()

    def get_transforms(self, image_size: int = 224):
        return super().get_transforms(image_size=image_size)

    def get_image_encoder(self):
        return self.vision_model

    def get_text_encoder(self):
        return self.text_model
