import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
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
    GATEImageEncoder,
    GATEImageTextEncoder,
    Modality,
)
from gate.models.backbones.timm import CLIPModelPaths, GATECLIPTextEncoder
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


class Wav2Vec2ModelPaths:
    base: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


class GATEWav2Vec2ImageEncoder(GATEImageEncoder):
    def __init__(
        self,
        model_name: str = Wav2Vec2ModelPaths.base,
        pretrained: bool = True,
        image_size: int = 224,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.image_size = image_size if image_size is not None else 224

        vision_embedding = ModifiedWav2Vec2Model.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
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

        if not pretrained:
            self.vision_model.init_weights()

        self.image_num_raw_features = vision_embedding.config.hidden_size

        self.image_num_features = (
            self.image_num_raw_features
            if num_projection_features is None
            else num_projection_features
        )
        self._num_projection_features = num_projection_features

        self.visual_projection = (
            nn.Linear(
                self.image_num_raw_features,
                num_projection_features,
                bias=False,
            )
            if num_projection_features is not None
            else nn.Identity()
        )

    @property
    def projection_layer(self):
        return self.visual_projection

    @property
    def num_projection_features(self):
        return self._num_projection_features

    @property
    def num_features(self):
        return self.image_num_features

    @property
    def num_raw_features(self):
        return self.image_num_raw_features

    @property
    def image_shape(self):
        return (self.image_size, self.image_size)

    def forward(self, x):
        return self.vision_model(image=x)

    def transforms(self, x):
        return self.vision_model.transforms(x)


@configurable(
    group="encoder",
    name="wav2vec2",
)
class Wav2vec2CLIPEncoder(GATEImageTextEncoder, nn.Module):
    def __init__(
        self,
        wav2vec2_model_name: str = Wav2Vec2ModelPaths.base,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        pretrained: bool = True,
        image_size: Optional[int] = 224,
        num_projection_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        image_embedding = GATEWav2Vec2ImageEncoder(
            model_name=wav2vec2_model_name,
            pretrained=pretrained,
            image_size=image_size,
            num_projection_features=num_projection_features,
        )
        text_embedding = GATECLIPTextEncoder(
            model_name=clip_model_name,
            num_projection_features=num_projection_features,
        )
        GATEImageTextEncoder.__init__(
            self,
            image_embedding,
            text_embedding,
            image_size,
            num_projection_features,
        )
