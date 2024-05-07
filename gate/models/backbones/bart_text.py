import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import BartConfig, BartPreTrainedModel
from transformers.models.bart.modeling_bart import BartEncoder

from gate.boilerplate.decorators import configurable
from gate.models.backbones import (
    GATEImageEncoder,
    GATEImageTextEncoder,
    Modality,
)
from gate.models.backbones.timm import CLIPModelPaths, GATECLIPTextEncoder
from gate.models.adapters.utils.modality_transfer import (
    VisionRootReplacedBackbone,
)

logger = logging.getLogger(__name__)


class ModifiedBartModel(BartPreTrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        encoder_outputs = self.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=image,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )

        return {
            "features": encoder_outputs.last_hidden_state.mean(dim=1),
            "raw_features": encoder_outputs.last_hidden_state,
            "per_layer_raw_features": encoder_outputs.hidden_states,
        }


class BartModelPaths:
    base: str = "facebook/bart-base"


class GATEBARTImageEncoder(GATEImageEncoder):
    def __init__(
        self,
        model_name: str = BartModelPaths.base,
        pretrained: bool = True,
        image_size: int = 224,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.image_size = image_size if image_size is not None else 224

        vision_embedding = ModifiedBartModel.from_pretrained(
            model_name,
            max_position_embeddings=(
                4097
                if image_size == 1024
                else 2049 if image_size == 512 else 1025
            ),
            ignore_mismatched_sizes=True,
        )

        self.vision_model = VisionRootReplacedBackbone(
            model=vision_embedding,
            num_root_features=vision_embedding.config.hidden_size,
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
    name="bart",
)
class BARTCLIPEncoder(GATEImageTextEncoder, nn.Module):
    def __init__(
        self,
        bart_model_name: str = BartModelPaths.base,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        pretrained: bool = True,
        image_size: Optional[int] = 224,
        num_projection_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        image_embedding = GATEBARTImageEncoder(
            model_name=bart_model_name,
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
