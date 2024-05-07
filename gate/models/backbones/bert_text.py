import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
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


class ModifiedBertEncoder(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output_attentions = self.config.output_attentions

        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = self.config.use_cache

        else:
            use_cache = False

        encoder_outputs = self.encoder(
            image,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output.mean(dim=1)

        return {
            "features": pooled_output,
            "raw_features": sequence_output,
            "per_layer_raw_features": encoder_outputs.hidden_states,
        }


class BertModelPaths:
    base_uncased: str = "bert-base-uncased"


class GATEBERTImageEncoder(GATEImageEncoder):
    def __init__(
        self,
        model_name: str = BertModelPaths.base_uncased,
        pretrained: bool = True,
        image_size: int = 224,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.image_size = image_size if image_size is not None else 224

        vision_embedding = ModifiedBertEncoder.from_pretrained(
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
    name="bert",
)
class BERTCLIPEncoder(GATEImageTextEncoder, nn.Module):
    def __init__(
        self,
        bert_model_name: str = BertModelPaths.base_uncased,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        pretrained: bool = True,
        image_size: Optional[int] = 224,
        num_projection_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        image_embedding = GATEBERTImageEncoder(
            model_name=bert_model_name,
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
