import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)

from gate.models.backbones import (
    Modality,
    TextProcessor,
    VisionTextGATEAdapter,
    forward_dict,
)
from gate.models.core import reinit
from gate.models.task_adapters.modality_transfer_classification import (
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
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output.mean(dim=1)

        return {
            "features": pooled_output,
            "raw_features": sequence_output,
            "per_layer_raw_features": encoder_outputs.hidden_states,
        }


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


class BertModelPaths:
    base_uncased: str = "bert-base-uncased"


class BertAdapter(VisionTextGATEAdapter, nn.Module):
    def __init__(
        self,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        bert_model_name: str = BertModelPaths.base_uncased,
        pretrained: bool = True,
        image_size: Optional[int] = None,
    ):
        VisionTextGATEAdapter.__init__(self)
        nn.Module.__init__(self)

        self.vision_preprocessor: CLIPProcessor = (
            CLIPProcessor.from_pretrained(clip_model_name)
        )
        self.text_preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            clip_model_name
        )
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.text_transforms = TextProcessor(self.text_preprocessor)

        if not pretrained:
            self.clip.init_weights()

        vision_embedding = ModifiedBertEncoder.from_pretrained(bert_model_name)

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
        self.visual_projection = nn.Linear(
            vision_embedding.config.hidden_size,
            self.clip.text_embed_dim,
            bias=False,
        )
        self.text_model = self.clip.text_model
        self.text_projection = self.clip.text_projection

        # setattr signature: setattr(object, name, value)

        setattr(self.vision_model, "legacy_forward", self.text_model.forward)
        setattr(self.text_model, "legacy_forward", self.text_model.forward)

        setattr(
            self.text_model, "forward", forward_dict.__get__(self.text_model)
        )

        self.image_num_features = self.clip.vision_embed_dim
        self.text_num_features = self.clip.text_embed_dim

    def init_weights(self):
        reinit(self)
