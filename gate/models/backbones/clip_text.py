import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import (
    CLIPEncoder,
    CLIPTextEmbeddings,
    _make_causal_mask,
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


class ModifiedCLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(
            embed_dim, eps=config.layer_norm_eps
        )

    def forward(
        self,
        image: Optional[torch.Tensor],
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = self.config.output_attentions

        output_hidden_states = self.config.output_hidden_states

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        hidden_states = self.embeddings(input_ids=image, position_ids=None)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(
            hidden_states.shape[:2],
            hidden_states.dtype,
            device=hidden_states.device,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state.mean(dim=1)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "features": pooled_output,
            "raw_features": last_hidden_state,
            "per_layer_raw_features": hidden_states,
        }


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


class CLIPTextAdapter(
    VisionTextGATEAdapter,
    nn.Module,
):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        VisionTextGATEAdapter.__init__(self)
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.clip = CLIPModel.from_pretrained(model_name)
        self.text_transforms = TextProcessor(self.preprocessor)

        if not pretrained:
            self.clip.init_weights()

        vision_embedding = ModifiedCLIPTextTransformer(
            self.clip.config.text_config
        )

        self.vision_model = VisionRootReplacedBackbone(
            model=vision_embedding,
            num_root_features=self.clip.text_embed_dim,
            backbone_root_layers_to_remove=["embeddings"],
            image_size=image_size,
            num_channels=3,
            patch_size=16,
            source_modality=Modality.image,
            target_modality=Modality.image,
        )
        self.visual_projection = self.clip.text_projection
        self.text_model = self.clip.text_model
        self.text_projection = self.clip.text_projection

        # setattr signature: setattr(object, name, value)

        setattr(self.vision_model, "legacy_forward", self.text_model.forward)
        setattr(self.text_model, "legacy_forward", self.text_model.forward)

        setattr(
            self.text_model, "forward", forward_dict.__get__(self.text_model)
        )

        self.image_num_features = self.vision_model.model.config.hidden_size
        self.text_num_features = self.clip.text_embed_dim

    @property
    def num_in_features_image(self):
        return self.image_num_features

    @property
    def num_in_features_text(self):
        return self.text_num_features

    @property
    def num_in_features_video(self):
        raise NotImplementedError("CLIP does not have a video backbone")

    def init_weights(self):
        return super().init_weights()

    def get_transforms(self, image_size: int = 224):
        return super().get_transforms(image_size=image_size)
