import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.mpnet.modeling_mpnet import (MPNetEncoder,
                                                      MPNetPreTrainedModel)

from gate.boilerplate.decorators import configurable
from gate.models.backbones import (GATEncoder, Modality, VisionTextGATEAdapter,
                                   forward_dict)
from gate.models.core import reinit
from gate.models.task_adapters.modality_transfer_classification import \
    VisionRootReplacedBackbone

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def text_transforms(self, x: Union[List[str], List[List[str]]]):
        if isinstance(x[0], list):
            x = [item for sublist in x for item in sublist]
        return self.preprocessor(
            text=x, return_tensors="pt", padding=True, truncation=True
        ).input_ids.squeeze(0)

    def apply_transform(self, text: Union[List[str], List[List[str]]]):
        if not all(
            isinstance(i, list) for i in text
        ):  # if text is list of strings
            text = [text]
        transformed_text = self.text_transforms(text)
        return transformed_text


class ModifiedMPNetModel(MPNetPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.encoder = MPNetEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            image,
            attention_mask=None,
            head_mask=head_mask,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )

        return {
            "features": encoder_outputs.last_hidden_state.mean(dim=1),
            "raw_features": encoder_outputs.last_hidden_state,
            "per_layer_raw_features": encoder_outputs.hidden_states,
        }


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


class MPNetModelPaths:
    base: str = "sentence-transformers/all-mpnet-base-v2"


@configurable(
    group="encoder",
    name="mpnet",
)
class MPNetAdapter(VisionTextGATEAdapter, GATEncoder):
    def __init__(
        self,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        mpnet_model_name: str = MPNetModelPaths.base,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        num_projection_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        VisionTextGATEAdapter.__init__(self)
        self.image_size = image_size
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            clip_model_name
        )

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.text_transforms = TextProcessor(self.preprocessor)

        if not pretrained:
            self.clip.init_weights()

        vision_embedding = ModifiedMPNetModel.from_pretrained(mpnet_model_name)

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
            self.clip.vision_embed_dim
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
