import logging
from typing import Optional

import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models.backbones import (
    GATEncoder,
    TextProcessor,
    VisionTextGATEAdapter,
    forward_dict,
)

logger = logging.getLogger(__name__)


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


@configurable(
    group="encoder",
    name="clip-image",
)
class CLIPVisionAdapter(VisionTextGATEAdapter, GATEncoder):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        VisionTextGATEAdapter.__init__(self)
        self.image_size = image_size
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.clip = CLIPModel.from_pretrained(model_name)
        self.text_transforms = TextProcessor(self.preprocessor)

        if not pretrained:
            self.clip.init_weights()

        self.vision_model = self.clip.vision_model
        self.visual_projection = self.clip.visual_projection
        self.text_model = self.clip.text_model
        self.text_projection = self.clip.text_projection

        # setattr signature: setattr(object, name, value)

        setattr(self.vision_model, "legacy_forward", self.vision_model.forward)
        setattr(self.text_model, "legacy_forward", self.text_model.forward)

        setattr(
            self.vision_model,
            "forward",
            forward_dict.__get__(self.vision_model),
        )
        setattr(
            self.text_model, "forward", forward_dict.__get__(self.text_model)
        )

        if image_size is not None:
            self.modify_expected_image_size(image_size)

        self.image_num_features = self.clip.vision_embed_dim
        self.text_num_features = self.clip.text_embed_dim

    @property
    def image_shape(self):
        return (self.image_size, self.image_size)

    def modify_expected_image_size(self, image_size: int):
        config = self.vision_model.config
        config.image_size = image_size
        updated_embeddings = CLIPVisionEmbeddings(config)
        logger.info(
            f"updating vision transformer embedding config to: {config}"
        )
        self.vision_model.embeddings = updated_embeddings

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

    def get_image_encoder(self):
        return self.vision_model

    def get_text_encoder(self):
        return self.text_model
