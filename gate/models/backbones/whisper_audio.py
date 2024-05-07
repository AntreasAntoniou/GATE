import logging
import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers.models.whisper.modeling_whisper import (
    WhisperConfig,
    WhisperEncoder,
    WhisperPreTrainedModel,
)

from gate.boilerplate.decorators import configurable
from gate.models.backbones import (
    GATEImageEncoder,
    GATEImageTextEncoder,
    Modality,
)
from gate.models.backbones.timm import GATECLIPTextEncoder
from gate.models.adapters.utils.modality_transfer import (
    VisionRootReplacedBackbone,
)

logger = logging.getLogger(__name__)


def find_nearest_n(x: Union[int, float], target: Union[int, float]) -> int:
    """
    Function to find the `n` that makes `(x / n) ** 2` closest to the target.

    Args:
        x (Union[int, float]): The input value.
        target (Union[int, float]): The target value to get as close as possible to.

    Returns:
        int: The value of `n` that makes `(x / n) ** 2` closest to the target.
    """
    # Calculate the ideal 'n' using reverse operation
    n_ideal = x / math.sqrt(target)

    # Calculate outputs for floor and ceiling of ideal 'n'
    out_floor = (x / math.floor(n_ideal)) ** 2
    out_ceil = (x / math.ceil(n_ideal)) ** 2

    # Calculate the differences from the target
    diff_floor = abs(target - out_floor)
    diff_ceil = abs(target - out_ceil)

    # Return the 'n' that gives the output closest to the target
    return (
        math.floor(n_ideal) if diff_floor < diff_ceil else math.ceil(n_ideal)
    )


class ModifiedWhisperModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = self.config.output_attentions

        self.config.use_cache
        return_dict = self.config.use_return_dict
        image = image.permute(0, 2, 1)

        if self.encoder.embed_positions.weight.view(-1).shape[0] > 1:
            self.encoder.embed_positions.weight = nn.Parameter(
                torch.zeros(1)
            ).to(image.device)

        if image.shape[1] != 3000:
            logger.debug(f"Resizing image from {image.shape} to (3000)")
            image = torch.nn.functional.interpolate(
                image, size=(3000), mode="nearest"
            )

        encoder_outputs = self.encoder(
            image,
            head_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        return {
            "features": encoder_outputs.last_hidden_state.mean(dim=1),
            "raw_features": encoder_outputs.last_hidden_state,
            "per_layer_raw_features": [encoder_outputs.hidden_states[-1]],
        }


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


class WhisperModelPaths:
    base: str = "openai/whisper-base"
    small: str = "openai/whisper-small"


class GATEWhisperImageEncoder(GATEImageEncoder):
    def __init__(
        self,
        model_name: str = WhisperModelPaths.base,
        pretrained: bool = True,
        image_size: int = 224,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.image_size = image_size if image_size is not None else 224

        vision_embedding = ModifiedWhisperModel.from_pretrained(model_name)
        patch_size = find_nearest_n(image_size, 3000)

        self.vision_model = VisionRootReplacedBackbone(
            model=vision_embedding,
            num_root_features=80,
            backbone_root_layers_to_remove=["embeddings"],
            image_size=image_size,
            num_channels=3,
            patch_size=patch_size,
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
    name="whisper",
)
class WhisperCLIPEncoder(GATEImageTextEncoder, nn.Module):
    def __init__(
        self,
        whisper_model_name: str = WhisperModelPaths.small,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        pretrained: bool = True,
        image_size: Optional[int] = 224,
        num_projection_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        image_embedding = GATEWhisperImageEncoder(
            model_name=whisper_model_name,
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
