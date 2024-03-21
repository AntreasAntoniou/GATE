import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from gate.models.task_adapters import BaseAdapterModule

# Possibilities:

# 1. Image to text
# 2. Text to image
# 3. Image to audio
# 4. Audio to image
# 5. Image to video
# 6. Video to image
# 7. Text to audio
# 8. Audio to text

# First let's define some base stem models that we can use for the different modalities.
# Root and exit layers are the same for all modalities, so we can define them here.

# Implement a way to transfer CLIP text to image, and its image to text, and then to text to audio, and image to audio
# This will serve as a small study for the GATE paper.


class BaseVisionRootLayer(nn.Module):
    # Adapted from the CLIPVisionEmbeddings class in the huggingface transformers library.
    def __init__(self, embed_dim, image_size, num_channels, patch_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(
            self.num_positions, self.embed_dim
        )
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1))
        )

    def forward(self, image: torch.FloatTensor) -> torch.Tensor:
        batch_size = image.shape[0]
        patch_embeds = self.patch_embedding(
            image
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class BaseTextRootLayer(nn.Module):
    # Adapted from the CLIPTextEmbeddings class in the huggingface transformers library.
    def __init__(self, embed_dim, vocab_size, max_position_embeddings):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(
        self,
        text: torch.LongTensor,
    ) -> torch.Tensor:
        seq_length = text.shape[-1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.token_embedding(text)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class BaseAudioRootLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_mel_bins,
        pad_token_id,
        max_source_positions,
        scale_embedding,
        dropout,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.padding_idx = pad_token_id
        self.max_source_positions = max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if scale_embedding else 1.0
        self.dropout = dropout

        self.conv1 = nn.Conv1d(
            self.num_mel_bins, embed_dim, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )

        self.embed_positions = nn.Embedding(
            self.max_source_positions, embed_dim
        )

    def forward(
        self,
        audio: torch.FloatTensor,
    ):
        inputs_embeds = nn.functional.gelu(self.conv1(audio))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        return hidden_states


MODALITY_ROOT_LAYERS = {
    "image": BaseVisionRootLayer,
    "text": BaseTextRootLayer,
    "audio": BaseAudioRootLayer,
}


class InputAgnosticIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        # return the first args or kwargs that is not None
        for arg in args:
            if arg is not None:
                return arg.float()

        for k, v in kwargs.items():
            if v is not None:
                return v.float()


class VisionRootReplacedBackbone(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_root_features: int,
        image_size: int,
        num_channels: int,
        patch_size: int,
        backbone_root_layers_to_remove: List[str],
        source_modality: str,
        target_modality: str,
    ):
        super().__init__()
        self.model = model

        for layer in backbone_root_layers_to_remove:
            setattr(self.model, layer, InputAgnosticIdentity())

        self.source_modality = source_modality
        self.target_modality = target_modality
        self.num_root_features = num_root_features
        self.root_layer = BaseVisionRootLayer(
            num_root_features, image_size, num_channels, patch_size
        )

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_dict is not None:
            x = self.root_layer(**input_dict)
            x = self.model(x)[self.target_modality]

        if image is not None:
            x = self.root_layer(image=image)
            x = self.model(image=x)

        if text is not None:
            x = self.root_layer(text=text)
            x = self.model(text=x)

        return x
