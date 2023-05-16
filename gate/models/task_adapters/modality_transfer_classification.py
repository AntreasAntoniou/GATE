import math
from typing import Dict, List, Optional

import torch.nn as nn
import torch

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


class RootReplacedBackboneWithLinearExitLayer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_root_features: int,
        num_backbone_features: int,
        num_classes: int,
        backbone_root_layers_to_remove: List[str],
        root_layer_kwargs: Dict,
        source_modality: str,
        target_modality: str,
    ):
        super().__init__()
        self.model = model

        for layer in backbone_root_layers_to_remove:
            setattr(self.model, layer, nn.Identity())

        self.source_modality = source_modality
        self.target_modality = target_modality
        self.num_root_features = num_root_features
        self.root_layer = MODALITY_ROOT_LAYERS[target_modality](
            embed_dim=num_root_features, **root_layer_kwargs
        )
        self.linear = nn.Linear(num_backbone_features, num_classes)

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_dict is not None:
            x = self.model(self.root_layer(**input_dict))[
                self.target_modality
            ]["features"]

        if image is not None:
            x = self.model(image=self.root_layer(image=image))["image"][
                "features"
            ]

        if text is not None:
            x = self.model(text=self.root_layer(text=text))["text"]["features"]

        if audio is not None:
            x = self.model(audio=self.root_layer(audio=audio))["audio"][
                "features"
            ]

        if video is not None:
            x = self.model(video=self.root_layer(video=video))["video"][
                "features"
            ]

        x = self.linear(x)

        return x
