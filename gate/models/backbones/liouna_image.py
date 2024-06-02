from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from sympy import per
from torchvision import transforms as T
from transformers import CLIPModel, CLIPProcessor

from gate.boilerplate.decorators import configurable
from gate.models.backbones import (
    GATEImageEncoder,
    GATEImageTextEncoder,
    GATETextEncoder,
    TextProcessor,
)


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


def reinit(input_module: nn.Module):
    for name, module in input_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                module.weight.data, nonlinearity="relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, **kwargs
    ) -> None:
        super().__init__()
        # to preserve the image resolution
        padding, stride = (kernel_size - 1) // 2, 1

        self.norm = nn.BatchNorm2d([in_channels], affine=False)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(2)

    def children(self):
        # Return layers including conv, norm, act, and pooling layer (if exists)
        return [self.norm, self.conv, self.act, self.pool]

    def forward(self, x):
        return nn.Sequential(*self.children())(x)


class PreTrainedLiouna(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        in_channels = in_channels
        plane_sizes = [64, 128, 256, 512, 1024, 2048]
        kernel_sizes = [3] * 6
        self.image_size = image_size

        for i, out_channels in enumerate(plane_sizes):
            self.add_module(
                "block_{}".format(i),
                Block(in_channels, plane_sizes[i], kernel_sizes[i]),
            )
            in_channels = out_channels

        self._transforms = T.Compose(
            [
                T.Resize(
                    size=(self.image_size, self.image_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
            ]
        )

    def forward(self, x):

        per_layer_raw_features = []

        for block in self.children():
            shape = (
                x.shape
                if isinstance(x, torch.Tensor)
                else [item.shape for item in x]
            )
            x = block(x)

            current_feature = nn.functional.interpolate(
                x,
                size=(
                    64,
                    64,
                ),
                mode="bilinear",
            )

            per_layer_raw_features.append(current_feature)
        per_layer_raw_features = torch.cat(per_layer_raw_features, dim=1)
        per_layer_raw_features = per_layer_raw_features.permute(
            0, 2, 3, 1
        ).reshape(
            per_layer_raw_features.shape[0],
            -1,
            per_layer_raw_features.shape[1],
        )
        per_layer_raw_features = [per_layer_raw_features]
        raw_features = x.permute(0, 2, 3, 1).reshape(
            x.shape[0], -1, x.shape[1]
        )
        features = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        # print(
        #     raw_features.shape, features.shape, per_layer_raw_features[0].shape
        # )

        return {
            "classifier": features,
            "features": features,
            "raw_features": raw_features,
            "per_layer_raw_features": per_layer_raw_features,
        }

    def init_weights(self):
        reinit(self)

    def transforms(self, x):
        return self._transforms(x)


class LiounaModelPaths:
    liouna_base = "FadyRezk/Liouna"
    softhebb_base = "FadyRezk/SoftHebb"
    simclr_base = "FadyRezk/SimCLR"


class GATELiounaImageEncoder(GATEImageEncoder):
    def __init__(
        self,
        model_name: str = LiounaModelPaths.liouna_base,
        pretrained: bool = True,
        image_size: int = 224,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.image_size = image_size if image_size is not None else 224

        self.vision_model = PreTrainedLiouna.from_pretrained(
            model_name, image_size=image_size
        )

        if not pretrained:
            self.vision_model.init_weights()

        output_shape = self.get_output_shape()
        self.image_num_raw_features = output_shape["raw_features"][-1]

        self.image_num_features = (
            output_shape["features"][-1]
            if num_projection_features is None
            else num_projection_features
        )
        self._num_projection_features = num_projection_features

        self.visual_projection = (
            nn.Linear(
                output_shape["features"][-1],
                num_projection_features,
                bias=False,
            )
            if num_projection_features is not None
            else nn.Identity()
        )

    def get_output_shape(self):
        output_dict = self.vision_model(
            torch.randn(1, 3, self.image_size, self.image_size)
        )

        return {
            key: (
                value.shape
                if isinstance(value, torch.Tensor)
                else [sub_value.shape for sub_value in value]
            )
            for key, value in output_dict.items()
        }

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
        return self.vision_model(x)

    def transforms(self, x):
        return self.vision_model.transforms(x)


class GATECLIPTextEncoder(GATETextEncoder):
    def __init__(
        self,
        model_name: str,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )

        self.clip = CLIPModel.from_pretrained(model_name)
        self.text_transforms = TextProcessor(self.preprocessor)

        self.text_model = self.clip.text_model

        self.text_num_features = (
            self.clip.text_embed_dim
            if num_projection_features is None
            else num_projection_features
        )
        self._num_projection_features = num_projection_features

        self.text_model = self.clip.text_model
        self.text_projection = (
            nn.Linear(
                self.text_model.config.hidden_size,
                num_projection_features,
            )
            if num_projection_features is not None
            else nn.Identity()
        )

        self.text_num_raw_features = self.text_model.config.hidden_size

    @property
    def projection_layer(self):
        return self.text_projection

    @property
    def num_projection_features(self):
        return self._num_projection_features

    @property
    def num_features(self):
        return self.text_num_features

    @property
    def num_raw_features(self):
        return self.text_num_raw_features

    def forward(self, input_ids: torch.Tensor):
        return self.text_model(input_ids=input_ids)

    def transforms(self, x):
        return self.text_transforms.apply_transform(x)


@configurable(
    group="encoder",
    name="liouna",
)
class LiounaCLIPEncoder(GATEImageTextEncoder, nn.Module):
    def __init__(
        self,
        liouna_model_name: str = LiounaModelPaths.liouna_base,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        pretrained: bool = True,
        image_size: Optional[int] = 224,
        num_projection_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        image_embedding = GATELiounaImageEncoder(
            model_name=liouna_model_name,
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


if __name__ == "__main__":
    model = PreTrainedLiouna.from_pretrained("FadyRezk/Liouna")
    print(model.block_0.conv.weight[0][0])
    print(model)
    print(list(model(torch.randn(1, 3, 224, 224)).keys()))

    gate_model = GATELiounaImageEncoder(num_projection_features=512)

    out = gate_model(torch.randn(1, 3, 224, 224))

    print(gate_model)
    print(list(out.keys()))

    gate_liouna_clip = LiounaCLIPEncoder()
