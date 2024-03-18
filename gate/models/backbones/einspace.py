import math
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from einops import rearrange, reduce
from einops.layers.torch import Reduce
from gate.boilerplate.decorators import configurable
from gate.data import image
from gate.models.backbones import (
    GATEImageEncoder,
    GATEImageTextEncoder,
    GATETextEncoder,
)
from gate.models.backbones.timm import CLIPModelPaths, GATECLIPTextEncoder
from rich import print

from einspace.compiler import Compiler
from einspace.search_spaces import EinSpace
from einspace.utils import millify


class ImageEincoder(GATEImageEncoder):
    def __init__(
        self,
        architecture_dict: dict,
        full_image_shape: list | tuple,
        num_projection_features: Optional[int] = None,
    ):
        super(ImageEincoder, self).__init__()
        self.backbone = Compiler().compile(architecture_dict)
        self.full_image_shape = full_image_shape  # (c, h, w)
        self._num_projection_features = num_projection_features

        self.build()

    def build(self):
        x_dummy = torch.zeros((1, *self.full_image_shape))
        out = x_dummy
        out = self.backbone(out)
        backbone_output_shape = out.shape
        self.backbone_output_shape = backbone_output_shape
        if self.num_projection_features is not None:
            if len(backbone_output_shape) == 3:
                self.head = nn.Sequential(
                    Reduce("b s d -> b s", "mean"),
                    nn.Linear(
                        backbone_output_shape[1], self.num_projection_features
                    ),
                )
            elif len(backbone_output_shape) == 4:
                self.head = nn.Sequential(
                    Reduce("b c h w -> b c", "mean"),
                    nn.Linear(
                        backbone_output_shape[1], self.num_projection_features
                    ),
                )
            out = self.head(out)

        print(
            f"Built Eincoder with output shape: {out.shape}, "
            f"backbone output shape: {backbone_output_shape}, "
            f"and backbone: {self.backbone}, num_projection_features: {self.num_projection_features}"
        )

    def forward(self, x):
        out = self.backbone(x)

        return {
            "features": out,
            "raw_features": (
                reduce(out, "b c h w -> b c", "mean").unsqueeze(1)
                if len(out.shape) == 4
                else out
            ),
        }

    @property
    def projection_layer(self):
        return self.head if self.num_projection_features is not None else None

    @property
    def num_projection_features(self):
        return self._num_projection_features

    @property
    def num_features(self):
        return self.backbone_output_shape[1]

    @property
    def num_raw_features(self):
        return self.backbone_output_shape[1]

    @property
    def image_shape(self):
        return self.full_image_shape[1:]

    def transforms(self, x):
        transform_list = [
            transforms.Resize(size=self.image_shape[1:]),
            transforms.ToTensor(),
        ]
        transform_fused = transforms.Compose(transform_list)
        return transform_fused(x)


def represent_dict_order(dumper, data):
    return dumper.represent_dict(data.items())


def construct_ordered_dict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


def fancy_yaml_dump(data, filename):
    yaml.add_representer(OrderedDict, represent_dict_order)
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def fancy_yaml_load(filename):
    # Add the custom constructor for mapping nodes (YAML dictionaries)
    yaml.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_ordered_dict
    )

    # Load the YAML string into an OrderedDict
    loaded_dict = yaml.load(filename, Loader=yaml.Loader)
    return loaded_dict


@configurable(
    group="encoder",
    name="eincoder",
)
class ImageTextEincoder(GATEImageTextEncoder, nn.Module):
    def __init__(
        self,
        architecture_dict: Optional[Any] = None,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        image_size: Optional[int] = 224,
        num_projection_features: Optional[int] = 512,
    ):
        nn.Module.__init__(self)
        if architecture_dict is None:
            einspace = EinSpace(
                input_shape=(3, image_size, image_size),
                input_mode="im",
                num_repeated_cells=1,
                device=torch.device("cpu"),
                computation_module_prob=0.5,
            )
            architecture_dict = einspace.sample()
            print(f"Sampled architecture: {architecture_dict}")

            fancy_yaml_dump(architecture_dict, "eincoder_architecture.yaml")

            architecture_dict = fancy_yaml_load(
                open("eincoder_architecture.yaml", "r")
            )
            print(f"Loaded architecture: {architecture_dict}")
        else:
            if architecture_dict.endswith(".yaml"):
                # Load the YAML file into a dictionary
                print(f"Loading architecture from {architecture_dict}")
                with open(architecture_dict, "r") as file:
                    architecture_dict = fancy_yaml_load(file)
                # Update the configuration with the loaded dictionary
        print(architecture_dict)
        image_embedding = ImageEincoder(
            architecture_dict=architecture_dict,
            full_image_shape=(3, image_size, image_size),
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
