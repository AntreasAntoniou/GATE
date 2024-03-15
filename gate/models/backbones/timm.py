import logging
from typing import List, Optional
from urllib.request import urlopen

import PIL
import PIL.Image as Image
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import CLIPModel, CLIPProcessor

from gate.boilerplate.decorators import configurable
from gate.models.backbones import (
    GATEImageEncoder,
    GATEImageTextEncoder,
    GATETextEncoder,
    Modality,
    TextProcessor,
    image_dim_reshape,
)

logger = logging.getLogger(__name__)

single_to_three_channel = T.Lambda(lambda x: x.repeat(3, 1, 1))


def apply_preprocessing_transforms(transforms, x, modality=Modality.image):
    input_shape = None
    is_5d_tensor = False

    if isinstance(x, PIL.Image.Image) and modality == Modality.image:
        x = x.convert("RGB")

    if isinstance(x, PIL.Image.Image) and modality == Modality.image:
        x = T.ToTensor()(x)
        if x.shape[0] == 1:
            x = single_to_three_channel(x)
        x = T.ToPILImage()(x)

    if isinstance(x, torch.Tensor) and modality == Modality.image:
        input_shape = x.shape
        x = image_dim_reshape(x)
        is_5d_tensor = len(x.shape) == 5

    if transforms is not None:
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 5:
                x = x.view(-1, *x.shape[2:])  # flatten batch and sequence dims
                x = [T.ToPILImage()(item) for item in x]
            elif len(x.shape) == 4:
                x = [T.ToPILImage()(item) for item in x]
            else:
                x = T.ToPILImage()(x)

        if isinstance(x, List) and Modality.image == modality:
            x = [transforms(item) for item in x]
            x = torch.stack(x)
        elif isinstance(x, List) and Modality.text == modality:
            x = transforms(x)
        else:
            x = transforms(x)

    if (
        input_shape is not None
        and isinstance(x, torch.Tensor)
        and is_5d_tensor
    ):
        x = x.view(input_shape[0], input_shape[1], *x.shape[1:])

    return x


class TimmModel(nn.Module):
    def __init__(
        self,
        model_identifier: str = "hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k",
        image_size: Optional[int] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        try:
            self.model = timm.create_model(
                model_name=model_identifier,
                pretrained=pretrained,
                features_only=True,
            )

        except RuntimeError as e:
            logger.info(
                f"Could not load model {model_identifier} because {e}, trying to load as vision transformer"
            )
            logger.info(
                f"model_identifier: {model_identifier}, pretrained: {pretrained}, img_size: {image_size}"
            )
            self.model = timm.create_model(
                model_name=model_identifier,
                img_size=image_size,
                pretrained=pretrained,
            )
        logger.info(f"Loaded Model {self.model}")
        if image_size is None:
            image_size = self.model.default_cfg["input_size"][-1]
        logger.info(f"image_size: {image_size}")
        # get model specific transforms (normalization, resize)
        self.transforms = create_transform(
            **resolve_data_config(
                self.model.pretrained_cfg,
                model=self.model,
                verbose=True,
                use_test_size=True,
            ),
            is_training=False,
        )

        self.transforms = T.Compose(
            [
                T.Resize(
                    size=(image_size, image_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                )
            ]
            + [
                transform
                for transform in self.transforms.transforms
                if "CenterCrop" not in transform.__class__.__name__
                and "Resize" not in transform.__class__.__name__
            ]
        )
        # iterate over compose transforms and remove centercrop and resize

        output_shape = self.get_output_shape()["raw_features"]
        self.num_output_features = output_shape[2]
        self.num_patches = output_shape[1]

    def forward(self, x):
        # output is a (1, num_features) shaped tensor

        if hasattr(self.model, "get_intermediate_layers"):
            per_layer_raw_features = self.model.get_intermediate_layers(
                x,
                n=[i for i in range(len(self.model.blocks))],
                reshape=False,
                norm=True,
            )
        else:
            per_layer_raw_features = [self.model(x)[-1]]

        if len(per_layer_raw_features) == 0:
            return None

        raw_features_as_sequence = None
        if per_layer_raw_features:
            raw_features = per_layer_raw_features[-1]
            if len(raw_features.shape) == 4:
                feature_shape = raw_features.shape
                if (
                    len(feature_shape) == 4
                ):  # this is a 2D CNN, must move channels and h*w around to match b, s, f format
                    raw_features_as_sequence = raw_features.permute(
                        [0, 2, 3, 1]
                    ).reshape(
                        feature_shape[0], -1, feature_shape[1]
                    )  # output should have shape (batch_size, num_patches, num_features)
            else:
                raw_features_as_sequence = raw_features

        features = (
            raw_features_as_sequence.mean(dim=1)
            if raw_features_as_sequence is not None
            else None
        )

        return {
            "classifier": features,
            "features": features,
            "raw_features": raw_features_as_sequence,
            "per_layer_raw_features": per_layer_raw_features,
        }

    def get_transforms(self):
        return {"image": lambda x: self.transforms(x)}

    def get_output_shape(self):
        img = Image.open(
            urlopen(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
        )
        output_dict = self.forward(self.transforms(img).unsqueeze(0))
        shape_dict = {
            k: (
                v.shape
                if isinstance(v, torch.Tensor)
                else [item.shape for item in v]
            )
            for k, v in output_dict.items()
        }
        return shape_dict


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


class TimmModelPaths:
    clip_vit_base_patch16: str = "vit_base_patch16_siglip_224"


class GATETimmImageEncoder(GATEImageEncoder):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        num_projection_features: Optional[int] = None,
    ):
        super().__init__()
        self.image_size = image_size if image_size is not None else 224

        self.vision_model = TimmModel(
            model_identifier=model_name,
            pretrained=pretrained,
            image_size=self.image_size,
        )

        self.image_num_raw_features = self.vision_model.get_output_shape()[
            "raw_features"
        ][2]

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
    name="timm",
)
class TimmCLIPEncoder(GATEImageTextEncoder, nn.Module):
    def __init__(
        self,
        timm_model_name: str = TimmModelPaths.clip_vit_base_patch16,
        clip_model_name: str = CLIPModelPaths.openai_b_16,
        pretrained: bool = True,
        image_size: Optional[int] = 224,
        num_projection_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        image_embedding = GATETimmImageEncoder(
            model_name=timm_model_name,
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
