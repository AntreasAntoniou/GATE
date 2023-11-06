from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional
from urllib.request import urlopen

import PIL
import PIL.Image as Image
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from timm.data import InterpolationMode, resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput

from gate.boilerplate.decorators import configurable
from gate.config.variables import HYDRATED_NUM_CLASSES
from gate.models.backbones import GATEncoder, Modality, image_dim_reshape
from gate.models.backbones.clip_image import TextProcessor
from gate.models.core import reinit

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


import logging

logger = logging.getLogger(__name__)


class TimmModel(nn.Module):
    def __init__(
        self,
        model_identifier: str = "hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k",
        img_size: Optional[List[int]] = None,
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
            print(
                f"model_identifier: {model_identifier}, pretrained: {pretrained}, img_size: {img_size}"
            )
            self.model = timm.create_model(
                model_name=model_identifier,
                img_size=img_size,
                pretrained=pretrained,
            )
        logger.info(f"Loaded Model {self.model}")
        if img_size is None:
            img_size = self.model.default_cfg["input_size"][-2:]

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
            [T.Resize(size=img_size, interpolation=InterpolationMode.BICUBIC)]
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
            per_layer_raw_features = [output for output in self.model(x)]

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
            k: v.shape
            if isinstance(v, torch.Tensor)
            else [item.shape for item in v]
            for k, v in output_dict.items()
        }
        return shape_dict


class CLIPModelPaths:
    laion_b_16: str = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    openai_b_16: str = "openai/clip-vit-base-patch16"


@configurable(
    group="encoder",
    name="timm",
    defaults=dict(num_classes=HYDRATED_NUM_CLASSES),
)
class TimmCLIPAdapter(GATEncoder):
    def __init__(
        self,
        timm_model_name: str,
        clip_model_name: str,
        pretrained: bool = True,
        image_size: Optional[List[int]] = None,
    ):
        super().__init__()
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            clip_model_name
        )

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.text_transforms = TextProcessor(self.preprocessor)

        self.vision_model = TimmModel(
            model_identifier=timm_model_name,
            pretrained=pretrained,
            img_size=image_size,
        )
        self.text_model = self.clip.text_model

        self.vision_model_output_shape = self.vision_model.get_output_shape()[
            "raw_features"
        ]
        self.image_num_features = self.vision_model_output_shape[2]
        self.text_num_features = self.clip.text_embed_dim

    @property
    def num_in_features_image(self):
        return self.image_num_features

    @property
    def num_in_features_text(self):
        return self.text_num_features

    @property
    def num_in_features_video(self):
        raise NotImplementedError(f"TimmCLIP does not have a video backbone")

    def init_weights(self):
        reinit(self)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if image is None and text is None and video is None:
            raise ValueError(
                f"Must provide at least one input modality"
                f"to {self.__class__.__name__}"
            )

        output_dict = defaultdict(dict)

        # self.model.forward expects
        # input_ids: Optional[torch.LongTensor] = None,
        # pixel_values: Optional[torch.FloatTensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # return_loss: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,

        if image is not None:
            output_dict["image"] = self.vision_model.forward(image)

        if video is not None:
            if len(video.shape) == 5:
                b, s, c, h, w = video.shape
                output_dict["video"] = self.vision_model.forward(
                    video.view(b * s, c, h, w)
                )
                for k, v in output_dict["video"].items():
                    if v is not None:
                        if isinstance(v, list) or isinstance(v, tuple):
                            v = torch.stack(v, dim=2)
                        output_dict["video"][k] = v.view(b, s, *v.shape[1:])
            else:
                output_dict["video"] = self.vision_model.forward(video)

        if text is not None:
            text: CLIPOutput = self.text_model(input_ids=text)
            output_dict["text"]["features"] = text.pooler_output
            output_dict["text"]["raw_features"] = text.last_hidden_state

        return output_dict

    def get_transforms(self):
        def image_transforms(x):
            return self.vision_model.transforms(x)

        def text_transforms(x):
            return self.text_transforms.apply_transform(x)

        def image_transforms_process_multi_type(x):
            if isinstance(x, List):
                return [
                    apply_preprocessing_transforms(
                        x=item,
                        transforms=image_transforms,
                        modality=Modality.image,
                    )
                    for item in x
                ]
            else:
                return apply_preprocessing_transforms(
                    x=x, transforms=image_transforms, modality=Modality.image
                )

        def text_transforms_process_multi_type(x):
            return apply_preprocessing_transforms(
                x=x, transforms=text_transforms, modality=Modality.text
            )

        def video_transforms_process_multi_type(x):
            return torch.stack(
                [image_transforms_process_multi_type(item) for item in x],
                dim=0,
            )

        return {
            "image": lambda x: image_transforms_process_multi_type(x),
            "text": lambda x: text_transforms_process_multi_type(x),
            "video": lambda x: video_transforms_process_multi_type(x),
        }
