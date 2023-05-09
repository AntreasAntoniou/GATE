from collections import defaultdict
from typing import Optional
from urllib.request import urlopen
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput
import timm
import PIL.Image as Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class TimmModel(nn.Module):
    def __init__(
        self,
        model_identifier: str = "hf_hub:timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k",
        pretrained: bool = True,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name=model_identifier,
            pretrained=pretrained,
            num_classes=0,  # remove classifier nn.Linear
        )

        # get model specific transforms (normalization, resize)
        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        self.num_output_features = self.infer_output_shape()[-1]

    def forward(self, x):
        # output is a (1, num_features) shaped tensor
        return self.forward_features(x)

    def forward_features(self, x):
        # output is unpooled, a (1, 257, 1024) shaped tensor
        return self.model.forward_head(x, pre_logits=True)

    def forward_raw_features(self, x):
        # output is a (1, num_features) shaped tensor
        return self.model.forward_features(x)

    def get_transforms(self):
        return {"image": self.transforms}

    def infer_output_shape(self):
        img = Image.open(
            urlopen(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
        )
        return self.forward(self.transforms(img).unsqueeze(0)).shape


class TimmCLIPAdapter(nn.Module):
    def __init__(
        self,
        timm_model_name: str,
        clip_model_name: str,
        pretrained: bool = True,
    ):
        super().__init__()
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            clip_model_name
        )
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        self.vision_model = TimmModel(
            model_identifier=timm_model_name, pretrained=pretrained
        )
        self.text_model = self.clip.text_model

        self.image_num_features = self.vision_model.num_output_features
        self.text_num_features = self.clip.text_embed_dim

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if image is None and text is None:
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
            image: CLIPOutput = self.vision_model(x=image)
            output_dict["image_features"] = image

        if text is not None:
            text: CLIPOutput = self.text_model(input_ids=text).pooler_output
            output_dict["text_features"] = text

        if len(output_dict) == 1:
            return output_dict[list(output_dict.keys())[0]]

        return output_dict

    def get_transforms(self):
        return {
            "image": lambda x: self.preprocessor(
                images=x, return_tensors="pt"
            ).pixel_values.squeeze(0),
            "text": lambda x: self.vision_model.get_transforms()["image"](x),
        }
