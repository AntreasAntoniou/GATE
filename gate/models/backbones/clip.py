from collections import defaultdict
from typing import Optional
from urllib.request import urlopen
from PIL import Image

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput

from gate.models.backbones import image_dim_reshape
from gate.models.core import reinit


def forward_dict(self, x):
    output = self.legacy_forward(x)
    return {
        "features": output.pooler_output,
        "raw_features": output.last_hidden_state,
    }


class CLIPAdapter(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.tokenizer = self.preprocessor
        self.clip = CLIPModel.from_pretrained(model_name)

        if not pretrained:
            self.clip.init_weights()

        self.vision_model = self.clip.vision_model
        self.text_model = self.clip.text_model

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

        self.image_num_features = self.clip.vision_embed_dim
        self.text_num_features = self.clip.text_embed_dim

    def init_weights(self):
        reinit(self)

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
            output_dict["image"] = self.vision_model(x=image)

        if text is not None:
            output_dict["text"] = self.text_model(x=text)

        return output_dict

    def get_transforms(self):
        return {
            "image": lambda x: self.preprocessor(
                images=x, return_tensors="pt"
            ).pixel_values.squeeze(0),
            "text": lambda x: self.preprocessor(
                text=x, return_tensors="pt", padding=True, truncation=True
            ).input_ids.squeeze(0),
        }
