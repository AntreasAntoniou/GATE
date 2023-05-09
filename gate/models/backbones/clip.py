from collections import defaultdict
from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput


class CLIPAdapter(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.clip = CLIPModel.from_pretrained(model_name)

        self.vision_model = self.clip.vision_model
        self.text_model = self.clip.text_model

        self.image_num_features = self.clip.vision_embed_dim
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
            image: CLIPOutput = self.vision_model(
                pixel_values=image
            ).pooler_output
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
            ).pixel_values.squeeze(1),
            "text": lambda x: self.preprocessor(
                text=x, return_tensors="pt", padding=True, truncation=True
            ).input_ids.squeeze(0),
        }
