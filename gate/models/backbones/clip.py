from collections import defaultdict
from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput


class CLIPAdapter(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name
        )
        self.clip = CLIPModel.from_pretrained(model_name)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
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

        if image:
            image: CLIPOutput = self.clip(pixel_values=image)
            output_dict[
                "image_features"
            ] = image.vision_model_output.last_hidden_state
            output_dict[
                "image_projection_output"
            ] = image.vision_model_output.image_embeds

        if text:
            text: CLIPOutput = self.clip(input_ids=text)
            output_dict[
                "text_features"
            ] = text.text_model_output.last_hidden_state
            output_dict[
                "text_projection_output"
            ] = text.text_model_output.text_embeds

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
