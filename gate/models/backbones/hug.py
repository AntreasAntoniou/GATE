# from collections import defaultdict
# from typing import List, Optional
# from urllib.request import urlopen

# import torch
# import torch.nn as nn
# from PIL import Image
# from transformers import CLIPModel, CLIPProcessor
# from transformers.models.clip.modeling_clip import CLIPOutput

# from gate.models.backbones import (
#     Modality,
#     apply_preprocessing_transforms,
#     image_dim_reshape,
# )
# from gate.models.core import reinit


# def forward_dict(self, x):
#     output = self.legacy_forward(x)
#     return {
#         "features": output.pooler_output,
#         "raw_features": output.last_hidden_state,
#     }


# from transformers import ViTImageProcessor, ViTModel
# from PIL import Image
# import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
# model = ViTModel.from_pretrained("facebook/dino-vitb16")

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

# laion_path = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
# deit_path = "facebook/deit-base-patch16-224"
# dino_path = "facebook/dino-vitb16"
# sam_path = "facebook/sam-vit-base"


# class CLIPAdapter(nn.Module):
#     def __init__(self, model_name: str, pretrained: bool = True):
#         super().__init__()
#         self.preprocessor: CLIPProcessor = CLIPProcessor.from_pretrained(
#             model_name
#         )
#         self.tokenizer = self.preprocessor.tokenizer
#         self.clip = CLIPModel.from_pretrained(model_name)

#         if not pretrained:
#             self.clip.init_weights()

#         self.vision_model = self.clip.vision_model
#         self.visual_projection = self.clip.visual_projection
#         self.text_model = self.clip.text_model
#         self.text_projection = self.clip.text_projection

#         setattr(self.vision_model, "legacy_forward", self.vision_model.forward)
#         setattr(self.text_model, "legacy_forward", self.text_model.forward)

#         setattr(
#             self.vision_model,
#             "forward",
#             forward_dict.__get__(self.vision_model),
#         )
#         setattr(
#             self.text_model, "forward", forward_dict.__get__(self.text_model)
#         )

#         self.image_num_features = self.clip.vision_embed_dim
#         self.text_num_features = self.clip.text_embed_dim

#     def init_weights(self):
#         reinit(self)

#     def forward(
#         self,
#         image: Optional[torch.Tensor] = None,
#         text: Optional[torch.Tensor] = None,
#         **kwargs,
#     ):
#         if image is None and text is None:
#             raise ValueError(
#                 f"Must provide at least one input modality"
#                 f"to {self.__class__.__name__}"
#             )
#         output_dict = defaultdict(dict)

#         # self.model.forward expects
#         # input_ids: Optional[torch.LongTensor] = None,
#         # pixel_values: Optional[torch.FloatTensor] = None,
#         # attention_mask: Optional[torch.Tensor] = None,
#         # position_ids: Optional[torch.LongTensor] = None,
#         # return_loss: Optional[bool] = None,
#         # output_attentions: Optional[bool] = None,
#         # output_hidden_states: Optional[bool] = None,
#         # return_dict: Optional[bool] = None,

#         if image is not None:
#             output_dict["image"] = self.vision_model(x=image)
#             output_dict["image"]["projection_output"] = self.visual_projection(
#                 output_dict["image"]["features"]
#             )

#         if text is not None:
#             output_dict["text"] = self.text_model(x=text)
#             output_dict["text"]["projection_output"] = self.text_projection(
#                 output_dict["text"]["features"]
#             )

#         return output_dict

#     def get_transforms(self):
#         def image_transforms(x):
#             return self.preprocessor(
#                 images=x, return_tensors="pt"
#             ).pixel_values.squeeze(0)

#         def text_transforms(x):
#             return self.preprocessor(
#                 text=x, return_tensors="pt", padding=True, truncation=True
#             ).input_ids.squeeze(0)

#         def image_transforms_process_multi_type(x):
#             if isinstance(x, List):
#                 return [
#                     apply_preprocessing_transforms(
#                         x=item,
#                         transforms=image_transforms,
#                         modality=Modality.image,
#                     )
#                     for item in x
#                 ]
#             else:
#                 return apply_preprocessing_transforms(
#                     x=x, transforms=image_transforms, modality=Modality.image
#                 )

#         def text_transforms_process_multi_type(x):
#             return apply_preprocessing_transforms(
#                 x=x, transforms=text_transforms, modality=Modality.text
#             )

#         return {
#             "image": lambda x: image_transforms_process_multi_type(x),
#             "text": lambda x: text_transforms_process_multi_type(x),
#         }