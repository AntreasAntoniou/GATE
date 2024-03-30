# import numpy as np
# import torch
# from transformers import AutoImageProcessor, TimesformerModel

# video = 2 * [8 * [torch.rand((3, 224, 224))]]


# processor = AutoImageProcessor.from_pretrained(
#     "facebook/timesformer-base-finetuned-k400"
# )
# model = TimesformerModel.from_pretrained(
#     "facebook/timesformer-base-finetuned-k400"
# )

# inputs = processor(video, return_tensors="pt")
# inputs["output_hidden_states"] = True
# with torch.no_grad():
#     outputs = model(**inputs)

# for key, value in outputs.items():
#     if isinstance(value, torch.Tensor):
#         print(f"{key}: {value.shape}")
#     elif isinstance(value, tuple):
#         for i, v in enumerate(value):
#             print(f"{key}[{i}]: {v.shape}")
#     else:
#         print(f"{key}: {len(value)}")
