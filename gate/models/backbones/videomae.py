# import numpy as np
# import torch
# from transformers import VideoMAEImageProcessor, VideoMAEModel

# video = 2 * [1 * [torch.rand((3, 224, 224))]]

# processor = VideoMAEImageProcessor.from_pretrained(
#     "MCG-NJU/videomae-base-finetuned-kinetics"
# )
# model = VideoMAEModel.from_pretrained(
#     "MCG-NJU/videomae-base-finetuned-kinetics"
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
