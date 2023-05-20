from urllib.request import urlopen
from PIL import Image
import timm
import torch

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)

model = timm.create_model(
    "vit_large_patch14_clip_224.openai_ft_in12k_in1k", pretrained=True
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=True)
print(transforms)
output = model(
    transforms(img).unsqueeze(0)
)  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(
    output.softmax(dim=1) * 100, k=5
)
