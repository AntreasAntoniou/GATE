import inspect

import timm
import torch
from rich import print


def get_class_definition_path(obj):
    # Get the class of the object
    cls = obj.__class__

    # Get the module of the class
    module = inspect.getmodule(cls)

    # Get the file path of the module
    file_path = inspect.getfile(module)

    return file_path


m = timm.create_model("resnest26d", features_only=True, pretrained=True)


# m = timm.create_model("vit_base_patch16_clip_224.laion2b", pretrained=True)
x = torch.randn(2, 3, 224, 224)
o = m.get_intermediate_layers(
    x=x, n=15, reshape=False, return_class_token=False, norm=True
)

# has forward, forward_features, forward_head, and get_intermediate_layers
# print(get_class_definition_path(m))
for idx, item in enumerate(o):
    print(f"{idx}: {item.shape}")
