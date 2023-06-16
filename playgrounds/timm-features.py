import torch
import timm

m = timm.create_model("resnest26d", features_only=True, pretrained=True)
o = m(torch.randn(2, 3, 224, 224))
for x in o:
    print(x.shape)
