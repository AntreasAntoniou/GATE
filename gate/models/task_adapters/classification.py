from typing import Dict

import torch.nn as nn


class BackboneWithLinear(nn.Module):
    def __init__(self, model: nn.Module, num_clip_features, num_classes: int):
        super().__init__()
        self.model = model
        self.linear = nn.Linear(num_clip_features, num_classes)

    def forward(self, input_dict: Dict):
        print(list(input_dict.keys()))
        x = self.model(**input_dict)
        x = self.linear(x)
        return x
