from typing import Dict, Optional

import torch.nn as nn
import torch


class BackboneWithLinear(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_clip_features,
        num_classes: int,
        modality: str,
    ):
        super().__init__()
        self.model = model
        self.modality = modality
        self.linear = nn.Linear(num_clip_features, num_classes)

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_dict is not None:
            x = self.model(**input_dict)[self.modality]["features"]

        if image is not None:
            x = self.model(image=image)["image"]["features"]

        if text is not None:
            x = self.model(text=text)["text"]["features"]

        if audio is not None:
            x = self.model(audio=audio)["audio"]["features"]

        if video is not None:
            x = self.model(video=video)["video"]["features"]

        x = self.linear(x)
        return x
