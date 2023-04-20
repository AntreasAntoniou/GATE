from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.nn as nn


@dataclass
class SourceModalityConfig:
    image: bool = False
    text: bool = False
    audio: bool = False
    video: bool = False


@dataclass
class TargetModalityConfig:
    image: Optional[List[SourceModalityConfig]] = None
    text: Optional[List[SourceModalityConfig]] = None
    audio: Optional[List[SourceModalityConfig]] = None
    video: Optional[List[SourceModalityConfig]] = None


class GATEModel(nn.Module):
    def __init__(self, config: Any, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = config

        for (
            target_modality_name,
            source_modality_dict_list,
        ) in self.config.__dict__.items():
            if source_modality_dict_list is not None:
                for source_modality_dict in source_modality_dict_list:
                    supported_modalities_str = "_".join(
                        [
                            key
                            for key, value in source_modality_dict.__dict__.items()
                            if value == True
                        ]
                    )
                    setattr(
                        self,
                        f"forward_from_"
                        f"{supported_modalities_str}"
                        f"_to_"
                        f"{target_modality_name}",
                        NotImplementedError,
                    )

    def forward(self, input_dict: Dict):
        return self.model(self.transform(input_dict))


if __name__ == "__main__":
    from rich import print

    gate_model_config = TargetModalityConfig(
        image=[
            SourceModalityConfig(text=True, image=True),
            SourceModalityConfig(audio=True, image=True),
            SourceModalityConfig(audio=True, text=True, image=True),
        ]
    )

    model = GATEModel(gate_model_config, nn.Linear(10, 10))

    print(model.__dict__)
