from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class ModelAndTransform:
    model: nn.Module
    transform: Any
