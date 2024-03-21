from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class FewShotLearningClassificationEpisode:
    support_set_inputs: Dict[str, torch.Tensor]
    query_set_inputs: Dict[str, torch.Tensor]
    support_set_labels: torch.Tensor
    query_set_labels: Optional[torch.Tensor] = (None,)
