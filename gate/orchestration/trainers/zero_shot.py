import logging
from typing import Any, Optional

import torch

from gate.boilerplate.decorators import configurable
from gate.orchestration.trainers.classification import ClassificationTrainer

logger = logging.getLogger(__name__)


@configurable(group="trainer", name="image_to_text_zero_shot_classification")
class ImageToTextZeroShotClassificationTrainer(ClassificationTrainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = "step",
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            optimizer,
            scheduler,
            scheduler_interval,
            experiment_tracker,
            source_modality="image_text",
            target_modality="image_text",
        )
