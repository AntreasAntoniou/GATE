import logging
from typing import Any, Optional

from gate.boilerplate.decorators import configurable
from gate.orchestration.evaluators.classification import (
    ClassificationEvaluator,
)

logger = logging.getLogger(__name__)


@configurable(group="evaluator", name="image_to_text_zero_shot_classification")
class ImageToTextZeroShotClassificationEvaluator(ClassificationEvaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image_text",
            target_modality="image_text",
            model_selection_metric_name="image_to_text_accuracy-epoch-mean",
            model_selection_metric_higher_is_better=True,
        )
