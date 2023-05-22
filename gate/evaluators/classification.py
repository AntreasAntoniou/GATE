from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics, configurable
from gate.boilerplate.utils import get_logger
from gate.metrics.core import accuracy_top_k
from gate.orchestration.evaluators import Evaluator, EvaluatorOutput

logger = get_logger(__name__)


def get_dict_shapes(x):
    print(x)
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


@dataclass
class StepOutput:
    output_metrics_dict: Dict
    loss: torch.Tensor


class ClassificationEvaluator(Evaluator):
    def step(self, model, batch, global_step, accelerator: Accelerator):
        # print({key: value.shape for key, value in batch.items()})
        output_dict = model.forward(batch)
        if "loss" not in output_dict:
            loss = F.cross_entropy(
                output_dict[self.target_modality][self.souce_modality],
                batch["labels"],
            )
            accuracy = accuracy_top_k(
                logits=output_dict[self.target_modality][self.souce_modality],
                labels=batch["labels"],
                k=1,
            )
            accuracy_top_5 = accuracy_top_k(
                logits=output_dict[self.target_modality][self.souce_modality],
                labels=batch["labels"],
                k=5,
            )
            output_dict = {
                "loss": loss,
                "accuracy_top_1": accuracy,
                "accuracy_top_5": accuracy_top_5,
            }
        else:
            loss = output_dict["loss"]

        accelerator.backward(loss)

        return StepOutput(
            output_metrics_dict=output_dict,
            loss=loss,
        )

    @collect_metrics
    def validation_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> EvaluatorOutput:
        model.train()

        self.optimizer.zero_grad()

        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
        )

        self.optimizer.step()

        metrics = step_output.output_metrics_dict

        return EvaluatorOutput(
            phase_name="validation",
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def testing_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> EvaluatorOutput:
        model.train()

        self.optimizer.zero_grad()

        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
        )

        self.optimizer.step()

        metrics = step_output.output_metrics_dict

        return EvaluatorOutput(
            phase_name="testing",
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )


@configurable(group="evaluator", name="image_classification")
class ImageClassificationEvaluator(ClassificationEvaluator):
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
            source_modality="image",
            target_modality="image",
        )


@configurable(group="evaluator", name="image_to_text_zero_shot_classification")
class ImageToTextZeroShotClassificationEvaluator(ClassificationEvaluator):
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
            target_modality="image",
        )
