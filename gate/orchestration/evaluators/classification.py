import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator

from gate.boilerplate.decorators import configurable
from gate.orchestration.evaluators import Evaluator, EvaluatorOutput

logger = logging.getLogger(__name__)


def get_dict_shapes(x):
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


@dataclass
class StepOutput:
    metrics: Dict
    loss: torch.Tensor
    experiment_tracker: Optional[Any] = None
    global_step: Optional[int] = None


class ClassificationEvaluator(Evaluator):
    def select_metrics_to_report(self, output_dict):
        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                if isinstance(value, torch.Tensor):
                    self.current_epoch_dict[key].append(value.detach().cpu())

    def step(self, model, batch, global_step, accelerator: Accelerator):
        output_dict = model.forward(batch)
        output_dict = output_dict[self.target_modality][self.source_modality]

        loss = output_dict["loss"]

        for key, value in output_dict.items():
            if isinstance(value, torch.Tensor):
                self.current_epoch_dict[key].append(
                    value.detach().float().mean().cpu()
                )

        return StepOutput(
            metrics=output_dict,
            loss=loss,
        )

    @torch.inference_mode()
    def validation_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> EvaluatorOutput:
        model.eval()

        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
        )

        metrics = step_output.metrics

        return EvaluatorOutput(
            phase_name="validation",
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @torch.inference_mode()
    def testing_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
        prefix: Optional[str] = None,
    ) -> EvaluatorOutput:
        model.eval()

        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
        )

        metrics = step_output.metrics

        return EvaluatorOutput(
            phase_name=f"testing/{prefix}" if prefix else "testing",
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )


@configurable(group="evaluator", name="image_classification")
class ImageClassificationEvaluator(ClassificationEvaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image",
            target_modality="image",
            model_selection_metric_name="accuracy_top_1-epoch-mean",
            model_selection_metric_higher_is_better=True,
        )

    def collect_image_classification_episode(
        self, output_dict, global_step, batch
    ):
        # if "logits" in output_dict:
        #     if global_step % 25 == 0:
        #         output_dict["image_class_episode"] = {
        #             "image": batch["image"],
        #             "logits": output_dict["logits"],
        #             "label": batch["labels"],
        #         }

        #     del output_dict["logits"]
        return output_dict

    def step(self, model, batch, global_step, accelerator: Accelerator):
        # batch["return_loss_and_metrics"] = True
        output_dict = model.forward(batch)[self.target_modality][
            self.source_modality
        ]

        loss = output_dict["loss"]

        for key, value in output_dict.items():
            if isinstance(value, torch.Tensor):
                self.current_epoch_dict[key].append(
                    value.detach().float().mean().cpu()
                )

        output_dict = self.collect_image_classification_episode(
            output_dict, global_step, batch
        )

        return StepOutput(
            metrics=output_dict,
            loss=loss,
        )


@configurable(group="evaluator", name="visual_relational_reasoning")
class VisualRelationalClassificationTrainer(ClassificationEvaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image_text",
            target_modality="image_text",
            model_selection_metric_name="accuracy_top_1-epoch-mean",
            model_selection_metric_higher_is_better=True,
        )
