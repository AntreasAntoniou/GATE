from collections import defaultdict
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics, configurable
from gate.boilerplate.utils import get_logger
from gate.config.variables import HYDRATED_LABEL_IDX_TO_CLASS_NAME
from gate.metrics.core import accuracy_top_k
from gate.metrics.multi_class_classification import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
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
    metrics: Dict
    loss: torch.Tensor
    experiment_tracker: Optional[Any] = None
    global_step: Optional[int] = None


class ClassificationEvaluator(Evaluator):
    def step(self, model, batch, global_step, accelerator: Accelerator):
        output_dict = model.forward(batch)
        print(output_dict)
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
    @collect_metrics
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
    @collect_metrics
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


@configurable(group="evaluator", name="image_semantic_segmentation")
class ImageSemanticSegmentationEvaluator(ClassificationEvaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image",
            target_modality="image",
            model_selection_metric_name="mean_iou-epoch-mean",
            model_selection_metric_higher_is_better=True,
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


@configurable(
    group="evaluator",
    name="multi_class_classification",
    defaults={"label_idx_to_class_name": HYDRATED_LABEL_IDX_TO_CLASS_NAME},
)
class MultiClassClassificationEvaluator(Evaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
        label_idx_to_class_name: Optional[Dict[int, str]] = None,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image",
            target_modality="image",
            model_selection_metric_name="auc-macro",
            model_selection_metric_higher_is_better=True,
        )
        self.label_idx_to_class_name = label_idx_to_class_name

    @property
    def metrics(self):
        return {
            "auc": roc_auc_score,
            "aps": average_precision_score,
            "bs": brier_score_loss,
        }

    def compute_epoch_metrics(self, global_step: int):
        phase_metrics = {}
        for key, value in self.current_epoch_dict.items():
            if key not in ["labels", "logits"]:
                phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
                phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        labels = torch.cat(self.current_epoch_dict["labels"]).detach()
        logits = torch.cat(self.current_epoch_dict["logits"]).detach()
        for metric_name, metric_fn in self.metrics.items():
            for c_idx, class_name in enumerate(self.label_idx_to_class_name):
                if metric_name == "bs":
                    phase_metrics[f"{class_name}-{metric_name}"] = metric_fn(
                        y_true=labels[:, c_idx], y_prob=logits[:, c_idx]
                    )
                else:
                    phase_metrics[f"{class_name}-{metric_name}"] = metric_fn(
                        y_true=labels[:, c_idx], y_score=logits[:, c_idx]
                    )
            phase_metrics[f"{metric_name}-macro"] = np.mean(
                [
                    phase_metrics[f"{class_name}-{metric_name}"]
                    for class_name in self.label_idx_to_class_name
                ]
            )
            phase_metrics["global_step"] = global_step
            for key, value in phase_metrics.items():
                if key not in self.per_epoch_metrics:
                    self.per_epoch_metrics[key] = [phase_metrics[key]]
                else:
                    self.per_epoch_metrics[key].append(phase_metrics[key])

        return phase_metrics

    def compute_step_metrics(self, output_dict, batch, loss):
        # fallback to numbering classes if no class names are provided
        if self.label_idx_to_class_name is None:
            self.label_idx_to_class_name = [
                f"class-{idx}" for idx in range(batch["labels"].shape[1])
            ]

        metrics = {"loss": loss.mean()}
        for c_idx, class_name in enumerate(self.label_idx_to_class_name):
            metrics[f"{class_name}-loss"] = loss[:, c_idx].mean()

        for key, value in metrics.items():
            self.per_epoch_metrics.setdefault(key, []).append(
                value.detach().cpu()
            )

        # we need to round the labels because they might be soft labels due to mixup/label smoothing
        self.current_epoch_dict.setdefault("labels", []).append(
            batch["labels"].cpu().round()
        )
        self.current_epoch_dict.setdefault("logits", []).append(
            output_dict[self.target_modality][self.source_modality]["logits"]
            .cpu()
            .sigmoid_()
        )

    def step(self, model, batch, global_step, accelerator: Accelerator):
        batch["return_loss_and_metrics"] = False
        output_dict = model.forward(batch)
        logits = output_dict[self.target_modality][self.source_modality][
            "logits"
        ]
        if "loss" not in output_dict:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch["labels"],
                reduction="none",
            )

            logits = logits.detach().cpu()
            self.compute_step_metrics(output_dict, batch, loss.detach().cpu())
            loss = loss.mean()
            output_dict = {
                "loss": loss,
            }
        else:
            loss = output_dict["loss"]

        return StepOutput(
            metrics=output_dict,
            loss=loss,
        )

    @torch.inference_mode()
    @collect_metrics
    def validation_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> StepOutput:
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
            metrics=metrics,
            global_step=global_step,
            experiment_tracker=self.experiment_tracker,
        )

    @torch.inference_mode()
    @collect_metrics
    def testing_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
        prefix: Optional[str] = None,
    ) -> StepOutput:
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
            metrics=metrics,
            global_step=global_step,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_validation(
        self,
        global_step: int,
    ):
        phase_metrics = self.compute_epoch_metrics(global_step)

        return EvaluatorOutput(
            global_step=global_step,
            metrics=phase_metrics,
            phase_name="validation",
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_testing(
        self,
        global_step: int,
        prefix: Optional[str] = None,
    ):
        phase_metrics = self.compute_epoch_metrics(global_step)

        return EvaluatorOutput(
            global_step=global_step,
            metrics=phase_metrics,
            phase_name=f"testing/{prefix}" if prefix else "testing",
            experiment_tracker=self.experiment_tracker,
        )
