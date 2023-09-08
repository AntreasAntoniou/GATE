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
from gate.metrics.multi_class_classification import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from gate.orchestration.trainers import Trainer, TrainerOutput

logger = get_logger(__name__)


def get_dict_shapes(x):
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


class ClassificationTrainer(Trainer):
    def get_optimizer(self):
        return self.optimizer

    def step(self, model, batch, global_step, accelerator: Accelerator):
        start_time = time.time()
        output_dict = model.forward(batch)[self.target_modality][
            self.source_modality
        ]
        logger.debug(f"Forward time {time.time() - start_time}")

        loss = output_dict["loss"]

        start_time = time.time()
        accelerator.backward(loss)
        logger.debug(f"Backward time {time.time() - start_time}")

        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                if isinstance(value, torch.Tensor):
                    self.current_epoch_dict[key].append(value.detach().cpu())

        return StepOutput(
            output_metrics_dict=output_dict,
            loss=loss,
        )

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        model.train()
        start_time = time.time()
        self.optimizer.zero_grad()
        logger.debug(f"Zero grad time {time.time() - start_time}")
        start_time = time.time()
        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
        )
        logger.debug(f"Step time {time.time() - start_time}")
        start_time = time.time()

        self.optimizer.step()
        self.scheduler.step(step_output.loss)
        logger.debug(f"Optimizer step time {time.time() - start_time}")

        metrics = step_output.output_metrics_dict
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return TrainerOutput(
            phase_name="training",
            opt_loss=step_output.output_metrics_dict["loss"],
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )


@configurable(group="trainer", name="image_classification")
class ImageClassificationTrainer(ClassificationTrainer):
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


@configurable(group="trainer", name="visual_relational_reasoning")
class VisualRelationalClassificationTrainer(ClassificationTrainer):
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


@configurable(
    group="trainer",
    name="multi_class_classification",
    defaults={"label_idx_to_class_name": HYDRATED_LABEL_IDX_TO_CLASS_NAME},
)
class MultiClassClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = "step",
        experiment_tracker: Optional[Any] = None,
        label_idx_to_class_name: Dict[int, str] = None,
    ):
        super().__init__(
            optimizer,
            scheduler,
            scheduler_interval,
            experiment_tracker,
            source_modality="image",
            target_modality="image",
        )
        self.label_idx_to_class_name = label_idx_to_class_name

    @property
    def metrics(self):
        return {
            "auc": roc_auc_score,
            "aps": average_precision_score,
            "bs": brier_score_loss,
        }

    def compute_epoch_metrics(
        self, phase_metrics: Dict[str, float], global_step: int
    ):
        for key, value in self.current_epoch_dict.items():
            if key not in ["labels", "logits"]:
                phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
                phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        labels = torch.cat(self.current_epoch_dict["labels"]).detach().cpu()
        logits = torch.cat(self.current_epoch_dict["logits"]).detach().cpu()
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
            for key, value in phase_metrics.items():
                if key not in self.current_epoch_dict:
                    self.current_epoch_dict[key] = {
                        global_step: phase_metrics[key]
                    }
                else:
                    self.current_epoch_dict[key][global_step] = phase_metrics[
                        key
                    ]
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
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                self.current_epoch_dict.setdefault(key, []).append(value)

        # we need to round the labels because they might be soft labels due to mixup/label smoothing
        self.current_epoch_dict.setdefault("labels", []).append(
            batch["labels"].cpu().round()
        )
        self.current_epoch_dict.setdefault("logits", []).append(
            output_dict[self.target_modality][self.source_modality]["logits"]
            .cpu()
            .sigmoid_()
        )

    def get_optimizer(self):
        return self.optimizer

    def step(self, model, batch, global_step, accelerator: Accelerator):
        # print({key: value.shape for key, value in batch.items()})
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

        accelerator.backward(loss)

        return StepOutput(
            output_metrics_dict=output_dict,
            loss=loss,
        )

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        model.train()

        self.optimizer.zero_grad()

        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
        )
        if not isinstance(
            step_output.output_metrics_dict["loss"], torch.Tensor
        ):
            opt_loss = torch.mean(
                torch.stack(step_output.output_metrics_dict["loss"])
            )
        else:
            opt_loss = step_output.output_metrics_dict["loss"]

        self.optimizer.step()
        self.scheduler.step(opt_loss)

        metrics = step_output.output_metrics_dict
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return TrainerOutput(
            phase_name="training",
            opt_loss=opt_loss,
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_training(
        self,
        global_step: int,
    ):
        phase_metrics = {}

        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics=phase_metrics,
            phase_name="training",
            experiment_tracker=self.experiment_tracker,
        )
