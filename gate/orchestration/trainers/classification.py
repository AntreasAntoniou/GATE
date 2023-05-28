from dataclasses import dataclass
import time
from typing import Any, Dict, Optional
from accelerate.utils.constants import op

import numpy as np
import torch
from torch.autograd import backward
import torch.nn.functional as F
from accelerate import Accelerator
from sklearn.metrics import average_precision_score, brier_score_loss

from gate.boilerplate.decorators import collect_metrics, configurable
from gate.boilerplate.utils import get_logger
from gate.metrics.core import accuracy_top_k
from gate.metrics.segmentation import roc_auc_score
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
        # print({key: value.shape for key, value in batch.items()})
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"{key}: {value.shape}")
        #     else:
        #         print(f"{key}: {value}") # uncomment for debugging
        pre_forward_time = time.time()
        output_dict = model.forward(batch)[self.target_modality][
            self.source_modality
        ]
        post_forward_time = time.time()
        forward_time = post_forward_time - pre_forward_time
        logger.debug(f"Forward time: {forward_time} seconds")

        # model(image, text) -> image_preds -- source: image_text and target: image
        # model(image) -> image_preds -- source: image and target: image

        if "loss" not in output_dict:
            loss = F.cross_entropy(
                output_dict["logits"],
                batch["labels"],
            )
            accuracy = accuracy_top_k(
                logits=output_dict["logits"],
                labels=batch["labels"],
                k=1,
            )
            accuracy_top_5 = accuracy_top_k(
                logits=output_dict["logits"],
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

        pre_backward_time = time.time()
        accelerator.backward(loss)
        post_backward_time = time.time()
        backward_time = post_backward_time - pre_backward_time
        logger.debug(f"Backward time: {backward_time} seconds")
        for key, value in output_dict.items():
            self.current_epoch_dict[key].append(value.detach().mean().cpu())

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

        pre_optimizer_step_time = time.time()
        self.optimizer.step()
        self.scheduler.step(epoch=global_step, metric=step_output.loss)
        post_optimizer_step_time = time.time()
        optimizer_step_time = (
            post_optimizer_step_time - pre_optimizer_step_time
        )
        logger.debug(f"Optimizer step time: {optimizer_step_time} seconds")

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


@configurable(group="trainer", name="multi_class_classification")
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

        labels = torch.cat(self.current_epoch_dict["labels"])
        logits = torch.cat(self.current_epoch_dict["logits"])
        for metric_name, metric_fn in self.metrics.items():
            for c_idx, class_name in enumerate(self.classes):
                y_true = labels[:, c_idx]
                y_score = logits[:, c_idx]
                if metric_name == "bs":
                    phase_metrics[f"{class_name}-{metric_name}"] = metric_fn(
                        y_true, y_prob=y_score
                    )
                else:
                    phase_metrics[f"{class_name}-{metric_name}"] = metric_fn(
                        y_true, y_score
                    )

            phase_metrics[f"{metric_name}-macro"] = np.mean(
                [
                    phase_metrics[f"{class_name}-{metric_name}"]
                    for class_name in self.classes
                ]
            )

        for key, value in phase_metrics.items():
            if key not in self.current_epoch_dict:
                self.current_epoch_dict[key] = {global_step: value}
            else:
                self.current_epoch_dict[key][global_step] = value

        return phase_metrics

    def compute_step_metrics(self, output_dict, batch, loss):
        # fallback to numbering classes if no class names are provided
        if self.label_idx_to_class_name is None:
            self.label_idx_to_class_name = [
                f"class-{idx}" for idx in range(batch["labels"].shape[1])
            ]

        metrics = {"loss": loss.mean()}
        for c_idx, class_name in enumerate(self.classes):
            metrics[f"{class_name}-loss"] = loss[:, c_idx].mean()

        for key, value in metrics.items():
            self.current_epoch_dict.setdefault(key, []).append(
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

    def get_optimizer(self):
        return self.optimizer

    def step(self, model, batch, global_step, accelerator: Accelerator):
        # print({key: value.shape for key, value in batch.items()})
        output_dict = model.forward(batch)
        if "loss" not in output_dict:
            loss = F.binary_cross_entropy_with_logits(
                output_dict[self.target_modality][self.source_modality][
                    "logits"
                ],
                batch["labels"],
                reduction="none",
            )

            self.compute_metrics(output_dict, batch, loss)

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

        self.optimizer.step()
        self.scheduler.step(global_step)

        metrics = step_output.output_metrics_dict
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return TrainerOutput(
            phase_name="training",
            opt_loss=torch.mean(
                torch.stack(step_output.output_metrics_dict["loss"])
            ),
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

        for key, value in self.current_epoch_dict.items():
            if key not in ["labels", "logits"]:
                phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
                phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        phase_metrics = self.compute_epoch_metrics(phase_metrics, global_step)

        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics=phase_metrics,
            phase_name="training",
            experiment_tracker=self.experiment_tracker,
        )
