import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics_mark, configurable
from gate.config.variables import HYDRATED_LABEL_IDX_TO_CLASS_NAME
from gate.orchestration.trainers import Trainer, TrainerOutput

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
    output_metrics_dict: Dict
    loss: torch.Tensor


class ClassificationTrainer(Trainer):
    def get_optimizer(self):
        return self.optimizer

    def select_metrics_to_report(self, output_dict):
        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                if isinstance(value, torch.Tensor):
                    self.current_epoch_dict[key].append(value.detach().cpu())

    def step(self, model, batch, global_step, accelerator: Accelerator):
        start_time = time.time()
        output_dict = model.forward(batch)[self.target_modality][
            self.source_modality
        ]
        fprop_time = time.time() - start_time

        loss = output_dict["loss"]

        start_time = time.time()
        accelerator.backward(loss)
        bprop_time = time.time() - start_time

        self.select_metrics_to_report(output_dict)

        return StepOutput(
            output_metrics_dict=output_dict
            | {"fprop_time": fprop_time, "bprop_time": bprop_time},
            loss=loss,
        )

    @collect_metrics_mark
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
        self.scheduler.step(step_output.loss)

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
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_interval: str = "step",
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            experiment_tracker=experiment_tracker,
            source_modality="image",
            target_modality="image",
        )

    def collect_image_episode(self, output_dict, global_step, batch):
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
        start_time = time.time()
        output_dict = model.forward(batch)[self.target_modality][
            self.source_modality
        ]
        fprop_time = time.time() - start_time

        loss = output_dict["loss"]

        start_time = time.time()
        accelerator.backward(loss)
        bprop_time = time.time() - start_time

        self.select_metrics_to_report(output_dict)
        output_dict = self.collect_image_episode(
            output_dict, global_step, batch
        )

        return StepOutput(
            output_metrics_dict=output_dict
            | {
                "fprop_time": fprop_time,
                "bprop_time": bprop_time,
            },
            loss=loss,
        )


@configurable(group="trainer", name="visual_relational_reasoning")
class VisualRelationalClassificationTrainer(ClassificationTrainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
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
