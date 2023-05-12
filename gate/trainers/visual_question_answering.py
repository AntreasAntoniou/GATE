from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.metrics import accuracy_top_k

from gate.boilerplate.decorators import collect_metrics
from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.trainers import Trainer, TrainerOutput

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
    accuracy: torch.Tensor
    accuracy_top_5: torch.Tensor


@configurable(group="trainer", name="classification")
class VQATrainer(Trainer):
    def get_optimizer(self):
        return self.optimizer

    def step(self, model, batch, global_step, accelerator: Accelerator):
        # print({key: value.shape for key, value in batch.items()})
        output_dict = model.forward(batch)
        loss = F.cross_entropy(output_dict["image"]["image"], batch["labels"])
        accuracy = accuracy_top_k(
            logits=output_dict["image"]["image"], labels=batch["labels"], k=1
        )
        accuracy_top_5 = accuracy_top_k(
            logits=output_dict["image"]["image"], labels=batch["labels"], k=5
        )
        output_metrics_dict = {
            "loss": loss,
            "accuracy_top_1": accuracy,
            "accuracy_top_5": accuracy_top_5,
        }
        keys = list(output_metrics_dict.keys())

        for key in keys:
            if "loss" not in key and "accuracy" not in key:
                del output_dict[key]

        accelerator.backward(loss)

        return StepOutput(
            output_metrics_dict=output_dict,
            loss=loss,
            accuracy=accuracy,
            accuracy_top_5=accuracy_top_5,
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

        overall_loss = []
        overall_accuracy = []
        overall_accuracy_top_5 = []
        overall_output_dict = {}

        self.optimizer.zero_grad()

        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
        )

        if step_output is not None:
            overall_output_dict |= step_output.output_metrics_dict
            overall_loss.append(step_output.loss)
            overall_accuracy.append(step_output.accuracy)
            overall_accuracy_top_5.append(step_output.accuracy_top_5)

        self.optimizer.step()

        if len(overall_loss) > 0:
            metrics = {
                "accuracy_top_1": torch.mean(torch.stack(overall_accuracy)),
                "accuracy_top_5": torch.mean(
                    torch.stack(overall_accuracy_top_5)
                ),
                "loss": torch.mean(torch.stack(overall_loss)),
            }

            for key, value in metrics.items():
                self.state_dict.setdefault(key, []).append(value)

            metrics |= overall_output_dict
        else:
            metrics = {}

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return TrainerOutput(
            phase_name="training",
            opt_loss=torch.mean(torch.stack(overall_loss))
            if len(overall_loss) > 0
            else None,
            global_step=global_step,
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )
