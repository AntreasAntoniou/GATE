from ast import Dict
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.metrics import accuracy_top_k

from gate.boilerplate.decorators import collect_metrics
from gate.boilerplate.decorators import configurable
from gate.trainers.classification import StepOutput
from gate.boilerplate.utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    return (
        {
            key: value.shape if isinstance(value, torch.Tensor) else len(value)
            for key, value in x.items()
        }
        if isinstance(x, dict)
        else get_dict_shapes(x.__dict__)
    )


class Evaluator(object):
    def __init__(self):
        pass


@dataclass
class EvaluatorOutput:
    global_step: int
    metrics: Dict
    phase_name: str
    experiment_tracker: Any = None


@configurable(group="evaluator", name="classification")
class ClassificationEvaluator(Evaluator):
    def step(self, model, batch, global_step, accelerator: Accelerator):
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

        return StepOutput(
            output_metrics_dict=output_metrics_dict,
            loss=loss,
            accuracy=accuracy,
            accuracy_top_5=accuracy_top_5,
        )

    @torch.inference_mode()
    def validation_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ):
        model.eval()

        with torch.no_grad():
            overall_loss = []
            overall_accuracy = []
            overall_accuracy_top_5 = []
            overall_output_dict = {}

            step_output: StepOutput = self.step(
                model=model,
                batch=batch,
                global_step=global_step,
                accelerator=accelerator,
            )

            keys = list(step_output.output_metrics_dict.keys())
            for key in keys:
                if "loss" not in key and "accuracy" not in key:
                    del step_output.output_metrics_dict[key]

            if step_output is not None:
                overall_output_dict |= step_output.output_metrics_dict
                overall_loss.append(step_output.loss)
                overall_accuracy.append(step_output.accuracy)
                overall_accuracy_top_5.append(step_output.accuracy_top_5)

            if len(overall_loss) > 0:
                metrics = {
                    "accuracy_top_1": torch.mean(
                        torch.stack(overall_accuracy)
                    ),
                    "accuracy_top_5": torch.mean(
                        torch.stack(overall_accuracy_top_5)
                    ),
                    "loss": torch.mean(torch.stack(overall_loss)),
                }
                metrics |= overall_output_dict
            else:
                metrics = {}

            for key, value in metrics.items():
                self.state_dict.setdefault(key, []).append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
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
    ):
        model.eval()

        with torch.no_grad():
            overall_loss = []
            overall_accuracy = []
            overall_accuracy_top_5 = []
            overall_output_dict = {}

            step_output: StepOutput = self.step(
                model=model,
                batch=batch,
                global_step=global_step,
                accelerator=accelerator,
            )

            keys = list(step_output.output_metrics_dict.keys())
            for key in keys:
                if "loss" not in key and "accuracy" not in key:
                    del step_output.output_metrics_dict[key]

            if step_output is not None:
                overall_output_dict |= step_output.output_metrics_dict
                overall_loss.append(step_output.loss)
                overall_accuracy.append(step_output.accuracy)
                overall_accuracy_top_5.append(step_output.accuracy_top_5)

            if len(overall_loss) > 0:
                metrics = {
                    "accuracy_top_1": torch.mean(
                        torch.stack(overall_accuracy)
                    ),
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

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="test",
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )
