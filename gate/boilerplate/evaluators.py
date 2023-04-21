from ast import Dict
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from accelerate import Accelerator

from gate.boilerplate.metrics import accuracy_top_k

from .decorators import collect_metrics
from .trainers import StepOutput
from .utils import get_logger

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


import torch.nn.functional as F


class ClassificationEvaluator(Evaluator):
    def __init__(self, experiment_tracker: Optional[Any] = None):
        super().__init__()
        self.state_dict = {}
        self.experiment_tracker = experiment_tracker

    def step(self, model, batch, global_step, accelerator: Accelerator):
        output_dict = model.forward(batch)
        loss = F.cross_entropy(output_dict["logits"], batch["labels"])
        accuracy = accuracy_top_k(output_dict["logits"], batch["labels"], k=1)
        accuracy_top_5 = accuracy_top_k(
            output_dict["logits"], batch["labels"], k=5
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
    def test_step(
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
            phase_name="test",
            metrics=metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def start_validation(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=self.state_dict,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def start_testing(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=self.state_dict,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_validation(
        self,
        global_step: int,
    ):
        phase_metrics = {}
        for key, value in self.state_dict.items():
            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=phase_metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_testing(
        self,
        global_step: int,
    ):
        phase_metrics = {}
        for key, value in self.state_dict.items():
            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=phase_metrics,
            experiment_tracker=self.experiment_tracker,
        )
