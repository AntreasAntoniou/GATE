from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics


class Evaluator(ABC):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
        source_modality: Optional[str] = None,
        target_modality: Optional[str] = None,
    ):
        super().__init__()
        self.state_dict = {}
        self.epoch_metrics = defaultdict(list)
        self.experiment_tracker = experiment_tracker
        self.starting_eval = True
        self.source_modality = source_modality
        self.target_modality = target_modality

    @abstractmethod
    def step(self, model, batch, global_step):
        pass

    @abstractmethod
    def validation_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        pass

    @abstractmethod
    def testing_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        pass

    def get_best_model_global_step_and_metric(
        self, metric_name: str, higher_is_better: bool
    ):
        # Finds the best model based on the metric name,
        # and returns the global step and the metric value of that model

        metrics = self.epoch_metrics[metric_name]
        global_steps = self.epoch_metrics["global_step"]
        print(metrics.shape)

        if higher_is_better:
            best_metric_idx = torch.argmax(torch.tensor(metrics))
        else:
            best_metric_idx = torch.argmin(torch.tensor(metrics))

        best_global_step = global_steps[best_metric_idx]
        best_metric = metrics[best_metric_idx]

        return best_global_step, best_metric

    @collect_metrics
    def start_validation(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        self.starting_eval = True
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
        self.starting_eval = True
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
            if "loss" not in key and "vqa_score" not in key:
                continue

            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()
            self.epoch_metrics[f"{key}-epoch-mean"].append(
                phase_metrics[f"{key}-epoch-mean"]
            )
            self.epoch_metrics[f"{key}-epoch-std"].append(
                phase_metrics[f"{key}-epoch-std"]
            )

        self.epoch_metrics["global_step"].append(global_step)

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
            if "loss" not in key and "vqa_score" not in key:
                continue
            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=phase_metrics,
            experiment_tracker=self.experiment_tracker,
        )


@dataclass
class EvaluatorOutput:
    global_step: int
    metrics: Dict
    phase_name: str
    experiment_tracker: Any = None
