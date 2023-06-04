from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics


class Evaluator(ABC):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
        source_modality: Optional[str] = None,
        target_modality: Optional[str] = None,
        model_selection_metric_name: Optional[str] = None,
        model_selection_metric_higher_is_better: Optional[bool] = None,
    ):
        super().__init__()
        self.current_epoch_dict = defaultdict(list)
        self.per_epoch_metrics = defaultdict(list)
        self.experiment_tracker = experiment_tracker
        self.starting_eval = True
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.model_selection_metric_name = model_selection_metric_name
        self.model_selection_metric_higher_is_better = (
            model_selection_metric_higher_is_better
        )
        self.val_idx = 0
        self.test_idx = 0

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
        self, metric_name: str, higher_is_better: bool, kth_best: int = 1
    ):
        # Finds the best model based on the metric name,
        # and returns the global step and the metric value of that model
        metrics = self.per_epoch_metrics[metric_name]
        global_steps = self.per_epoch_metrics["global_step"]
        print(
            f"global_steps: {global_steps}, per_epoch_metrics: {self.per_epoch_metrics}, current_epoch_dict: {self.current_epoch_dict}"
        )

        if isinstance(metrics, List):
            if len(metrics) == 0:
                raise ValueError(
                    f"No epoch values found for {metric_name}, "
                    f"the available metrics are: {self.per_epoch_metrics.keys()}"
                )
            metrics = [torch.tensor(metric) for metric in metrics]
            metrics = torch.stack(metrics)

        metric_sorting = torch.argsort(torch.tensor(metrics))

        # if higher_is_better:
        #     best_metric_idx = metric_sorting[-kth_best:]
        # else:
        #     best_metric_idx = metric_sorting[:kth_best]

        best_global_step = list(
            set([global_steps[idx] for idx in metric_sorting])
        )
        best_metric = list(set([metrics[idx] for idx in metric_sorting]))

        if higher_is_better:
            return best_global_step[-kth_best:], best_metric[-kth_best:]
        else:
            return best_global_step[:kth_best], best_metric[:kth_best]

    @collect_metrics
    def start_validation(
        self,
        global_step: int,
    ):
        self.current_epoch_dict = defaultdict(list)
        self.starting_eval = True

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=self.current_epoch_dict,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def start_testing(
        self,
        global_step: int,
        prefix: Optional[str] = None,
    ):
        self.current_epoch_dict = defaultdict(list)
        self.starting_eval = True

        return EvaluatorOutput(
            global_step=global_step,
            phase_name=f"testing/{prefix}" if prefix else "testing",
            metrics=self.current_epoch_dict,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_validation(
        self,
        global_step: int,
    ):
        phase_metrics = {}
        for key, value in self.current_epoch_dict.items():
            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()
            self.per_epoch_metrics[f"{key}-epoch-mean"].append(
                phase_metrics[f"{key}-epoch-mean"]
            )
            self.per_epoch_metrics[f"{key}-epoch-std"].append(
                phase_metrics[f"{key}-epoch-std"]
            )

        self.per_epoch_metrics["global_step"].append(global_step)

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
        prefix: Optional[str] = None,
    ):
        phase_metrics = {}
        for key, value in self.current_epoch_dict.items():
            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()
            self.per_epoch_metrics[f"{key}-epoch-mean"].append(
                phase_metrics[f"{key}-epoch-mean"]
            )
            self.per_epoch_metrics[f"{key}-epoch-std"].append(
                phase_metrics[f"{key}-epoch-std"]
            )

        self.per_epoch_metrics["global_step"].append(global_step)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name=f"testing/{prefix}" if prefix else "testing",
            metrics=phase_metrics,
            experiment_tracker=self.experiment_tracker,
        )


@dataclass
class EvaluatorOutput:
    global_step: int
    metrics: Dict
    phase_name: str
    experiment_tracker: Any = None
