from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
import wandb

from gate.boilerplate.decorators import collect_metrics


@dataclass
class TrainerOutput:
    opt_loss: torch.Tensor
    global_step: int
    metrics: Dict[str, Any]
    phase_name: str
    experiment_tracker: Any = None


class Trainer(ABC):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = "step",
        experiment_tracker: Optional[Any] = None,
        source_modality: Optional[str] = None,
        target_modality: Optional[str] = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.state_dict = {}
        self.starting_train = True
        self.souce_modality = source_modality
        self.target_modality = target_modality

        if self.scheduler is not None:
            assert scheduler_interval in {"step"}
            self.scheduler_interval = scheduler_interval

    @abstractmethod
    def step(self, model, batch, global_step):
        pass

    @abstractmethod
    def training_step(
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

        if higher_is_better:
            best_metric_idx = torch.argmax(torch.tensor(metrics))
        else:
            best_metric_idx = torch.argmin(torch.tensor(metrics))

        best_global_step = global_steps[best_metric_idx]
        best_metric = metrics[best_metric_idx]

        return best_global_step, best_metric

    @collect_metrics
    def start_training(
        self,
        global_step: int,
    ):
        self.state_dict = {}
        self.starting_train = True
        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics={},
            phase_name="training",
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_training(
        self,
        global_step: int,
    ):
        phase_metrics = {}
        for key, value in self.state_dict.items():
            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics=phase_metrics,
            phase_name="training",
            experiment_tracker=self.experiment_tracker,
        )


def log_data_to_wandb_table(
    questions: list,
    answers: list,
    predicted_answers: list,
    global_step: list,
    phase_name: str,
):
    # Initialize a table
    table = wandb.Table(
        columns=["Global Step", "Question", "Answers", "Predicted Answer"]
    )

    # Zip the lists together and add each data point to the table
    for question, answer, predicted_answer in zip(
        questions, answers, predicted_answers
    ):
        table.add_data(global_step, question, answer, predicted_answer)

    # Log the table
    wandb.log({f"{phase_name}/qa_table": table})
