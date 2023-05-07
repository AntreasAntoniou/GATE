from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.metrics import accuracy_top_k

from ..decorators import collect_metrics, configurable
from ..utils import get_logger

logger = get_logger(__name__)


def get_dict_shapes(x):
    print(x)
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


class Trainer(object):
    def __init__(self):
        pass


@dataclass
class TrainerOutput:
    opt_loss: torch.Tensor
    global_step: int
    metrics: Dict[str, Any]
    phase_name: str
    experiment_tracker: Any = None


@dataclass
class StepOutput:
    output_metrics_dict: Dict
    loss: torch.Tensor
    accuracy: torch.Tensor
    accuracy_top_5: torch.Tensor


@configurable
class ClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = "step",
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.state_dict = {}

        if self.scheduler is not None:
            assert scheduler_interval in {"step"}
            self.scheduler_interval = scheduler_interval

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
            metrics |= overall_output_dict
        else:
            metrics = {}

        for key, value in metrics.items():
            self.state_dict.setdefault(key, []).append(value)

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

    @collect_metrics
    def start_training(
        self,
        global_step: int,
    ):
        self.state_dict = {}
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
            print(key, value)
            phase_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            phase_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return TrainerOutput(
            opt_loss=None,
            global_step=global_step,
            metrics=phase_metrics,
            phase_name="training",
            experiment_tracker=self.experiment_tracker,
        )
