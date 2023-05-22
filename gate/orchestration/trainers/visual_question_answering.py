from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics, configurable
from gate.boilerplate.utils import get_logger
from gate.metrics.vqa_eval import vqa_metric
from gate.orchestration.trainers import (
    Trainer,
    TrainerOutput,
    log_data_to_wandb_table,
)

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


# TODO: Make sure it's easier for user, with autocheck.


@configurable(group="trainer", name="visual_question_answering")
class VQATrainer(Trainer):
    def get_optimizer(self):
        return self.optimizer

    def step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
        phase_name: Optional[str] = None,
    ):
        output_dict = model.forward(batch)["text"]["image_text"]
        loss = output_dict["loss"]

        output_metrics = {"loss": loss}

        accelerator.backward(loss)

        if torch.rand(1) > 0.95:
            metrics = self.sample_answers_and_compute_vqa_score(
                model=model,
                batch=batch,
                global_step=global_step,
                phase_name=phase_name,
            )
            output_metrics |= metrics

        return StepOutput(
            output_metrics_dict=output_metrics,
            loss=loss,
        )

    def sample_answers_and_compute_vqa_score(
        self, model, batch, global_step, phase_name
    ):
        with torch.no_grad():
            # Generate answers and get the ground truth
            predicted_answers = model.model.generate_text(**batch)

        ground_truth_answers = batch["text"][
            "answer_original"
        ]  # Assuming this is where the true answers are

        questions = batch["text"]["question_original"]

        # Run the evaluation
        result = vqa_metric(
            answers=ground_truth_answers, predicted_answers=predicted_answers
        )
        if self.starting_train:
            log_data_to_wandb_table(
                questions=questions,
                answers=ground_truth_answers,
                predicted_answers=predicted_answers,
                global_step=global_step,
                phase_name=phase_name,
            )
            self.starting_train = False

        return {"vqa_score": torch.mean(torch.tensor(result["overall"]))}

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        model.train()

        overall_output_dict = {}

        self.optimizer.zero_grad()

        step_output: StepOutput = self.step(
            model=model,
            batch=batch,
            global_step=global_step,
            accelerator=accelerator,
            phase_name="training",
        )
        overall_output_dict |= step_output.output_metrics_dict

        self.optimizer.step()
        self.scheduler.step(global_step)

        overall_output_dict["lr"] = self.optimizer.param_groups[0]["lr"]

        return TrainerOutput(
            phase_name="training",
            opt_loss=overall_output_dict["loss"],
            global_step=global_step,
            metrics=overall_output_dict,
            experiment_tracker=self.experiment_tracker,
        )
