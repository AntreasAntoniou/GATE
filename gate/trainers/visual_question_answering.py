from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics
from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.metrics.vqa_eval import AnswerData, VQAItem, vqa_metric
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
    vqa_score: Optional[float] = None


# TODO: Make sure it's easier for user, with autocheck.


@configurable(group="trainer", name="visual_question_answering")
class VQATrainer(Trainer):
    def get_optimizer(self):
        return self.optimizer

    def step(self, model, batch, global_step, accelerator: Accelerator):
        output_dict = model.forward(batch)["text"]["image_text"]
        loss = output_dict["loss"]

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

        output_metrics_dict = {
            "loss": loss,
            "vqa_score": torch.mean(
                torch.tensor(result["overall"])
            ),  # Use the mean VQA score here
        }

        accelerator.backward(loss)

        return StepOutput(
            output_metrics_dict=output_metrics_dict,
            loss=loss,
            vqa_score=output_metrics_dict[
                "vqa_score"
            ],  # Add the VQA score here
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
        overall_vqa_score = []
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
            overall_vqa_score.append(step_output.vqa_score)

        self.optimizer.step()

        keys = list(step_output.output_metrics_dict.keys())
        for key in keys:
            if (
                "loss" not in key and "vqa_score" not in key
            ):  # Consider vqa_score now
                del step_output.output_metrics_dict[key]

        if len(overall_loss) > 0:
            metrics = {
                "vqa_score": torch.mean(
                    torch.stack(overall_vqa_score)
                ),  # Include vqa_score here
                "loss": torch.mean(torch.stack(overall_loss)),
            }
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
