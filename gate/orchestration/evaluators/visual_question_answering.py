from ast import Dict
from dataclasses import dataclass
from typing import Any, Optional

import torch
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics, configurable
from gate.boilerplate.utils import get_logger
from gate.orchestration.evaluators import Evaluator
from gate.metrics.vqa_eval import vqa_metric
from gate.models.core import GATEModel
from gate.orchestration.trainers import log_data_to_wandb_table
from gate.orchestration.trainers.classification import StepOutput

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


@dataclass
class EvaluatorOutput:
    global_step: int
    metrics: Dict
    phase_name: str
    experiment_tracker: Any = None


@dataclass
class StepOutput:
    output_metrics_dict: Dict
    loss: torch.Tensor


@configurable(group="evaluator", name="visual_question_answering")
class VQAEvaluator(Evaluator):
    def step(
        self,
        model: GATEModel,
        batch: Dict,
        global_step: int,
        accelerator: Accelerator,
        phase_name: Optional[str] = None,
    ):
        output_dict = model.forward(batch)["text"]["image_text"]
        loss = output_dict["loss"]

        output_metrics = {"loss": loss}

        if torch.rand(1) > 0.80:
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
        if self.starting_eval:
            log_data_to_wandb_table(
                questions=questions,
                answers=ground_truth_answers,
                predicted_answers=predicted_answers,
                global_step=global_step,
                phase_name=phase_name,
            )
            self.starting_eval = False

        return {"vqa_score": torch.mean(torch.tensor(result["overall"]))}

    @collect_metrics
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
            step_output: StepOutput = self.step(
                model=model,
                batch=batch,
                global_step=global_step,
                accelerator=accelerator,
                phase_name="validation",
            )

            for key, value in step_output.output_metrics_dict.items():
                self.state_dict.setdefault(key, []).append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=step_output.output_metrics_dict,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
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
            step_output: StepOutput = self.step(
                model=model,
                batch=batch,
                global_step=global_step,
                accelerator=accelerator,
                phase_name="validation",
            )

            for key, value in step_output.output_metrics_dict.items():
                self.state_dict.setdefault(key, []).append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="test",
            metrics=step_output.output_metrics_dict,
            experiment_tracker=self.experiment_tracker,
        )
