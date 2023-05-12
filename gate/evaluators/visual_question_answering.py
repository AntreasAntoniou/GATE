from ast import Dict
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics
from gate.boilerplate.decorators import configurable
from gate.evaluators import Evaluator
from gate.metrics.vqa_eval import AnswerData, VQAItem, vqa_metric
from gate.models.core import GATEModel
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


@dataclass
class EvaluatorOutput:
    global_step: int
    metrics: Dict
    phase_name: str
    experiment_tracker: Any = None


@configurable(group="evaluator", name="visual_question_answering")
class VQAEvaluator(Evaluator):
    def step(
        self,
        model: GATEModel,
        batch: Dict,
        global_step: int,
        accelerator: Accelerator,
    ):
        output_dict = model.forward(batch)
        loss = output_dict["loss"]

        # Generate answers and get the ground truth
        predicted_answers = model.generate_text(batch)
        ground_truth_answers = batch[
            "answers"
        ]  # Assuming this is where the true answers are

        # Prepare data for VQA evaluation
        vqa_data = {
            question_id: VQAItem(
                answers=ground_truth_answers[i],
                image_id=batch["image_ids"][i],
                question=batch["questions"][i],
                question_id=batch["question_ids"][i],
                question_type=None,
                answer_type=None,
            )
            for i, question_id in enumerate(batch["question_ids"])
        }
        vqa_predictions = {
            question_id: AnswerData(answer=predicted_answer)
            for question_id, predicted_answer in zip(
                batch["question_ids"], predicted_answers
            )
        }

        # Run the evaluation
        result = vqa_metric(vqa_data, vqa_predictions)

        output_metrics_dict = {
            "loss": loss,
            "vqa_score": torch.mean(
                torch.tensor(result["overall"])
            ),  # Use the mean VQA score here
        }

        keys = list(output_metrics_dict.keys())
        for key in keys:
            if "loss" not in key and "vqa_score" not in key:
                del output_dict[key]

        return StepOutput(
            output_metrics_dict=output_metrics_dict,
            loss=loss,
            vqa_score=output_metrics_dict[
                "vqa_score"
            ],  # Add the VQA score here
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
            overall_vqa_score = []  # List to hold all VQA scores
            overall_output_dict = {}

            step_output: StepOutput = self.step(
                model=model,
                batch=batch,
                global_step=global_step,
                accelerator=accelerator,
            )

            keys = list(step_output.output_metrics_dict.keys())
            for key in keys:
                if (
                    "loss" not in key and "vqa_score" not in key
                ):  # Consider vqa_score now
                    del step_output.output_metrics_dict[key]

            if step_output is not None:
                overall_output_dict |= step_output.output_metrics_dict
                overall_loss.append(step_output.loss)
                overall_vqa_score.append(
                    step_output.vqa_score
                )  # Add the VQA score here

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
            overall_vqa_score = []  # List to hold all VQA scores
            overall_output_dict = {}

            step_output: StepOutput = self.step(
                model=model,
                batch=batch,
                global_step=global_step,
                accelerator=accelerator,
            )

            keys = list(step_output.output_metrics_dict.keys())
            for key in keys:
                if (
                    "loss" not in key and "vqa_score" not in key
                ):  # Consider vqa_score now
                    del step_output.output_metrics_dict[key]

            if step_output is not None:
                overall_output_dict |= step_output.output_metrics_dict
                overall_loss.append(step_output.loss)
                overall_vqa_score.append(
                    step_output.vqa_score
                )  # Add the VQA score here

            if len(overall_loss) > 0:
                metrics = {
                    "vqa_score": torch.mean(
                        torch.stack(overall_vqa_score)
                    ),  # Include vqa_score here
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
