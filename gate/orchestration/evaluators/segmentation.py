from typing import Any, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics, configurable
from gate.boilerplate.utils import get_logger
from gate.orchestration.evaluators import EvaluatorOutput
from gate.orchestration.evaluators.classification import (
    ClassificationEvaluator,
)

logger = get_logger(__name__)


@configurable(group="evaluator", name="image_semantic_segmentation")
class ImageSemanticSegmentationEvaluator(ClassificationEvaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image",
            target_modality="image",
            model_selection_metric_name="mean_iou-epoch-mean",
            model_selection_metric_higher_is_better=True,
        )
        self.model = None

    def step(self, model, batch, global_step, accelerator: Accelerator):
        # start_time = time.time()

        if self.model is None:
            self.model = model

        output_dict = model.forward(batch)
        # logger.info(f"forward time: {time.time() - start_time}")
        output_dict = output_dict[self.target_modality][self.source_modality]

        loss = output_dict["loss"]

        if "logits" in output_dict:
            if self.starting_eval:
                height, width = output_dict["logits"].shape[-2:]
                output_dict["seg_episode"] = {
                    "image": F.interpolate(
                        batch["image"], size=(height, width)
                    ),
                    "logits": output_dict["logits"].argmax(dim=1).squeeze(1),
                    "label": batch["labels"].squeeze(1),
                    "label_idx_to_description": {
                        i: str(i)
                        for i in range(output_dict["logits"].shape[1])
                    },
                }
                self.starting_eval = False

            del output_dict["logits"]

        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                if isinstance(value, torch.Tensor):
                    self.current_epoch_dict[key].append(
                        value.detach().float().mean().cpu()
                    )

        return StepOutput(
            metrics=output_dict,
            loss=loss,
        )

    @collect_metrics
    def validation_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        output: EvaluatorOutput = super().validation_step(
            model, batch, global_step, accelerator
        )

        if "seg_episode" in output.metrics:
            seg_episode = output.metrics["seg_episode"]
            # ae_episode = output.metrics["ae_episode"]
            output.metrics = {"seg_episode": seg_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics
    def testing_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        output: EvaluatorOutput = super().testing_step(
            model, batch, global_step, accelerator
        )
        if "seg_episode" in output.metrics:
            seg_episode = output.metrics["seg_episode"]
            # ae_episode = output.metrics["ae_episode"]
            output.metrics = {"seg_episode": seg_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics
    def end_validation(self, global_step):
        evaluator_output: EvaluatorOutput = super().end_validation(global_step)
        iou_metrics = self.model.model.compute_across_set_iou()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_testing(self, global_step):
        evaluator_output: EvaluatorOutput = super().end_testing(global_step)
        iou_metrics = self.model.model.compute_across_set_iou()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )


@configurable(group="evaluator", name="medical_semantic_segmentation")
class MedicalSemanticSegmentationEvaluator(ClassificationEvaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image",
            target_modality="image",
            model_selection_metric_name="dice_loss-epoch-mean",
            model_selection_metric_higher_is_better=True,
        )
        self.model = None

    def step(self, model, batch, global_step, accelerator: Accelerator):
        if self.model is None:
            self.model = model

        output_dict = model.forward(batch)
        output_dict = output_dict[self.target_modality][self.source_modality]

        loss = output_dict["loss"]

        if "logits" in output_dict:
            if self.starting_eval:
                height, width = output_dict["logits"].shape[-2:]
                output_dict["med_seg_episode"] = {
                    "image": F.interpolate(
                        batch["image"], size=(height, width)
                    ),
                    "logits": output_dict["logits"].argmax(dim=1).squeeze(1),
                    "label": batch["labels"].squeeze(1),
                    "label_idx_to_description": {
                        i: str(i)
                        for i in range(output_dict["logits"].shape[1])
                    },
                }
                self.starting_eval = False

            del output_dict["logits"]

        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                if isinstance(value, torch.Tensor):
                    self.current_epoch_dict[key].append(
                        value.detach().float().mean().cpu()
                    )

        return StepOutput(
            metrics=output_dict,
            loss=loss,
        )

    @collect_metrics
    def validation_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        output: EvaluatorOutput = super().validation_step(
            model, batch, global_step, accelerator
        )

        if "med_seg_episode" in output.metrics:
            med_seg_episode = output.metrics["med_seg_episode"]
            output.metrics = {"med_seg_episode": med_seg_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics
    def testing_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        output: EvaluatorOutput = super().testing_step(
            model, batch, global_step, accelerator
        )
        if "med_seg_episode" in output.metrics:
            med_seg_episode = output.metrics["med_seg_episode"]
            output.metrics = {"med_seg_episode": med_seg_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics
    def end_validation(self, global_step):
        evaluator_output: EvaluatorOutput = super().end_validation(global_step)
        iou_metrics = self.model.model.compute_across_set_iou()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics
    def end_testing(self, global_step):
        evaluator_output: EvaluatorOutput = super().end_testing(global_step)
        iou_metrics = self.model.model.compute_across_set_iou()

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="testing",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )
