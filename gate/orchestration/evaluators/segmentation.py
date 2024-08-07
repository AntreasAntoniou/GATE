import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import collect_metrics_mark, configurable
from gate.orchestration.evaluators import EvaluatorOutput
from gate.orchestration.evaluators.classification import (
    ClassificationEvaluator,
    StepOutput,
)
from gate.orchestration.trainers.segmentation import (
    integrate_output_list,
    sub_batch_generator,
)

logger = logging.getLogger(__name__)


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
            model_selection_metric_name="mIoU",
            model_selection_metric_higher_is_better=True,
        )
        self.model = None

    def step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
        prefix: Optional[str] = None,
    ):
        if self.model is None:
            self.model = model

        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}-"

        output_dict = model.forward(batch)
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
                    self.current_epoch_dict[f"{prefix}{key}"].append(
                        value.detach().float().mean().cpu()
                    )

        return StepOutput(
            metrics=output_dict,
            loss=loss,
        )

    @collect_metrics_mark
    def validation_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        model = model.eval()
        output: EvaluatorOutput = super().validation_step(
            model, batch, global_step, accelerator
        )

        if "seg_episode" in output.metrics:
            seg_episode = output.metrics["seg_episode"]

            output.metrics = {"seg_episode": seg_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics_mark
    def testing_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
        prefix: Optional[str] = None,
    ):
        model = model.eval()
        output: EvaluatorOutput = super().testing_step(
            model, batch, global_step, accelerator, prefix=prefix
        )
        if "seg_episode" in output.metrics:
            seg_episode = output.metrics["seg_episode"]

            output.metrics = {"seg_episode": seg_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics_mark
    def end_validation(self, global_step):
        evaluator_output: EvaluatorOutput = super().end_validation(global_step)
        iou_metrics = self.model.model.compute_across_set_metrics()

        for key, value in iou_metrics.items():
            self.current_epoch_dict[key].append(value)
            self.per_epoch_metrics[key].append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics_mark
    def end_testing(
        self,
        global_step,
        model: Optional[nn.Module] = None,
        prefix: Optional[str] = None,
    ):
        if model is None:
            model = self.model

        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}-"

        evaluator_output: EvaluatorOutput = super().end_testing(
            global_step, prefix=prefix, model=model
        )
        iou_metrics = model.model.compute_across_set_metrics()

        for key, value in iou_metrics.items():
            self.current_epoch_dict[f"{prefix}{key}"].append(value)
            self.per_epoch_metrics[f"{prefix}{key}"].append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name=f"testing/{prefix}" if prefix else "testing",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )


@configurable(group="evaluator", name="medical_semantic_segmentation")
class MedicalSemanticSegmentationEvaluator(ClassificationEvaluator):
    def __init__(
        self,
        experiment_tracker: Optional[Any] = None,
        sub_batch_size: int = 20,
    ):
        super().__init__(
            experiment_tracker,
            source_modality="image",
            target_modality="image",
            model_selection_metric_name="dice_loss-epoch-mean",
            model_selection_metric_higher_is_better=False,
        )
        self.model = None
        self.sub_batch_size = sub_batch_size

    def collect_segmentation_episode(self, output_dict, global_step, batch):
        if "logits" in output_dict:
            if self.starting_eval:
                b, s, c = batch["image"].shape[:3]
                height, width = output_dict["logits"].shape[-2:]

                image = F.interpolate(
                    input=batch["image"].view(-1, *batch["image"].shape[2:]),
                    size=(height, width),
                )
                image = image.reshape(b, s, c, height, width)
                logits = output_dict["logits"].argmax(dim=1).squeeze(1)
                logits = logits.reshape(b, s, *logits.shape[1:])
                label = batch["labels"].squeeze(1)

                output_dict["med_episode"] = {
                    "image": image,
                    "logits": logits,
                    "label": label,
                    "label_idx_to_description": {
                        i: str(i)
                        for i in range(output_dict["logits"].shape[2])
                    },
                }

                self.starting_eval = False
            del output_dict["logits"]
        return output_dict

    def step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
        prefix: Optional[str] = None,
    ):
        if self.model is None:
            self.model = model

        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}-"

        output_list = []
        for sub_batch in sub_batch_generator(batch, self.sub_batch_size):
            output_dict = model.forward(sub_batch)
            output_dict = output_dict[self.target_modality][
                self.source_modality
            ]

            loss = output_dict["loss"]

            for key, value in output_dict.items():
                if "loss" in key or "iou" in key or "accuracy" in key:
                    if isinstance(value, torch.Tensor):
                        self.current_epoch_dict[f"{prefix}{key}"].append(
                            value.detach().float().mean().cpu()
                        )
            output_list.append(output_dict)

        output_dict = integrate_output_list(output_list)
        output_dict = self.collect_segmentation_episode(
            output_dict=output_dict, global_step=global_step, batch=batch
        )

        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                output_dict[key] = value.mean()

        return StepOutput(
            metrics=output_dict,
            loss=loss,
        )

    @collect_metrics_mark
    def validation_step(
        self, model, batch, global_step, accelerator: Accelerator
    ):
        model = model.eval()
        output: EvaluatorOutput = super().validation_step(
            model, batch, global_step, accelerator
        )

        if "med_episode" in output.metrics:
            med_episode = output.metrics["med_episode"]
            output.metrics = {"med_episode": med_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics_mark
    def testing_step(
        self,
        model,
        batch,
        global_step,
        accelerator: Accelerator,
        prefix: Optional[str] = None,
    ):
        model = model.eval()
        output: EvaluatorOutput = super().testing_step(
            model, batch, global_step, accelerator, prefix=prefix
        )
        if "med_episode" in output.metrics:
            med_episode = output.metrics["med_episode"]
            output.metrics = {"med_episode": med_episode}
        else:
            output.metrics = {}
        return output

    @collect_metrics_mark
    def end_validation(self, global_step):
        evaluator_output: EvaluatorOutput = super().end_validation(global_step)
        iou_metrics = self.model.model.compute_across_set_metrics()

        for key, value in iou_metrics.items():
            self.current_epoch_dict[key].append(value)
            self.per_epoch_metrics[key].append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name="validation",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )

    @collect_metrics_mark
    def end_testing(
        self,
        global_step,
        model: Optional[nn.Module] = None,
        prefix: Optional[str] = None,
    ):
        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}-"

        if model is None:
            model = self.model

        evaluator_output: EvaluatorOutput = super().end_testing(
            global_step, prefix=prefix, model=model
        )
        iou_metrics = model.model.compute_across_set_metrics()

        for key, value in iou_metrics.items():
            self.current_epoch_dict[f"{prefix}{key}"].append(value)
            self.per_epoch_metrics[f"{prefix}{key}"].append(value)

        return EvaluatorOutput(
            global_step=global_step,
            phase_name=f"testing/{prefix}" if prefix else "testing",
            metrics=evaluator_output.metrics | iou_metrics,
            experiment_tracker=self.experiment_tracker,
        )
