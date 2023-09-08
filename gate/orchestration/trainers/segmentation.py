import time
from typing import Any, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import get_logger
from gate.orchestration.trainers.classification import ClassificationTrainer

logger = get_logger(__name__)


@configurable(group="trainer", name="image_semantic_segmentation")
class ImageSemanticSegmentationTrainer(ClassificationTrainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = "step",
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            optimizer,
            scheduler,
            scheduler_interval,
            experiment_tracker,
            source_modality="image",
            target_modality="image",
        )

    def step(self, model, batch, global_step, accelerator: Accelerator):
        start_time = time.time()
        output_dict = model.forward(batch)[self.target_modality][
            self.source_modality
        ]
        logger.debug(f"Forward time {time.time() - start_time}")

        loss = output_dict["loss"]

        start_time = time.time()
        accelerator.backward(loss)
        logger.debug(f"Backward time {time.time() - start_time}")

        if "logits" in output_dict:
            if global_step % 100 == 0:
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

            del output_dict["logits"]

        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                if isinstance(value, torch.Tensor):
                    self.current_epoch_dict[key].append(value.detach().cpu())

        return StepOutput(
            output_metrics_dict=output_dict,
            loss=loss,
        )


def sub_batch_generator(batch_dict, sub_batch_size):
    """
    Generator function to yield sub-batches from a given batch dictionary.

    Parameters:
    - batch_dict (dict): Dictionary containing original batch data. Each key maps to a tensor or list.
    - sub_batch_size (int): Size of each sub-batch to be returned.

    Yields:
    - dict: Dictionary containing sub-batch data.
    """
    batch_size = None

    # Validate input and get original batch size
    for key, value in batch_dict.items():
        # print(f"key: {key}, value.shape: {value.shape}")
        if batch_size is None:
            batch_size = value.shape[0] * value.shape[1]
        elif batch_size != value.shape[0] * value.shape[1]:
            raise ValueError(
                f"Batch sizes for different keys in batch_dict must be the same. Mismatch at key: {key}"
            )

    # Generate and yield sub-batches
    for start_idx in range(0, batch_size, sub_batch_size):
        end_idx = min(start_idx + sub_batch_size, batch_size)
        sub_batch = {}

        for key, value in batch_dict.items():
            sub_batch[key] = value.reshape(-1, *value.shape[2:])[
                start_idx:end_idx
            ]

        yield sub_batch


def integrate_output_list(output_list):
    def accumulate_outputs(output_accumulator, output_dict):
        for key, value in output_dict.items():
            if isinstance(value, dict):
                if key not in output_accumulator:
                    output_accumulator[key] = {}
                accumulate_outputs(output_accumulator[key], value)
            else:
                if isinstance(value, torch.Tensor):
                    if key not in output_accumulator:
                        output_accumulator[key] = []
                    output_accumulator[key].append(value)

    output_accumulator = {}
    for output_dict in output_list:
        accumulate_outputs(output_accumulator, output_dict)

    def concatenate_tensors(nested_dict):
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                concatenate_tensors(value)
            else:
                if isinstance(value, list) and len(value) > 0:
                    if value[0].dim() == 0:
                        # Handle scalar tensors
                        nested_dict[key] = torch.cat(
                            [v.unsqueeze(0) for v in value], dim=0
                        )
                    else:
                        # Handle non-scalar tensors
                        nested_dict[key] = torch.cat(value, dim=0)

    concatenate_tensors(output_accumulator)
    return output_accumulator


@configurable(group="trainer", name="medical_semantic_segmentation")
class MedicalSemanticSegmentationTrainer(ClassificationTrainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = "step",
        experiment_tracker: Optional[Any] = None,
    ):
        super().__init__(
            optimizer,
            scheduler,
            scheduler_interval,
            experiment_tracker,
            source_modality="image",
            target_modality="image",
        )

    def step(self, model, batch, global_step, accelerator: Accelerator):
        start_time = time.time()
        output_dict = model.forward(batch)[self.target_modality][
            self.source_modality
        ]

        logger.debug(f"Forward time {time.time() - start_time}")

        loss = output_dict["loss"]

        start_time = time.time()
        accelerator.backward(loss)
        logger.debug(f"Backward time {time.time() - start_time}")

        if "logits" in output_dict:
            if global_step % 100 == 0:
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

            del output_dict["logits"]

        for key, value in output_dict.items():
            if "loss" in key or "iou" in key or "accuracy" in key:
                if isinstance(value, torch.Tensor):
                    self.current_epoch_dict[key].append(value.detach().cpu())

        return StepOutput(
            output_metrics_dict=output_dict,
            loss=loss,
        )