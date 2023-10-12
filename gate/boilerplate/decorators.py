import functools
import importlib
import inspect
import pkgutil
import threading
from queue import PriorityQueue
from time import sleep
from typing import Any, Callable, Dict, Optional

import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from regex import P

import wandb
from gate.boilerplate.utils import get_logger
from gate.boilerplate.wandb_utils import (
    log_wandb_3d_volumes_and_masks,
    log_wandb_images,
    log_wandb_masks,
    visualize_video_with_labels,
)

logger = get_logger(__name__)


class BackgroundLogging(threading.Thread):
    def __init__(self, experiment_tracker: wandb, queue: PriorityQueue):
        super().__init__()
        self.experiment_tracker = experiment_tracker  # Experiment tracker
        self.queue = queue  # Priority queue passed in

    def run(self):
        while True:
            if not self.queue.empty():
                global_step, item = self.queue.get()
                (
                    metrics_dict,
                    phase_name,
                ) = item

                for metric_key, computed_value in metrics_dict.items():
                    if computed_value is not None:
                        value = (
                            computed_value.detach()
                            if isinstance(computed_value, torch.Tensor)
                            else computed_value
                        )

                        log_dict = {f"{phase_name}/{metric_key}": value}

                        if "seg_episode" in metric_key:
                            mask_dict = log_wandb_masks(
                                images=value["image"],
                                logits=value["logits"],
                                labels=value["label"],
                                label_idx_to_description=value[
                                    "label_idx_to_description"
                                ],
                                prefix=phase_name,
                            )
                            log_dict.update(mask_dict)

                        if "med_episode" in metric_key:
                            volume_dict = log_wandb_3d_volumes_and_masks(
                                volumes=value["image"],
                                logits=value["logits"],
                                labels=value["label"],
                                prefix=phase_name,
                            )
                            log_dict.update(volume_dict)

                        if "ae_episode" in metric_key:
                            image_dict = log_wandb_images(
                                images=value["image"],
                                reconstructions=value["recon"],
                                prefix=phase_name,
                            )
                            log_dict.update(image_dict)

                        if "video_episode" in metric_key:
                            video_dict = visualize_video_with_labels(
                                name=phase_name,
                                video=value["video"],
                                logits=value["logits"],
                                labels=value["label"],
                            )
                            log_dict.update(video_dict)

                        self.experiment_tracker.log(log_dict, step=global_step)


# Initialize a global priority queue to hold metrics to be logged
metrics_queue = PriorityQueue()
wandb_logging_thread = BackgroundLogging(
    experiment_tracker=wandb, queue=metrics_queue
)
wandb_logging_thread.start()


def configurable(
    group: str,
    name: str,
    defaults: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    A decorator for making functions configurable and setting default values for their arguments.

    Args:
        group (str): The group name under which the configurable function will be registered in the config store.
        name (str): The name of the configurable function in the config store.
        defaults (Optional[Dict[str, Any]], optional): A dictionary of default values for the function's arguments. Defaults to None.
    """

    def wrapper(func: Callable) -> Callable:
        func.__configurable__ = True
        func.__config_group__ = group
        func.__config_name__ = name

        def build_config(**kwargs):
            if defaults:
                kwargs = {**defaults, **kwargs}
            return builds(func, **kwargs)

        setattr(func, "__config__", build_config)
        return func

    return wrapper


def register_configurables(package_name: str) -> ConfigStore:
    """
    Registers all configurable functions in the specified package to the config store.

    Args:
        package_name (str): The name of the package containing the configurable functions.
    """

    config_store = ConfigStore.instance()

    package = importlib.import_module(package_name)
    prefix = package.__name__ + "."

    for _, module_name, _ in pkgutil.walk_packages(package.__path__, prefix):
        module = importlib.import_module(module_name)

        for name, obj in inspect.getmembers(module):
            if not (
                inspect.isfunction(obj) or inspect.isclass(obj)
            ):  # Skip if not a function or class
                continue
            if hasattr(obj, "__configurable__") and obj.__configurable__:
                group = obj.__config_group__
                name = obj.__config_name__

                config_store.store(
                    group=group,
                    name=name,
                    node=obj.__config__(populate_full_signature=True),
                )

    return config_store


def collect_metrics(func: Callable) -> Callable:
    def collect_metrics(
        metrics_dict: dict(),
        phase_name: str,
        experiment_tracker: Any,
        global_step: int,
    ) -> None:
        if experiment_tracker is None:
            experiment_tracker = wandb

        detached_metrics_dict = {}
        for metric_key, computed_value in metrics_dict.items():
            if computed_value is not None:
                value = (
                    computed_value.detach()
                    if isinstance(computed_value, torch.Tensor)
                    else computed_value
                )
                detached_metrics_dict[metric_key] = value

        metrics_queue.put((global_step, (detached_metrics_dict, phase_name)))

    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs):
        outputs = func(*args, **kwargs)
        collect_metrics(
            metrics_dict=outputs.metrics,
            phase_name=outputs.phase_name,
            experiment_tracker=outputs.experiment_tracker,
            global_step=outputs.global_step,
        )
        return outputs

    return wrapper_collect_metrics
