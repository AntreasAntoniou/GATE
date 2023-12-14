import functools
import importlib
import inspect
import logging
import pkgutil
import threading
from typing import Any, Callable, Dict, Optional

import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra_zen import builds

from gate.boilerplate.wandb_utils import (
    log_wandb_3d_volumes_and_masks,
    log_wandb_images,
    log_wandb_masks,
    visualize_video_with_labels,
)

logger = logging.getLogger(__name__)


class BackgroundLogging(threading.Thread):
    def __init__(
        self,
        experiment_tracker: wandb,
        global_step: int,
        phase_name: str,
        metrics_dict: Dict,
    ):
        super().__init__()
        self.experiment_tracker = experiment_tracker  # Experiment tracker
        self.global_step = global_step  # Global step
        self.phase_name = phase_name  # Phase name
        self.metrics_dict = metrics_dict  # Metrics dictionary

    def run(self):
        metrics_dict = self.metrics_dict
        phase_name = self.phase_name
        global_step = self.global_step

        for metric_key, computed_value in metrics_dict.items():
            if computed_value is not None:
                value = (
                    computed_value.detach()
                    if isinstance(computed_value, torch.Tensor)
                    else computed_value
                )

                log_dict = {
                    f"{phase_name}/{metric_key}": value,
                    "global_step": global_step,
                }

                # if "image_class_episode" in metric_key:
                #     image_dict = log_wandb_image_classification(
                #         images=value["image"],
                #         logits=value["logits"],
                #         labels=value["label"],
                #         prefix=phase_name,
                #     )
                #     log_dict.update(image_dict)

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

                self.experiment_tracker.log(log_dict)


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


def ensemble_marker(func):
    """
    Decorator to mark a function or method with an attribute '__used_in_ensemble__'.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    setattr(wrapper, "__used_in_ensemble__", True)  # Set attribute here
    return wrapper


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

    background_logging = BackgroundLogging(
        experiment_tracker, global_step, phase_name, detached_metrics_dict
    )
    background_logging.start()


def collect_metrics_mark(func: Callable) -> Callable:
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
