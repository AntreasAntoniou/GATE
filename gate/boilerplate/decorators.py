import functools
import importlib
import inspect
import pkgutil
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate
from rich import print
from traitlets import default

from gate.boilerplate.utils import (
    get_logger,
    log_wandb_images,
    log_wandb_masks,
)

logger = get_logger(__name__)


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


def register_configurables(
    package_name: str,
):
    """
    Registers all configurable functions in the specified package to the config store.

    Args:
        package_name (str): The name of the package containing the configurable functions.
    """

    config_store = ConfigStore.instance()

    package = importlib.import_module(package_name)
    prefix = package.__name__ + "."

    def _process_module(module_name):
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module):
            if hasattr(obj, "__configurable__") and obj.__configurable__:
                if hasattr(obj, "__config_group__") and hasattr(
                    obj, "__config_name__"
                ):
                    group = obj.__config_group__
                    name = obj.__config_name__

                    config_store.store(
                        group=group,
                        name=name,
                        node=obj.__config__(populate_full_signature=True),
                    )
                else:
                    logger.debug(
                        f"Excluding {name} from config store, as it does not have a group or name."
                    )

    for importer, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, prefix
    ):
        if is_pkg:
            register_configurables(module_name)
        else:
            _process_module(module_name)


class BackgroundLogging(threading.Thread):
    def __init__(
        self, experiment_tracker, metrics_dict, phase_name, global_step
    ):
        super().__init__()
        self.experiment_tracker = experiment_tracker
        self.metrics_dict = metrics_dict
        self.phase_name = phase_name
        self.global_step = global_step

    def run(self):
        for metric_key, computed_value in self.metrics_dict.items():
            if computed_value is not None:
                value = (
                    computed_value.detach()
                    if isinstance(computed_value, torch.Tensor)
                    else computed_value
                )
                # print(f"logging {metric_key}")
                if "seg_episode" in metric_key:
                    # print("logging seg episode")
                    seg_episode = value

                    log_wandb_masks(
                        experiment_tracker=self.experiment_tracker,
                        images=seg_episode["image"],
                        logits=seg_episode["logits"],
                        labels=seg_episode["label"],
                        label_idx_to_description=seg_episode[
                            "label_idx_to_description"
                        ],
                        global_step=self.global_step,
                    )
                    continue
                if "ae_episode" in metric_key:
                    # print("logging ae episode")
                    ae_episode = value
                    log_wandb_images(
                        experiment_tracker=self.experiment_tracker,
                        images=ae_episode["image"],
                        reconstructions=ae_episode["recon"],
                        global_step=self.global_step,
                    )
                    continue

                self.experiment_tracker.log(
                    {f"{self.phase_name}/{metric_key}": value},
                    step=self.global_step,
                )


def collect_metrics(func: Callable) -> Callable:
    def collect_metrics(
        metrics_dict: dict(),
        phase_name: str,
        experiment_tracker: Any,
        global_step: int,
    ) -> None:
        detached_metrics_dict = {}
        for metric_key, computed_value in metrics_dict.items():
            if computed_value is not None:
                value = (
                    computed_value.detach()
                    if isinstance(computed_value, torch.Tensor)
                    else computed_value
                )
                detached_metrics_dict[metric_key] = value

        logging_thread = BackgroundLogging(
            wandb, detached_metrics_dict, phase_name, global_step
        )
        logging_thread.start()

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


if __name__ == "__main__":

    @configurable
    def build_something(batch_size: int, num_layers: int):
        return batch_size, num_layers

    build_something_config = build_something.build_config(
        populate_full_signature=True
    )
    dummy_config = build_something_config(batch_size=32, num_layers=2)
    print(dummy_config)

    from hydra_zen import builds, instantiate

    def build_something(batch_size: int, num_layers: int):
        return batch_size, num_layers

    dummy_config = builds(build_something, populate_full_signature=True)

    dummy_function_instantiation = instantiate(dummy_config)

    print(dummy_function_instantiation)
