import os
from dataclasses import dataclass
from typing import Any, Optional

import hydra
from omegaconf import OmegaConf
from rich.syntax import Syntax
import torch
import yaml
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, ZenField, builds, make_config
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from gate.boilerplate.callbacks import UploadCheckpointsToHuggingFace
from gate.boilerplate.core import Learner
from gate.boilerplate.decorators import (
    register_configurables,
)
from rich import print
from gate.boilerplate.utils import (
    get_hydra_config,
    get_logger,
    pretty_config,
)
from gate.config.variables import (
    CODE_DIR,
    CURRENT_EXPERIMENT_DIR,
    DATASET_DIR,
    DUMMY_BATCH_MODE,
    EVAL_BATCH_SIZE,
    EXPERIMENT_NAME,
    EXPERIMENTS_ROOT_DIR,
    GPU_MEMORY,
    HF_CACHE_DIR,
    HF_USERNAME,
    LOGGER_LEVEL,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    RESUME,
    SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_ITERS,
)

hydra_logger = get_logger("hydra")


@dataclass
class Any:
    """
    A dataclass for storing the base configuration for the application.
    🛠 Contains all necessary configurations for model, dataset, dataloader,
    optimizer, scheduler, learner, and callbacks.
    """

    # Must be passed at command line -- necessary arguments
    exp_name: str = MISSING

    # Defaults for these are provided in the collect_config_store method,
    # but will be often overridden at command line

    model: Any = MISSING
    dataset: Any = MISSING
    trainer: Any = MISSING
    evaluator: Any = MISSING

    dataloader: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    learner: Any = MISSING
    callbacks: Any = MISSING

    # 🌐 Other configurations with default values or environment variables
    hf_username: str = HF_USERNAME
    seed: int = SEED
    train_batch_size: int = TRAIN_BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    train_iters: int = TRAIN_ITERS
    resume: bool = RESUME
    resume_from_checkpoint: Optional[int] = None
    print_config: bool = True
    num_workers: int = NUM_WORKERS
    prefetch_factor: int = PREFETCH_FACTOR
    persistent_workers: bool = PERSISTENT_WORKERS
    pin_memory: bool = PIN_MEMORY
    train: bool = True
    test: bool = True
    dummy_batch_mode: bool = DUMMY_BATCH_MODE
    logger_level: str = LOGGER_LEVEL
    experiments_root_dir: str = EXPERIMENTS_ROOT_DIR
    dataset_dir: str = DATASET_DIR
    current_experiment_dir: str = "${experiments_root_dir}/${exp_name}"
    hf_repo_path: str = "${hf_username}/${exp_name}"
    hf_cache_dir: str = "${current_experiment_dir}/hf_cache"
    code_dir: str = CODE_DIR


# Using hydra might look a bit more verbose but it saves having
# to manually define
# future args, and makes it a lot easier to add whatever we need
# from the command line


def collect_config_store():
    """
    Collects configurations and stores them in the config store.
    🎛 Includes model, dataset, dataloader, optimizer, scheduler, learner,
    and callbacks configurations.
    """

    config_store = ConfigStore.instance()

    register_configurables("gate")

    ##########################################################################
    # Dataloader configs

    dataloader_config = builds(
        DataLoader, dataset=None, populate_full_signature=True
    )

    config_store.store(
        group="dataloader",
        name="default",
        node=dataloader_config(
            batch_size=1,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
        ),
    )
    ##########################################################################
    # Optimizer configs

    adamw_optimizer_config = builds(
        torch.optim.AdamW,
        populate_full_signature=True,
        zen_partial=True,
    )

    config_store.store(
        group="optimizer",
        name="adamw",
        node=adamw_optimizer_config(lr=1e-5, weight_decay=0.0),
    )

    ##########################################################################
    # Scheduler configs

    cosine_learning_rate_scheduler_config = builds(
        CosineLRScheduler,
        populate_full_signature=True,
        zen_partial=True,
    )

    cosine_learning_rate_scheduler_config = (
        cosine_learning_rate_scheduler_config()
    )

    config_store.store(
        group="scheduler",
        name="cosine-annealing",
        node=cosine_learning_rate_scheduler_config,
    )

    ##########################################################################
    # Callback configs

    HFModelUploadConfig = builds(
        UploadCheckpointsToHuggingFace, populate_full_signature=True
    )

    hf_upload = HFModelUploadConfig(
        repo_name="${exp_name}", repo_owner=HF_USERNAME
    )

    default_callbacks = dict(hf_uploader=hf_upload)

    config_store.store(
        group="callbacks", name="default", node=default_callbacks
    )

    ###########################################################################
    # 🌐 Hydra configs
    config_store.store(
        group="hydra",
        name="default",
        node=get_hydra_config(logger_level=LOGGER_LEVEL),
    )

    # Pretty print dict with rich
    # Convert dictionary to YAML and remove newlines after hyphens for a more compact representation
    yaml_data = yaml.dump(config_store.repo)
    print(yaml_data)
    # Create a Syntax instance for the YAML
    syntax = Syntax(yaml_data, "yaml")

    print(syntax)

    ###########################################################################
    # 🌐 Hydra Zen global configs
    zen_config = []

    for value in Any.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    ###########################################################################
    # 🌐 Hydra Zen defaults

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            dict(learner="default"),
            dict(optimizer="adamw"),
            dict(scheduler="cosine-annealing"),
            dict(model="clip"),
            dict(dataset="cifar100"),
            dict(trainer="classification"),
            dict(evaluator="classification"),
            dict(dataloader="default"),
            dict(hydra="default"),
            dict(callbacks="default"),
        ],
    )
    config_store.store(name="config", node=config)

    return config_store


def main():
    from rich import print

    config = collect_config_store()

    @hydra.main(config_path=None, config_name="config", version_base=None)
    def test(cfg: Any):
        print(pretty_config(cfg, resolve=True))

    test()


if __name__ == "__main__":
    main()
