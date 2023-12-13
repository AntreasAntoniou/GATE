import logging
from dataclasses import dataclass
from typing import Any, Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, ZenField, builds, make_config
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from gate.boilerplate.callbacks import UploadCheckpointsToHuggingFace
from gate.boilerplate.decorators import register_configurables
from gate.boilerplate.utils import get_hydra_config, pretty_config
from gate.config.variables import (
    CODE_DIR,
    DATASET_DIR,
    DUMMY_BATCH_MODE,
    EVAL_BATCH_SIZE,
    EXPERIMENTS_ROOT_DIR,
    HF_USERNAME,
    HYDRATED_NUM_WORKERS,
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
from gate.data.core import collate_fn_with_token_pad

hydra_logger = logging.getLogger("hydra")


@dataclass
class Config:
    """
    A dataclass for storing the base configuration for the application.
    🛠 Contains all necessary configurations for model, dataset, dataloader,
    optimizer, scheduler, learner, and callbacks.
    """

    # Must be passed at command line -- necessary arguments
    exp_name: str = MISSING

    # Defaults for these are provided in the collect_config_store method,
    # but will be often overridden at command line

    encoder: Any = MISSING
    adapter: Any = MISSING
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
    resume_from_checkpoint: Optional[str] = None
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
    current_experiment_dir: str = "${experiments_root_dir}/"
    hf_repo_path: str = "${hf_username}/${exp_name}"
    hf_cache_dir: str = "${current_experiment_dir}/huggingface_cache"
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
        DataLoader,
        dataset=None,
        populate_full_signature=True,
        collate_fn=collate_fn_with_token_pad,
    )

    config_store.store(
        group="dataloader",
        name="default",
        node=dataloader_config(
            batch_size=1,
            num_workers=HYDRATED_NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
        ),
    )
    ##########################################################################
    # Optimizer configs

    # adamw_optimizer_config = builds(
    #     transformers.AdamW, populate_full_signature=True, zen_partial=True
    # )

    config_store.store(
        group="optimizer",
        name="adamw",
        node=dict(
            lr=1e-5,
            weight_decay=0.0,
            betas=(0.9, 0.98),
            eps=1e-6,
        ),
    )

    ##########################################################################
    # Scheduler configs

    cosine_learning_rate_scheduler_config = builds(
        CosineLRScheduler,
        populate_full_signature=True,
        zen_partial=True,
    )

    config_store.store(
        group="scheduler",
        name="cosine-annealing",
        node=cosine_learning_rate_scheduler_config(),
    )

    plateu_learning_rate_scheduler_config = builds(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        populate_full_signature=True,
        zen_partial=True,
    )

    config_store.store(
        group="scheduler",
        name="plateu",
        node=plateu_learning_rate_scheduler_config(
            mode="min",
            factor=0.5,
            patience=1000,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False,
        ),
    )

    linear_learning_rate_scheduler_config = builds(
        torch.optim.lr_scheduler.LinearLR,
        populate_full_signature=True,
        zen_partial=True,
    )

    config_store.store(
        group="scheduler",
        name="linear",
        node=linear_learning_rate_scheduler_config(
            start_factor=1.0,
            end_factor=1.0 / 10000,
            total_iters=10000,
            last_epoch=-1,
            verbose=False,
        ),
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

    # # Convert dictionary to YAML
    # yaml_data = OmegaConf.to_yaml(config_store.repo, resolve=False)

    # # Pretty print YAML with rich
    # syntax = Syntax(yaml_data, "yaml", theme="one-dark")

    # print(syntax)

    ###########################################################################
    # 🌐 Hydra Zen global configs
    zen_config = []

    for value in Config.__dataclass_fields__.values():
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
            dict(scheduler="plateu"),
            dict(encoder="clip-image"),
            dict(adapter="backbone-with-linear-single-classifier"),
            dict(dataset="cifar100"),
            dict(trainer="image_classification"),
            dict(evaluator="image_classification"),
            dict(dataloader="default"),
            dict(hydra="default"),
            dict(callbacks="default"),
        ],
    )
    config_store.store(name="config", node=config)

    return config_store


def main():
    from rich import print

    collect_config_store()

    @hydra.main(config_path=None, config_name="config", version_base=None)
    def test(cfg: Any):
        print(pretty_config(cfg, resolve=True))

    test()


if __name__ == "__main__":
    main()
