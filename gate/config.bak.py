from dataclasses import dataclass
import os
from typing import Any, Optional

import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    MISSING,
    ZenField,
    builds,
    make_config,
)
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from gate.boilerplate.core import Learner
from gate.boilerplate.callbacks import UploadCheckpointsToHuggingFace
from gate.boilerplate.utils import get_hydra_config, get_logger
from gate.data.data import build_dataset
from gate.models.models import build_model


CHECKPOINT_DIR = "${hf_cache_dir}"
NUM_WORKERS = "${num_workers}"
HF_USERNAME = "${hf_username}"
CODE_DIR = "${code_dir}"
DATASET_DIR = "${dataset_dir}"
EXPERIMENT_NAME = "${exp_name}"
EXPERIMENTS_ROOT_DIR = "${root_experiment_dir}"
TRAIN_BATCH_SIZE = "${train_batch_size}"
CURRENT_EXPERIMENT_DIR = "${current_experiment_dir}"
TRAIN_ITERS = "${learner.train_iters}"
REPO_PATH = "${repo_path}"
EXP_NAME = "${exp_name}"
SEED = "${seed}"
RESUME = "${resume}"
LOGGER_LEVEL = "${logger_level}"
GPU_MEMORY = 24  # in GB
DUMMY_BATCH_MODE = "${dummy_batch_mode}"
PREFETCH_FACTOR = "${prefetch_factor}"
PERSISTENT_WORKERS = "${persistent_workers}"
PIN_MEMORY = "${pin_memory}"


hydra_logger = get_logger("hydra")


@dataclass
class BaseConfig:
    # Must be passed at command line -- neccesary arguments

    exp_name: str = MISSING

    # Defaults for these are provided in the collect_config_store method,
    # but will be often overridden at command line

    model: Any = MISSING
    dataset: Any = MISSING
    dataloader: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    learner: Any = MISSING
    callbacks: Any = MISSING

    hf_username: str = (
        os.environ["HF_USERNAME"] if "HF_USERNAME" in os.environ else MISSING
    )

    seed: int = 42

    freeze_backbone: bool = False
    resume: bool = False
    resume_from_checkpoint: Optional[int] = None
    print_config: bool = True
    # Dataloader config
    train_num_samples_per_episode: int = 96
    eval_num_samples_per_episode: int = 96
    num_workers: int = 2
    prefetch_factor: int = 1
    persistent_workers: bool = True
    pin_memory: bool = True

    train: bool = True
    test: bool = False
    dummy_batch_mode: bool = False
    download_latest: bool = True
    download_checkpoint_with_name: Optional[str] = None
    logger_level: str = "INFO"

    root_experiment_dir: str = (
        os.environ["EXPERIMENTS_DIR"]
        if "EXPERIMENTS_DIR" in os.environ
        else "/experiments"
    )

    dataset_dir: str = (
        os.environ["DATASET_DIR"] if "DATASET_DIR" in os.environ else "/data"
    )

    current_experiment_dir: str = "${root_experiment_dir}/${exp_name}"
    hf_repo_path: str = "${hf_username}/${exp_name}"
    hf_cache_dir: str = "${current_experiment_dir}/repo"
    code_dir: str = (
        os.environ["CODE_DIR"]
        if "CODE_DIR" in os.environ
        else "${hydra:runtime.cwd}"
    )


# Using hydra might look a bit more verbose but it saves having
# to manually define
# future args, and makes it a lot easier to add whatever we need
# from the command line


def collect_config_store():
    config_store = ConfigStore.instance()
    ##########################################################################
    # Model configs

    model_config = build_model()

    config_store.store(group="model", name="default", node=model_config)

    data_config: Any = build_dataset

    food101_config = data_config(dataset_name="food101", data_dir=DATASET_DIR)

    config_store.store(group="dataset", name="food101", node=food101_config)

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

    cosine_learning_rate_scheduler_config = builds(
        CosineLRScheduler,
        populate_full_signature=True,
        zen_partial=True,
    )

    cosine_learning_rate_scheduler_config = (
        cosine_learning_rate_scheduler_config()
    )

    config_store.store(
        group="optimizer",
        name="adamw",
        node=adamw_optimizer_config(lr=1e-5, weight_decay=0.0),
    )

    config_store.store(
        group="scheduler",
        name="cosine-annealing",
        node=cosine_learning_rate_scheduler_config,
    )

    ##########################################################################
    learner_config = builds(Learner, populate_full_signature=True)

    learner_config = learner_config(
        model=None,
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=CHECKPOINT_DIR,
        resume=RESUME,
        evaluate_every_n_steps=1000,
        checkpoint_after_validation=True,
        checkpoint_every_n_steps=500,
        train_iters=100000,
        limit_val_iters=250,
        dummy_batch_mode=DUMMY_BATCH_MODE,
        print_model_parameters=False,
    )
    config_store.store(
        group="learner",
        name="default",
        node=learner_config,
    )

    ##########################################################################
    HFModelUploadConfig = builds(
        UploadCheckpointsToHuggingFace, populate_full_signature=True
    )

    hf_upload = HFModelUploadConfig(
        repo_name=EXPERIMENT_NAME, repo_owner=HF_USERNAME
    )

    default_callbacks = dict(hf_uploader=hf_upload)

    config_store.store(
        group="callbacks", name="default", node=default_callbacks
    )

    ###########################################################################
    config_store.store(
        group="hydra",
        name="default",
        node=get_hydra_config(logger_level=LOGGER_LEVEL),
    )

    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            dict(learner="default"),
            dict(optimizer="adamw"),
            dict(scheduler="cosine-annealing"),
            dict(model="tali_image_text_base_patch16_224"),
            dict(dataset="tali_image_text_dataset"),
            dict(dataloader="default"),
            dict(hydra="default"),
            dict(callbacks="default"),
        ],
    )
    # Config
    config_store.store(name="config", node=config)

    return config_store
