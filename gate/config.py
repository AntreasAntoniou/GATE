import os
from dataclasses import dataclass
from typing import Any, Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, ZenField, builds, make_config
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from gate.boilerplate.callbacks import UploadCheckpointsToHuggingFace
from gate.boilerplate.core import Learner
from gate.boilerplate.evaluators.classification import ClassificationEvaluator
from gate.boilerplate.trainers.classification import ClassificationTrainer
from gate.boilerplate.utils import get_hydra_config, get_logger, pretty_config
from gate.data.image.classification.food101 import Food101Dataset
from gate.models.clip import build_gate_model, build_model


def get_env_var(key: str, default: Any) -> Any:
    return os.environ.get(key, default)


HF_CACHE_DIR = get_env_var(
    "HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface")
)
HF_USERNAME = get_env_var("HF_USERNAME", None)
CODE_DIR = get_env_var("CODE_DIR", "")
DATASET_DIR = get_env_var("DATASET_DIR", "data/")
EXPERIMENT_NAME = get_env_var("EXPERIMENT_NAME", "exp_0")
EXPERIMENTS_ROOT_DIR = get_env_var("EXPERIMENTS_ROOT_DIR", "experiments/")
CURRENT_EXPERIMENT_DIR = get_env_var(
    "CURRENT_EXPERIMENT_DIR", f"{EXPERIMENTS_ROOT_DIR}/{EXPERIMENT_NAME}"
)
TRAIN_BATCH_SIZE = get_env_var("TRAIN_BATCH_SIZE", 128)
EVAL_BATCH_SIZE = get_env_var("EVAL_BATCH_SIZE", 256)
NUM_WORKERS = get_env_var("NUM_WORKERS", 8)
PREFETCH_FACTOR = get_env_var("PREFETCH_FACTOR", 2)
PERSISTENT_WORKERS = get_env_var("PERSISTENT_WORKERS", True)
PIN_MEMORY = get_env_var("PIN_MEMORY", True)

TRAIN_ITERS = get_env_var("TRAIN_ITERS", 10000)
SEED = get_env_var("SEED", 42)
RESUME = get_env_var("RESUME", True)
LOGGER_LEVEL = get_env_var("LOGGER_LEVEL", "INFO")
DUMMY_BATCH_MODE = get_env_var("DUMMY_BATCH_MODE", False)
GPU_MEMORY = 24  # in GB


hydra_logger = get_logger("hydra")


@dataclass
class BaseConfig:
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

    ##########################################################################
    # Model configs

    model_config = build_gate_model.__config__(populate_full_signature=True)

    config_store.store(
        group="model", name="clip-base16", node=model_config(num_classes=101)
    )

    ##########################################################################
    # Dataset configs

    food101_config: Any = (
        Food101Dataset.build_gate_food_101_dataset.__config__(
            populate_full_signature=True
        )
    )

    config_store.store(
        group="dataset",
        name="food101",
        node={"food101": food101_config(data_dir=DATASET_DIR)},
    )
    ##########################################################################
    # Trainer configs

    classification_trainer_config = ClassificationTrainer.__config__(
        populate_full_signature=True
    )

    config_store.store(
        group="trainer",
        name="classification",
        node=classification_trainer_config(optimizer=None),
    )

    ##########################################################################
    # Evaluator configs

    classification_evaluator_config = ClassificationEvaluator.__config__(
        populate_full_signature=True
    )

    config_store.store(
        group="evaluator",
        name="classification",
        node=classification_evaluator_config(),
    )

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
    learner_config = builds(Learner, populate_full_signature=True)

    learner_config = learner_config(
        model=None,
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=CURRENT_EXPERIMENT_DIR,
        resume=RESUME,
        evaluate_every_n_steps=1000,
        checkpoint_after_validation=True,
        checkpoint_every_n_steps=500,
        train_iters=TRAIN_ITERS,
        limit_val_iters=1000,
        dummy_batch_mode=DUMMY_BATCH_MODE,
        print_model_parameters=False,
    )
    config_store.store(
        group="learner",
        name="default",
        node=learner_config,
    )

    ##########################################################################
    # Callback configs

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
    # 🌐 Hydra configs
    config_store.store(
        group="hydra",
        name="default",
        node=get_hydra_config(logger_level=LOGGER_LEVEL),
    )

    ###########################################################################
    # 🌐 Hydra Zen global configs
    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
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
            dict(model="clip-base16"),
            dict(dataset="food101"),
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
    def test(cfg: BaseConfig):
        print(pretty_config(cfg, resolve=True))

    test()


if __name__ == "__main__":
    main()
