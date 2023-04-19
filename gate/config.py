import os
from typing import Any, Optional

import torch
from hydra_zen import (
    builds,
)
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from gate.boilerplate.core import Learner
from gate.boilerplate.callbacks import UploadCheckpointsToHuggingFace
from gate.boilerplate.utils import get_logger
from gate.data.data import build_dataset
from gate.models.models import build_model

import fire

logger = get_logger(set_rich=True)


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
NUM_WORKERS = get_env_var("NUM_WORKERS", 2)
PREFETCH_FACTOR = get_env_var("PREFETCH_FACTOR", 2)
PERSISTENT_WORKERS = get_env_var("PERSISTENT_WORKERS", True)
PIN_MEMORY = get_env_var("PIN_MEMORY", True)

TRAIN_ITERS = get_env_var("TRAIN_ITERS", 10000)
SEED = get_env_var("SEED", 42)
RESUME = get_env_var("RESUME", True)
LOGGER_LEVEL = get_env_var("LOGGER_LEVEL", "INFO")
DUMMY_BATCH_MODE = get_env_var("DUMMY_BATCH_MODE", False)
GPU_MEMORY = 24  # in GB


class BaseConfig:
    def __init__(
        self,
        exp_name: str = EXPERIMENT_NAME,
        model: Any = "default",
        dataset: Any = "default",
        dataloader: Any = "default",
        optimizer: Any = "default",
        scheduler: Any = "default",
        learner: Any = "default",
        callbacks: Any = "default",
        hf_username: str = HF_USERNAME,
        seed: int = SEED,
        resume: bool = RESUME,
        resume_from_checkpoint: Optional[int] = None,
        print_config: bool = True,
        num_workers: int = NUM_WORKERS,
        prefetch_factor: int = PREFETCH_FACTOR,
        persistent_workers: bool = PERSISTENT_WORKERS,
        pin_memory: bool = PIN_MEMORY,
        train: bool = True,
        test: bool = False,
        dummy_batch_mode: bool = DUMMY_BATCH_MODE,
        logger_level: str = LOGGER_LEVEL,
        experiments_root_dir: str = EXPERIMENTS_ROOT_DIR,
        dataset_dir: str = DATASET_DIR,
        current_experiment_dir: str = CURRENT_EXPERIMENT_DIR,
        hf_repo_path: str = f"{HF_USERNAME}/${EXPERIMENT_NAME}",
        hf_cache_dir: str = HF_CACHE_DIR,
        code_dir: str = CODE_DIR,
    ):
        self.config_store = self.collect_config_store()
        self.model = self.config_store["model"][model]
        self.dataset = self.config_store["dataset"][dataset]
        self.dataloader = self.config_store["dataloader"][dataloader]
        self.optimizer = self.config_store["optimizer"][optimizer]
        self.scheduler = self.config_store["scheduler"][scheduler]
        self.learner = self.config_store["learner"][learner]
        self.callbacks = self.config_store["callbacks"][callbacks]

        self.exp_name = exp_name
        self.hf_username = hf_username
        self.seed = seed
        self.resume = resume
        self.resume_from_checkpoint = resume_from_checkpoint
        self.print_config = print_config
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.train = train
        self.test = test
        self.dummy_batch_mode = dummy_batch_mode
        self.logger_level = logger_level
        self.experiments_root_dir = experiments_root_dir
        self.dataset_dir = dataset_dir
        self.current_experiment_dir = current_experiment_dir
        self.hf_repo_path = hf_repo_path
        self.hf_cache_dir = hf_cache_dir
        self.code_dir = code_dir

    def collect_config_store():
        config_store = {
            "model": {},
            "dataset": {},
            "dataloader": {},
            "optimizer": {},
            "scheduler": {},
            "learner": {},
            "callbacks": {},
        }
        ##########################################################################
        # Model configs
        model_config = build_model()

        config_store["model"]["default"] = model_config

        data_config: Any = build_dataset

        food101_config = data_config(
            dataset_name="food101", data_dir=DATASET_DIR
        )

        config_store["dataset"]["default"] = food101_config
        config_store["dataset"]["food101"] = food101_config

        dataloader_config = builds(
            DataLoader, dataset=None, populate_full_signature=True
        )

        config_store["dataloader"]["default"] = (
            dataloader_config(
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

        config_store["optimizer"]["adamw"] = adamw_optimizer_config(
            lr=1e-5, weight_decay=0.0
        )

        cosine_learning_rate_scheduler_config = builds(
            CosineLRScheduler,
            populate_full_signature=True,
            zen_partial=True,
        )

        config_store["scheduler"][
            "cosine-annealing"
        ] = cosine_learning_rate_scheduler_config()

        ##########################################################################
        learner_config = builds(Learner, populate_full_signature=True)

        config_store["learner"]["default"] = learner_config(
            model=None,
            experiment_name=EXPERIMENT_NAME,
            experiment_dir=CURRENT_EXPERIMENT_DIR,
            resume=RESUME,
            evaluate_every_n_steps=1000,
            checkpoint_after_validation=True,
            checkpoint_every_n_steps=500,
            train_iters=100000,
            limit_val_iters=250,
            dummy_batch_mode=DUMMY_BATCH_MODE,
            print_model_parameters=False,
        )

        ##########################################################################
        HFModelUploadConfig = builds(
            UploadCheckpointsToHuggingFace, populate_full_signature=True
        )

        hf_upload = HFModelUploadConfig(
            repo_name=EXPERIMENT_NAME, repo_owner=HF_USERNAME
        )

        default_callbacks = dict(hf_uploader=hf_upload)

        config_store["callbacks"]["default"] = default_callbacks

        return config_store


if __name__ == "__main__":
    fire.Fire(BaseConfig)
