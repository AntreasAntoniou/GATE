import multiprocessing as mp
import os
from typing import Any


def get_env_var(key: str, default: Any) -> Any:
    return os.environ.get(key, default)


## Define env variables, and config defaults here
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
NUM_WORKERS = get_env_var("NUM_WORKERS", mp.cpu_count())
PREFETCH_FACTOR = get_env_var("PREFETCH_FACTOR", 2)
PERSISTENT_WORKERS = get_env_var("PERSISTENT_WORKERS", False)
PIN_MEMORY = get_env_var("PIN_MEMORY", True)

TRAIN_ITERS = get_env_var("TRAIN_ITERS", 10000)
SEED = get_env_var("SEED", 42)
RESUME = get_env_var("RESUME", True)
LOGGER_LEVEL = get_env_var("LOGGER_LEVEL", "INFO")
DUMMY_BATCH_MODE = get_env_var("DUMMY_BATCH_MODE", False)
GPU_MEMORY = 24  # in GB
HF_OFFLINE_MODE = get_env_var("HF_OFFLINE_MODE", False)
WANDB_OFFLINE_MODE = get_env_var("WANDB_OFFLINE_MODE", False)

## Define yaml variable access codes here
HYDRATED_EXPERIMENT_NAME = "${exp_name}"
HYDRATED_MODEL_CONFIG = "${model}"
HYDRATED_DATASET_CONFIG = "${dataset}"
HYDRATED_DATASET_INGORE_INDEX = "${dataset.ignore_index}"
HYDRATED_LABEL_IDX_TO_CLASS_NAME = "${dataset.label_idx_to_class_name}"
HYDRATED_NUM_CLASSES = "${dataset.num_classes}"
HYDRATED_TASK_NAME = "${dataset.task_name}"
HYDRATED_IMAGE_SIZE = "${dataset.image_size}"
HYDRATED_TRAINER_CONFIG = "${trainer}"
HYDRATED_EVALUATOR_CONFIG = "${evaluator}"

HYDRATED_DATALOADER_CONFIG = "${dataloader}"
HYDRATED_OPTIMIZER_CONFIG = "${optimizer}"
HYDRATED_SCHEDULER_CONFIG = "${scheduler}"
HYDRATED_LEARNER_CONFIG = "${learner}"
HYDRATED_CALLBACKS_CONFIG = "${callbacks}"

HYDRATED_HF_USERNAME = "${hf_username}"
HYDRATED_SEED = "${seed}"
HYDRATED_TRAIN_BATCH_SIZE = "${train_batch_size}"
HYDRATED_EVAL_BATCH_SIZE = "${eval_batch_size}"
HYDRATED_TRAIN_ITERS = "${train_iters}"
HYDRATED_RESUME = "${resume}"
HYDRATED_RESUME_FROM_CHECKPOINT = "${resume_from_checkpoint}"
HYDRATED_PRINT_CONFIG = "${print_config}"
HYDRATED_NUM_WORKERS = "${num_workers}"
HYDRATED_PREFETCH_FACTOR = "${prefetch_factor}"
HYDRATED_PERSISTENT_WORKERS = "${persistent_workers}"
HYDRATED_PIN_MEMORY = "${pin_memory}"
HYDRATED_TRAIN = "${train}"
HYDRATED_TEST = "${test}"
HYDRATED_DUMMY_BATCH_MODE = "${dummy_batch_mode}"
HYDRATED_LOGGER_LEVEL = "${logger_level}"
HYDRATED_EXPERIMENTS_ROOT_DIR = "${experiments_root_dir}"
HYDRATED_DATASET_DIR = "${dataset_dir}"
HYDRATED_CURRENT_EXPERIMENT_DIR = "${current_experiment_dir}"
HYDRATED_HF_REPO_PATH = "${hf_repo_path}"
HYDRATED_HF_CACHE_DIR = "${hf_cache_dir}"
HYDRATED_CODE_DIR = "${code_dir}"
