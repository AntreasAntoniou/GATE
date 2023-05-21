import os
import pathlib
from typing import Any, Callable, Optional

import hydra
import neptune
import wandb
from hydra_zen import instantiate
from omegaconf import OmegaConf
from rich import print
from rich.traceback import install
from torch.utils.data import Subset

from gate.boilerplate.callbacks import instantiate_callbacks
from gate.boilerplate.core import Learner
from gate.boilerplate.utils import (
    create_hf_model_repo_and_download_maybe,
    get_logger,
    pretty_config,
    set_seed,
)
from gate.config.config import collect_config_store
from gate.data.core import GATEDataset
from gate.models.core import GATEModel

# Set environmental variables for better debugging
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# Install rich tracebacks for better visibility during debugging
install()

import torch

# Collecting configuration
config_store = collect_config_store()

# Initializing logger
logger = get_logger(name=__name__)


def setup(ckpt_path: Optional[str], cfg: Any) -> tuple:
    """
    Function to set up and return the global step and experiment tracker

    Args:
        ckpt_path (str): The path to the checkpoint file
        cfg (Any): The configuration parameters

    Returns:
        tuple: global step and experiment tracker
    """
    if ckpt_path is not None and cfg.resume is True:
        trainer_state = torch.load(
            pathlib.Path(ckpt_path) / "trainer_state.pt"
        )
        global_step = trainer_state["global_step"]
        experiment_tracker = neptune.init_run(
            source_files=["gate/*.py", "kubernetes/*.py"]
        )
    else:
        global_step = 0
        experiment_tracker = neptune.init_run(
            source_files=["gate/*.py", "kubernetes/*.py"]
        )

    return global_step, experiment_tracker


def log_checkpoint_path(ckpt_path: Optional[str], cfg: Any) -> None:
    """
    Log the checkpoint path.

    Args:
        ckpt_path (str): The path to the checkpoint file
        cfg (Any): The configuration parameters
    """
    if ckpt_path is not None:
        logger.info(
            f"ckpt_path: {ckpt_path}, exists: {ckpt_path.exists()}, "
            f"resume: {cfg.resume}"
        )
    else:
        logger.info(f"ckpt_path: {ckpt_path}, resume: {cfg.resume},")


def log_experiment_parameters(
    experiment_tracker: Any, config_dict: dict, global_step: int
) -> None:
    """
    Log parameters to the experiment tracker and Weights & Biases.

    Args:
        experiment_tracker (Any): The experiment tracker
        config_dict (dict): The configuration dictionary
        global_step (int): The global step
    """
    from neptune.utils import stringify_unsupported

    experiment_tracker["config"] = stringify_unsupported(config_dict)
    experiment_tracker["init_global_step"] = global_step


def log_wandb_parameters(config_dict: dict, global_step: int) -> None:
    """
    Log parameters to Weights & Biases.

    Args:
        config_dict (dict): The configuration dictionary
        global_step (int): The global step
    """
    wandb.config.update(config_dict)
    wandb.config.update({"init_global_step": global_step})


def get_datasets(dataset: GATEDataset, global_step: int):
    """
    Get training, validation, and test datasets.

    Args:
        dataset (GATEDataset): The main dataset.
        global_step (int): The global training step.

    Returns:
        tuple: Training, validation, and test datasets.
    """
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    if global_step > 0:
        train_dataset = Subset(
            train_dataset, range(global_step, len(train_dataset))
        )

    return train_dataset, val_dataset, test_dataset


def instantiate_dataloader(
    cfg: Any, dataset: GATEDataset, batch_size: int, shuffle: bool
):
    """
    Instantiate a data loader.

    Args:
        cfg (Any): The configuration parameters.
        dataset (GATEDataset): The dataset.
        batch_size (int): The batch size.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: The instantiated data loader.
    """
    return instantiate(
        cfg.dataloader, dataset=dataset, batch_size=batch_size, shuffle=shuffle
    )


def count_model_parameters(model: GATEModel):
    """
    Count the number of parameters in a model.

    Args:
        model (GATEModel): The model.

    Returns:
        int: The number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def instantiate_optimizer(cfg: Any, model: GATEModel):
    """
    Instantiate an optimizer.

    Args:
        cfg (Any): The configuration parameters.
        model (GATEModel): The model.

    Returns:
        Optimizer: The instantiated optimizer.
    """
    return instantiate(
        cfg.optimizer, params=model.parameters(), _partial_=False
    )


def instantiate_scheduler(cfg: Any, optimizer):
    """
    Instantiate a scheduler.

    Args:
        cfg (Any): The configuration parameters.
        optimizer (Optimizer): The optimizer.

    Returns:
        _LRScheduler: The instantiated scheduler.
    """
    return instantiate(
        cfg.scheduler,
        optimizer=optimizer,
        t_initial=cfg.learner.train_iters,
        _partial_=False,
    )


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: Any) -> None:
    """
    The main function for training and testing the model.

    Args:
        cfg (Any): The configuration parameters
    """
    # Pretty print the configuration
    print(pretty_config(cfg, resolve=True))

    os.environ["HF_REPO_PATH"] = cfg.hf_repo_path
    os.environ["HF_CACHE_DIR"] = cfg.hf_cache_dir
    os.environ["CURRENT_EXPERIMENT_DIR"] = cfg.current_experiment_dir

    # Set the seed for reproducibility
    set_seed(seed=cfg.seed)

    ckpt_dict = create_hf_model_repo_and_download_maybe(
        cfg=cfg,
        hf_repo_path=cfg.hf_repo_path,
        hf_cache_dir=cfg.hf_cache_dir,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        resume=cfg.resume,
    )
    ckpt_path = ckpt_dict["root_filepath"] if ckpt_dict else None

    # Log checkpoint path
    log_checkpoint_path(ckpt_path, cfg)

    logger.info(f"Using checkpoint: {ckpt_path}")

    global_step, experiment_tracker = setup(ckpt_path, cfg)

    model_and_transform = instantiate(cfg.model)

    model: GATEModel = model_and_transform.model
    transform: Optional[Callable] = model_and_transform.transform

    wandb.init()
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    log_experiment_parameters(experiment_tracker, config_dict, global_step)
    log_wandb_parameters(config_dict, global_step)

    dataset: GATEDataset = instantiate(cfg.dataset, transforms=transform)
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset, global_step
    )

    train_dataloader = instantiate_dataloader(
        cfg, train_dataset, cfg.train_batch_size, shuffle=True
    )
    val_dataloader = instantiate_dataloader(
        cfg, val_dataset, cfg.eval_batch_size, shuffle=False
    )
    test_dataloader = instantiate_dataloader(
        cfg, test_dataset, cfg.eval_batch_size, shuffle=False
    )

    experiment_tracker["num_parameters"] = count_model_parameters(model)

    optimizer = instantiate_optimizer(cfg, model)
    scheduler = instantiate_scheduler(cfg, optimizer)

    trainer = instantiate(
        cfg.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_tracker=experiment_tracker,
    )

    evaluator = instantiate(
        cfg.evaluator, experiment_tracker=experiment_tracker
    )

    learner: Learner = instantiate(
        cfg.learner,
        model=model,
        trainers=[trainer],
        evaluators=[evaluator],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callbacks=instantiate_callbacks(cfg.callbacks),
        resume=ckpt_path,
        experiment_tracker=experiment_tracker,
    )

    if cfg.train:
        learner.train()

    if cfg.test:
        learner.test(test_dataloader=test_dataloader)


if __name__ == "__main__":
    run()
