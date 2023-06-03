import pathlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb
from hydra_zen import instantiate
from torch.utils.data import Subset

from gate.boilerplate.utils import get_logger
from gate.data.core import GATEDataset
from gate.models.core import GATEModel

logger = get_logger(__name__)


def setup(ckpt_path: Optional[str], cfg: Any) -> tuple:
    """
    Function to set up and return the global step and experiment tracker

    Args:
        ckpt_path (str): The path to the checkpoint file
        cfg (Any): The configuration parameters

    Returns:
        tuple: global step and experiment tracker
    """
    global_step = 0
    if ckpt_path is not None and cfg.resume is True:
        trainer_state = torch.load(
            pathlib.Path(ckpt_path) / "trainer_state.pt"
        )
        global_step = trainer_state["global_step"]

    return global_step


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
        _partial_=False,
    )

    # return instantiate(
    #     cfg.scheduler,
    #     optimizer=optimizer,
    #     t_initial=cfg.learner.train_iters,
    #     _partial_=False,
    # )
