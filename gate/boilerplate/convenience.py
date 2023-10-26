import pathlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import transformers
from hydra_zen import instantiate
from torch.utils.data import Subset

import wandb
from gate.data.core import GATEDataset
from gate.models.core import GATEModel

logger = logging.getLogger(__name__)


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


def get_datasets(dataset: Dict[str, GATEDataset], global_step: int):
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


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def instantiate_optimizer(cfg: Any, model: GATEModel):
    """
    Instantiate an optimizer.

    Args:
        cfg (Any): The configuration parameters.
        model (GATEModel): The model.

    Returns:
        Optimizer: The instantiated optimizer.
    """
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay
    decay_parameters = get_parameter_names(
        model=model,
        forbidden_layer_types=[
            torch.nn.LayerNorm,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm1d,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
        ],
    )

    decoder_decay_parameters = [
        name
        for name in decay_parameters
        if "decoder_head" in name and "bias" not in name
    ]

    encoder_decay_parameters = [
        name
        for name in decay_parameters
        if "bias" not in name and name not in decoder_decay_parameters
    ]

    decay_parameters = encoder_decay_parameters + decoder_decay_parameters

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in encoder_decay_parameters
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters and "decoder_head" not in n
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decoder_decay_parameters
            ],
            "weight_decay": weight_decay,
            "lr": lr * 10.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters and "decoder_head" in n
            ],
            "weight_decay": 0.0,
            "lr": lr * 10.0,
        },
    ]

    logger.info(f"Optimizer hyperparameters: {cfg.optimizer}")

    return transformers.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-06,
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
