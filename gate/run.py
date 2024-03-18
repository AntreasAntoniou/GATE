import os
from copy import deepcopy
from typing import Any, Callable, Optional

import yaml
from gate.models.backbones.einspace import fancy_yaml_load

# Set environmental variables for better debugging
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import logging

import hydra
import wandb
from accelerate import Accelerator
from gate.boilerplate.callbacks import instantiate_callbacks
from gate.boilerplate.convenience import (
    count_model_parameters,
    get_datasets,
    instantiate_dataloader,
    instantiate_optimizer,
    instantiate_scheduler,
    log_checkpoint_path,
    log_wandb_parameters,
    setup,
)
from gate.boilerplate.core import Learner
from gate.boilerplate.utils import (
    create_hf_model_repo_and_download_maybe,
    pretty_config,
    set_seed,
)
from gate.config.config import collect_config_store
from gate.data.core import GATEDataset
from gate.models.core import GATEModel
from hydra_zen import instantiate
from omegaconf import OmegaConf
from rich import print
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.traceback import install
from torch import nn

# Install rich tracebacks for better visibility during debugging
install(width=150, word_wrap=True)

# Collecting configuration
config_store = collect_config_store()

# Initializing logger
logger = logging.getLogger(__name__)

# logging.getLogger("gate").setLevel(logging.DEBUG)


def pretty_print_parameters(model: nn.Module):
    console = Console()

    table = Table(title=Text("Model Parameters", style=Style(color="green")))

    table.add_column("Name", justify="left", style="cyan")
    table.add_column("Shape", justify="center", style="magenta")
    table.add_column("Data Type", justify="center", style="yellow")
    table.add_column("Device", justify="center", style="green")
    print(f"Model Parameters: {model.__class__.__name__}")
    for name, param in model.named_parameters():
        table.add_row(
            Text(str(name), style=Style(color="blue")),
            Text(str(tuple(param.shape)), style=Style(color="red")),
            Text(str(param.dtype), style=Style(color="yellow")),
            Text(str(param.device), style=Style(color="green")),
        )

    console.print(table)
    return table


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: Any) -> None:
    """
    The main function for training and testing the model.

    Args:
        cfg (Any): The configuration parameters
    """

    accelerator = Accelerator()
    # Pretty print the configuration
    print(pretty_config(cfg, resolve=True))

    os.environ["HF_REPO_PATH"] = cfg.hf_repo_path  # make this optional
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

    global_step = setup(ckpt_path, cfg)

    encoder = instantiate(cfg.encoder)
    task_adapted_model = instantiate(cfg.adapter, encoder=encoder)
    transform: Optional[Callable] = deepcopy(
        task_adapted_model.adapter_transforms
    )

    model: GATEModel = GATEModel(
        config=task_adapted_model.modality_config, model=task_adapted_model
    )

    wandb.init()
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    log_wandb_parameters(config_dict, global_step)

    dataset: GATEDataset = instantiate(cfg.dataset, transforms=transform)
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset, global_step
    )

    model.meta_data = val_dataset.meta_data

    train_dataloader = instantiate_dataloader(
        cfg, train_dataset, cfg.train_batch_size, shuffle=True
    )
    val_dataloader = instantiate_dataloader(
        cfg, val_dataset, cfg.eval_batch_size, shuffle=False
    )
    test_dataloader = instantiate_dataloader(
        cfg, test_dataset, cfg.eval_batch_size, shuffle=False
    )

    optimizer = instantiate_optimizer(cfg, model)
    scheduler = instantiate_scheduler(cfg, optimizer)

    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = accelerator.prepare(train_dataloader, val_dataloader, test_dataloader)

    pretty_print_parameters(model)
    wandb.log({"model/num_parameters": count_model_parameters(model)})

    trainer = instantiate(
        cfg.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    evaluator = instantiate(
        cfg.evaluator,
    )

    learner: Learner = instantiate(
        cfg.learner,
        model=model,
        accelerator=accelerator,
        trainer=trainer,
        evaluator=evaluator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callbacks=instantiate_callbacks(cfg.callbacks),
        resume=ckpt_path,
    )

    if cfg.train:
        learner.train()

    if cfg.test:
        learner.test(test_dataloader=test_dataloader)


if __name__ == "__main__":
    run()
