import os
import pathlib
from typing import Any, Callable, Optional

from accelerate import Accelerator

# Set environmental variables for better debugging
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import hydra
import neptune
import wandb
from hydra_zen import instantiate
from omegaconf import OmegaConf
from rich import print
from rich.traceback import install
from torch.utils.data import Subset

from gate.boilerplate.callbacks import instantiate_callbacks
from gate.boilerplate.convenience import (
    count_model_parameters,
    get_datasets,
    instantiate_dataloader,
    instantiate_optimizer,
    instantiate_scheduler,
    log_checkpoint_path,
    log_experiment_parameters,
    log_wandb_parameters,
    setup,
)
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

# Install rich tracebacks for better visibility during debugging
install()

# Collecting configuration
config_store = collect_config_store()

# Initializing logger
logger = get_logger(name=__name__)

accelerator = Accelerator()


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
    # task_adaptor = instantiate(cfg.task_adaptor)
    # model = task_adaptor(model_and_transform.model)

    model: GATEModel = model_and_transform.model
    model = accelerator.prepare(model)
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

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = accelerator.prepare(train_dataloader, val_dataloader, test_dataloader)

    experiment_tracker["num_parameters"] = count_model_parameters(model)

    optimizer = instantiate_optimizer(cfg, model)
    scheduler = instantiate_scheduler(cfg, optimizer)

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    trainer = instantiate(
        cfg.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_tracker=experiment_tracker,
    )

    evaluator = instantiate(
        cfg.evaluator, experiment_tracker=experiment_tracker
    )
    # TODO: allow losses and task adapters to be defined at this level

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
        experiment_tracker=experiment_tracker,
    )

    if cfg.train:
        learner.train()

    if cfg.test:
        learner.test(test_dataloader=test_dataloader)


if __name__ == "__main__":
    run()
