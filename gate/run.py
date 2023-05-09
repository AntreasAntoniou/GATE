import os
import pathlib
from typing import Callable, Optional, Any

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
from gate.data.core import CustomConcatDataset, GATEDataset
from gate.models.core import GATEModel

os.environ[
    "HYDRA_FULL_ERROR"
] = "1"  # Makes sure that stack traces produced by hydra instantiation functions produce
# traceback errors related to the modules they built, rather than generic instantiate related errors that
# are generally useless for debugging

os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"  # extremely useful when debugging DDP setups

install()  # beautiful and clean tracebacks for debugging

import torch

config_store = collect_config_store()

logger = get_logger(name=__name__)


def setup(ckpt_path: str, cfg: Any):
    if ckpt_path is not None and cfg.resume is True:
        trainer_state = torch.load(
            pathlib.Path(ckpt_path) / "trainer_state.pt"
        )
        global_step = trainer_state["global_step"]
        # neptune_id = (
        #     trainer_state["neptune_id"]
        #     if "neptune_id" in trainer_state
        #     else None
        # )
        experiment_tracker = neptune.init_run(
            source_files=["gate/*.py", "kubernetes/*.py"],
            # with_id=neptune_id,
        )
    else:
        global_step = 0
        experiment_tracker = neptune.init_run(
            source_files=["gate/*.py", "kubernetes/*.py"]
        )

    return global_step, experiment_tracker


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: Any) -> None:
    ckpt_dict = create_hf_model_repo_and_download_maybe(cfg)
    ckpt_path = ckpt_dict["root_filepath"]

    if ckpt_path is not None:
        logger.info(
            f"ckpt_path: {ckpt_path}, exists: {ckpt_path.exists()}, "
            f"resume: {cfg.resume}, not resume: {not cfg.resume}"
        )
    else:
        logger.info(
            f"ckpt_path: {ckpt_path}, resume: {cfg.resume}, "
            f"not resume: {not cfg.resume}"
        )

    logger.info(f"Using checkpoint: {ckpt_path}")

    print(pretty_config(cfg, resolve=True))
    set_seed(seed=cfg.seed)

    global_step, experiment_tracker = setup(ckpt_path, cfg)

    model_and_transform = instantiate(cfg.model)

    model: GATEModel = model_and_transform.model
    transform: Optional[Callable] = model_and_transform.transform

    wandb.init()
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_tracker["config"] = config_dict
    experiment_tracker["init_global_step"] = global_step

    wandb.config.update(config_dict)
    wandb.config.update({"init_global_step": global_step})

    dataset: GATEDataset = instantiate(cfg.dataset, transforms=transform)

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    if global_step > 0:
        train_dataset = Subset(
            train_dataset,
            range(global_step, len(train_dataset)),
        )

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )

    test_dataloader = instantiate(
        cfg.dataloader,
        dataset=test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )

    experiment_tracker["num_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters(), _partial_=False
    )

    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler,
        optimizer=optimizer,
        t_initial=cfg.learner.train_iters,
        _partial_=False,
    )

    trainer = instantiate(
        cfg.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_tracker=experiment_tracker,
    )

    evaluator = instantiate(
        cfg.evaluator,
        experiment_tracker=experiment_tracker,
    )

    learner: Learner = instantiate(
        cfg.learner,
        model=model,
        trainers=[trainer],
        evaluators=[
            evaluator,
        ],
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
