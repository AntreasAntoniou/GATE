import pathlib
from pathlib import Path
from time import sleep
import time
from typing import List, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from neptune import Run
from torch.utils.data import DataLoader
from tqdm import tqdm

from gate.boilerplate.callbacks import Callback, CallbackHandler
from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import download_model_with_name, get_logger
from gate.config.variables import (
    DUMMY_BATCH_MODE,
    HYDRATED_CURRENT_EXPERIMENT_DIR,
    HYDRATED_EXPERIMENT_NAME,
    HYDRATED_HF_CACHE_DIR,
    HYDRATED_HF_REPO_PATH,
    HYDRATED_TRAIN_ITERS,
    RESUME,
)
from gate.orchestration.evaluators.classification import (
    Evaluator,
)
from gate.orchestration.trainers.classification import (
    Trainer,
)

logger = get_logger(__name__)

# silence logger for accelerate
accelerate_logger = get_logger("accelerate", logging_level="ERROR")


@configurable(
    group="learner",
    name="default",
    defaults=dict(
        model=None,
        experiment_name=HYDRATED_EXPERIMENT_NAME,
        root_dir=HYDRATED_CURRENT_EXPERIMENT_DIR,
        resume=RESUME,
        evaluate_every_n_steps=1000,
        checkpoint_after_validation=True,
        train_iters=HYDRATED_TRAIN_ITERS,
        limit_val_iters=1000,
        dummy_batch_mode=DUMMY_BATCH_MODE,
        print_model_parameters=False,
        hf_cache_dir=HYDRATED_HF_CACHE_DIR,
        hf_repo_path=HYDRATED_HF_REPO_PATH,
    ),
)
class Learner(nn.Module):
    def __init__(
        self,
        experiment_name: str,
        root_dir: Union[str, Path],
        model: torch.nn.Module,
        trainer: Trainer,
        evaluator: Evaluator,
        resume: Union[bool, str] = False,
        evaluate_every_n_steps: Optional[int] = None,
        checkpoint_every_n_steps: Optional[int] = None,
        checkpoint_after_validation: Optional[bool] = False,
        train_iters: Optional[int] = None,
        train_dataloader: Optional[DataLoader] = None,
        limit_train_iters: Optional[int] = None,
        val_dataloader: Optional[Union[List[DataLoader], DataLoader]] = None,
        limit_val_iters: Optional[int] = None,
        test_dataloader: Optional[Union[List[DataLoader], DataLoader]] = None,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        print_model_parameters: Optional[bool] = False,
        hf_cache_dir: Optional[str] = None,
        hf_repo_path: Optional[str] = None,
        experiment_tracker: Optional[Run] = None,
        dummy_batch_mode: Optional[bool] = False,
    ):
        """
        Initialize the Learner class.

        :param experiment_name: The name of the experiment.
        :param experiment_dir: The directory for the experiment.
        :param model: The PyTorch model to be trained.
        :param train_dataloader: A list of DataLoaders for training.
        :param val_dataloader: A list of DataLoaders for validation.
        :param test_dataloader: A list of DataLoaders for testing.
        :param trainer: A list of trainer objects for training.
        :param evaluator: A list of evaluator objects for evaluation.
        :param evaluate_every_n_steps: The number of steps between evaluations.
        :param checkpoint_every_n_steps: The number of steps between checkpoints.
        :param checkpoint_after_validation: Whether to save a checkpoint after validation.
        :param train_iters: The number of training iterations.
        :param resume: Whether to resume training from a saved checkpoint.
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.root_dir = (
            root_dir if isinstance(root_dir, Path) else Path(root_dir)
        )
        self.experiment_dir = self.root_dir / experiment_name
        self.hf_cache_dir = hf_cache_dir
        self.hf_repo_path = hf_repo_path
        self.background_threads = []
        self.checkpoints_dir = Path(self.experiment_dir / "checkpoints")
        self.neptune_run = experiment_tracker

        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True)

        if not self.checkpoints_dir.exists():
            self.checkpoints_dir.mkdir(parents=True)

        self.model = model
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_after_validation = checkpoint_after_validation
        self.step_idx = 0
        self.global_step = 0

        self.limit_train_iters = limit_train_iters
        self.limit_val_iters = limit_val_iters
        self.dummy_batch_mode = dummy_batch_mode

        self.train_iters = train_iters

        self.train_dataloader = train_dataloader

        self.val_dataloader = val_dataloader

        self.test_dataloader = test_dataloader

        for name, params in self.model.named_parameters():
            logger.debug(f"{name}, {params.shape}")

        self.callbacks = (
            [callbacks] if isinstance(callbacks, Callback) else callbacks
        )

        if self.callbacks is None:
            self.callbacks = []

        self.callback_handler = CallbackHandler(self.callbacks)

        self.callback_handler.on_init_start(
            experiment=self,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            test_dataloader=self.test_dataloader,
        )

        self.resume = resume

        if self.evaluate_every_n_steps is None:
            self.evaluate_every_n_steps = 99999999999

        self.trainer = trainer
        self.evaluator = evaluator

        self.callback_handler.on_init_end(
            experiment=self,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            test_dataloader=self.test_dataloader,
        )

        self.accelerator = Accelerator()

        self.trainer.optimizer = self.accelerator.prepare(
            self.trainer.optimizer
        )

        if self.trainer.scheduler is not None:
            self.trainer.scheduler = self.accelerator.prepare(
                self.trainer.scheduler
            )

        if self.train_dataloader is not None:
            self.train_dataloader = self.accelerator.prepare(
                self.train_dataloader
            )

        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        if self.test_dataloader is not None:
            self.test_dataloader = self.accelerator.prepare(
                self.test_dataloader
            )

        if isinstance(resume, str):
            checkpoint_path = Path(resume)
            if not checkpoint_path.exists():
                raise ValueError(
                    f"Checkpoint path {checkpoint_path} does not exist, please check your resume= argument"
                )
            self.load_checkpoint(checkpoint_path=checkpoint_path)

        elif isinstance(resume, Path):
            self.load_checkpoint(checkpoint_path=resume)

        if print_model_parameters:
            for key, value in self.named_parameters():
                logger.debug(
                    f"Parameter {key} -> {value.shape} requires grad {value.requires_grad}"
                )

    def run(self):
        self.train()

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        attributes = "\n".join(
            [f"{key}={value}" for key, value in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}\n {attributes}"

    def __str__(self):
        return self.__repr__()

    def training_step(self, model, batch):
        self.callback_handler.on_batch_start(model, batch)
        self.callback_handler.on_training_step_start(model, batch)
        output_list = []

        cur_output_dict = self.trainer.training_step(
            model=model,
            batch=batch,
            global_step=self.global_step,
            accelerator=self.accelerator,
        )
        output_list.append(cur_output_dict)

        self.callback_handler.on_batch_end(model, batch)
        self.callback_handler.on_training_step_end(model, batch)
        self.global_step += 1
        return output_list

    def validation_step(self, model, batch):
        self.callback_handler.on_batch_start(model, batch)
        self.callback_handler.on_validation_step_start(model, batch)

        self.evaluator.validation_step(
            model=model,
            batch=batch,
            global_step=self.global_step,
            accelerator=self.accelerator,
        )

        self.callback_handler.on_batch_end(model, batch)
        self.callback_handler.on_validation_step_end(model, batch)

    def testing_step(self, model, batch):
        self.callback_handler.on_batch_start(model, batch)
        self.callback_handler.on_testing_step_start(model, batch)

        self.evaluator.testing_step(
            model=model,
            batch=batch,
            global_step=self.global_step,
            accelerator=self.accelerator,
        )

        self.callback_handler.on_batch_end(model, batch)
        self.callback_handler.on_testing_step_end(model, batch)

    def start_training(self):
        self.callback_handler.on_train_start(
            experiment=self,
            model=self.model,
        )

        self.trainer.start_training(
            global_step=self.global_step,
        )

        logger.debug("Starting training 🏋🏽")

    def end_training(self):
        self.callback_handler.on_train_end(
            experiment=self,
            model=self.model,
        )

        self.trainer.end_training(global_step=self.global_step)

        while len(self.background_threads) > 0:
            self.check_manage_background_threads()
            sleep(1)

        logger.debug("Training finished 🎉")

    def check_manage_background_threads(self):
        # iterate threads to find up to where they are done, and start the next one
        for thread in self.background_threads:
            if not thread.done:
                if not thread.is_alive():
                    print(f"Starting thread {thread}")
                    thread.start()
                    break
            else:
                self.background_threads.remove(thread)
                print(f"Removing thread {thread} since it is done")

    def start_validation(self):
        self.callback_handler.on_validation_start(
            experiment=self, model=self.model
        )

        self.evaluator.start_validation(
            global_step=self.global_step,
        )

        logger.debug("Starting validation 🧪")

    def end_validation(self):
        self.callback_handler.on_validation_end(
            experiment=self, model=self.model
        )

        self.evaluator.end_validation(
            global_step=self.global_step,
        )

        if self.checkpoint_after_validation:
            logger.debug("Saving checkpoint after validation")
            self.save_checkpoint(checkpoint_name=f"ckpt_{self.global_step}")

        logger.debug("Validation finished 🎉")

    def start_testing(self):
        self.callback_handler.on_testing_start(
            experiment=self, model=self.model
        )

        self.evaluator.start_testing(
            global_step=self.global_step,
        )
        logger.debug("Starting testing 🧪")

    def end_testing(self):
        self.callback_handler.on_testing_end(
            experiment=self,
            model=self.model,
        )

        self.evaluator.end_testing(
            global_step=self.global_step,
        )

        logger.debug("Testing finished 🎉")

    def train(self, train_dataloader: DataLoader = None):
        if train_dataloader is not None:
            train_dataloader = self.accelerator.prepare(train_dataloader)
            self.train_dataloader = train_dataloader

        self._training_loop(train_dataloader=self.train_dataloader)

    def validate(
        self, val_dataloader: List[DataLoader] = None, model: nn.Module = None
    ):
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)
            self.val_dataloader = val_dataloader

        model = self.accelerator.prepare(model)
        self._validation_loop(val_dataloader=self.val_dataloader, model=model)

    def test(
        self, test_dataloader: List[DataLoader] = None, model: nn.Module = None
    ):
        if test_dataloader is not None:
            test_dataloader = self.accelerator.prepare(test_dataloader)
            self.test_dataloader = test_dataloader

        if model is None:
            if self.evaluator[0].model_selection_metric_name is not None:
                self.load_best_model(
                    metric_name=self.evaluator[0].model_selection_metric_name,
                    higher_is_better=self.evaluator[
                        0
                    ].model_selection_metric_higher_is_better,
                )
            model = self.accelerator.prepare(self.model)

        self._testing_loop(
            test_dataloader=self.test_dataloader,
            model=model,
        )

    def _training_loop(self, train_dataloader: DataLoader = None):
        if train_dataloader is None:
            train_dataloader = self.train_dataloader

        if train_dataloader is not None:
            self.start_training()

            if self.train_iters is None:
                self.train_iters = len(train_dataloader)

            with tqdm(
                initial=self.step_idx, total=self.train_iters
            ) as pbar_steps:
                while self.step_idx < self.train_iters:
                    if self.limit_train_iters is not None:
                        if self.step_idx >= self.limit_train_iters:
                            return self.end_training()

                    for batch_idx, batch in enumerate(train_dataloader):
                        output_list = self.training_step(
                            model=self.model,
                            batch=batch,
                        )

                        if (
                            self.checkpoint_every_n_steps is not None
                            and self.step_idx % self.checkpoint_every_n_steps
                            == 0
                            and self.step_idx > 0
                        ):
                            self.save_checkpoint(
                                checkpoint_name=f"ckpt_{self.global_step}"
                            )

                        if self.step_idx % self.evaluate_every_n_steps == 0:
                            self._validation_loop()
                            self.check_manage_background_threads()

                        if self.step_idx >= self.train_iters:
                            return self.end_training()

                        self.step_idx += 1

                        loss = torch.mean(
                            torch.stack(
                                [
                                    value
                                    for item in output_list
                                    for key, value in item.__dict__.items()
                                    if "opt_loss" in key
                                ]
                            )
                        )

                        pbar_steps.update(1)
                        pbar_steps.set_description(f"Loss: {loss:.4f}")

            return self.end_training()

    def _validation_loop(
        self, val_dataloader: List[DataLoader] = None, model: nn.Module = None
    ):
        if val_dataloader is None:
            val_dataloader = self.val_dataloader

        if model is None:
            model = self.model

        if val_dataloader is not None:
            self.start_validation()

            with tqdm(total=len(val_dataloader)) as pbar_dataloaders:
                pre_batch_time = time.time()
                for batch_idx, batch in enumerate(val_dataloader):
                    if self.limit_val_iters is not None:
                        if batch_idx >= self.limit_val_iters:
                            break
                    post_batch_time = time.time()
                    logger.debug(
                        f"Batch {batch_idx} loaded in {post_batch_time - pre_batch_time} seconds"
                    )
                    self.validation_step(
                        model=model,
                        batch=batch,
                    )
                    pbar_dataloaders.update(1)
                    pre_batch_time = time.time()

            self.end_validation()

    def _testing_loop(
        self,
        test_dataloader: List[DataLoader] = None,
        model: nn.Module = None,
    ):
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        if model is None:
            model = self.model

        if test_dataloader is not None:
            self.start_testing()

            with tqdm(total=len(test_dataloader)) as pbar_dataloaders:
                for batch_idx, batch in enumerate(test_dataloader):
                    self.testing_step(
                        model=model,
                        batch=batch,
                    )
                    pbar_dataloaders.update(1)

            self.end_testing()

    def save_checkpoint(
        self,
        checkpoint_name: str,
    ):
        ckpt_save_path = self.checkpoints_dir / checkpoint_name

        if not ckpt_save_path.exists():
            ckpt_save_path.mkdir(parents=True)

        experiment_hyperparameters = dict(
            step_idx=self.step_idx,
            global_step=self.global_step,
            current_epoch_dict={
                "train": self.trainer.current_epoch_dict,
                "eval": self.evaluator.current_epoch_dict,
            },
            per_epoch_metrics={
                "eval": self.evaluator.current_epoch_dict,
            },
            neptune_id=self.neptune_run._id if self.neptune_run else None,
        )

        torch.save(
            obj=experiment_hyperparameters,
            f=ckpt_save_path / "trainer_state.pt",
        )
        self.accelerator.save_state(ckpt_save_path)
        logger.debug(f"Saved checkpoint to {ckpt_save_path}")
        self.callback_handler.on_save_checkpoint(
            model=self.model,
            optimizer=self.trainer.optimizer,
            experiment=self,
            checkpoint_path=ckpt_save_path,
        )

        return ckpt_save_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ):
        checkpoint_path = (
            checkpoint_path
            if isinstance(checkpoint_path, Path)
            else Path(checkpoint_path)
        )

        if not (pathlib.Path(checkpoint_path) / "trainer_state.pt").exists():
            return

        logger.debug(f"Loading checkpoint from {checkpoint_path}")

        trainer_state = torch.load(
            pathlib.Path(checkpoint_path) / "trainer_state.pt"
        )
        self.step_idx = trainer_state["step_idx"]
        self.global_step = trainer_state["global_step"]
        current_epoch_dict = trainer_state["current_epoch_dict"]
        per_epoch_metrics = trainer_state["per_epoch_metrics"]

        if isinstance(current_epoch_dict["train"], List):
            loaded_trainer_epoch_dict = current_epoch_dict["train"][0]
        else:
            loaded_trainer_epoch_dict = current_epoch_dict["train"]

        if isinstance(current_epoch_dict["eval"], List):
            loaded_evaluator_epoch_dict = current_epoch_dict["eval"][0]
        else:
            loaded_evaluator_epoch_dict = current_epoch_dict["eval"]

        if isinstance(per_epoch_metrics["eval"], List):
            loaded_evaluator_per_epoch_metrics = per_epoch_metrics["eval"][0]
        else:
            loaded_evaluator_per_epoch_metrics = per_epoch_metrics["eval"]

        setattr(
            self.trainer,
            "current_epoch_dict",
            loaded_trainer_epoch_dict,
        )

        setattr(
            self.evaluator,
            "current_epoch_dict",
            loaded_evaluator_epoch_dict,
        )

        setattr(
            self.evaluator,
            "per_epoch_metrics",
            loaded_evaluator_per_epoch_metrics,
        )

        self.accelerator.load_state(checkpoint_path)

        self.callback_handler.on_load_checkpoint(
            model=self.model,
            optimizer=self.trainer.optimizer,
            experiment=self,
            checkpoint_path=checkpoint_path,
        )

    def load_best_model(self, metric_name: str, higher_is_better: bool):
        best_global_step, best_metric = self.evaluator[
            0
        ].get_best_model_global_step_and_metric(metric_name, higher_is_better)
        print(
            f"Best {metric_name}: {best_metric} at step {best_global_step}, downloading model..."
        )
        print(
            f"hf_repo_path: {self.hf_repo_path}, hf_cache_dir: {self.hf_cache_dir}, model_name: ckpt_{best_global_step}"
        )
        download_dict = download_model_with_name(
            hf_repo_path=self.hf_repo_path,
            hf_cache_dir=self.hf_cache_dir,
            model_name=f"ckpt_{best_global_step}",
        )

        self.load_checkpoint(checkpoint_path=download_dict["root_filepath"])
