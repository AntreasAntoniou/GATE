import copy
import logging
import pathlib
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from zmq import has

from gate.boilerplate.callbacks import Callback, CallbackHandler
from gate.boilerplate.decorators import configurable
from gate.boilerplate.utils import download_model_with_name
from gate.config.variables import (
    DUMMY_BATCH_MODE,
    HYDRATED_CURRENT_EXPERIMENT_DIR,
    HYDRATED_EXPERIMENT_NAME,
    HYDRATED_HF_CACHE_DIR,
    HYDRATED_HF_REPO_PATH,
    HYDRATED_TRAIN_ITERS,
    RESUME,
)
from gate.models.core import Ensemble, GATEModel
from gate.orchestration.evaluators.classification import Evaluator
from gate.orchestration.trainers.classification import Trainer

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Union


class ExperimentStatus(Enum):
    COMPLETED: str = "completed"
    TRAINING: str = "training"
    TESTING: str = "testing"
    STARTING: str = "starting"


@configurable(
    group="learner",
    name="default",
    defaults=dict(
        model=None,
        experiment_name=HYDRATED_EXPERIMENT_NAME,
        root_dir=HYDRATED_CURRENT_EXPERIMENT_DIR,
        resume=RESUME,
        evaluate_every_n_steps=250,
        checkpoint_after_validation=True,
        train_iters=HYDRATED_TRAIN_ITERS,
        limit_val_iters=None,
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
        accelerator: Accelerator,
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
        self.accelerator = accelerator
        self.root_dir = (
            root_dir if isinstance(root_dir, Path) else Path(root_dir)
        )
        self.experiment_dir = self.root_dir / experiment_name
        self.hf_cache_dir = hf_cache_dir
        self.hf_repo_path = hf_repo_path
        self.background_threads = []
        self.checkpoints_dir = Path(self.experiment_dir / "checkpoints")

        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True)

        if not self.checkpoints_dir.exists():
            self.checkpoints_dir.mkdir(parents=True)

        self.model = model
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_after_validation = checkpoint_after_validation
        self.status = ExperimentStatus.STARTING
        self.global_step = 0

        self.limit_train_iters = limit_train_iters
        self.limit_val_iters = limit_val_iters
        self.dummy_batch_mode = dummy_batch_mode

        self.train_iters = train_iters

        self.train_dataloader = train_dataloader

        self.val_dataloader = val_dataloader

        self.test_dataloader = test_dataloader

        self.trainer = trainer
        self.evaluator = evaluator

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

        self.trainer = trainer
        self.evaluator = evaluator

        self.callback_handler.on_init_end(
            experiment=self,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            test_dataloader=self.test_dataloader,
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
        model = model.train()
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

        if cur_output_dict.global_step != self.global_step:
            self.global_step = cur_output_dict.global_step + 1
        else:
            self.global_step += 1

        return output_list

    def validation_step(self, model, batch):
        model = model.eval()
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

    def testing_step(self, model, batch, prefix):
        model = model.eval()
        self.callback_handler.on_batch_start(model, batch)
        self.callback_handler.on_testing_step_start(model, batch)

        self.evaluator.testing_step(
            model=model,
            batch=batch,
            global_step=self.global_step,
            accelerator=self.accelerator,
            prefix=prefix,
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

        logger.debug("Training finished 🎉")

    def check_manage_background_threads(self):
        # iterate threads to find up to where they are done, and start the next one
        TIME_LIMIT = 3 * 60  # 60 minutes
        STOP_THREAD_FLAG = "_stop_thread"

        for thread in self.background_threads:
            if not thread.done:
                if not thread.is_alive() and not thread.started:
                    logger.info(f"Starting thread {thread}")
                    thread.start()

                else:
                    # Check if the thread has been running for too long
                    elapsed_time = time.time() - thread.start_time
                    if elapsed_time > TIME_LIMIT:
                        logger.info(
                            f"Thread {thread} has been running for too long. Stopping it."
                        )
                        setattr(thread, STOP_THREAD_FLAG, True)
                        setattr(thread, "done", True)
                        # The thread should stop itself upon checking the STOP_THREAD_FLAG
            else:
                self.background_threads.remove(thread)
                logger.info(f"Removing thread {thread} since it is done")

    def complete_background_threads(self):
        # iterate threads to find up to where they are done, and start the next one
        TIME_LIMIT = 180  # 10 minutes
        STOP_THREAD_FLAG = "_stop_thread"

        while self.background_threads:
            for thread in self.background_threads:
                if not thread.done:
                    if not thread.is_alive() and not thread.started:
                        logger.info(f"Starting thread {thread}")
                        thread.start()
                        break
                    else:
                        # Check if the thread has been running for too long
                        elapsed_time = time.time() - thread.start_time
                        if elapsed_time > TIME_LIMIT:
                            logger.info(
                                f"Thread {thread} has been running for too long. Stopping it."
                            )
                            setattr(thread, STOP_THREAD_FLAG, True)
                            setattr(thread, "done", True)
                            # The thread should stop itself upon checking the STOP_THREAD_FLAG
                else:
                    self.background_threads.remove(thread)
                    logger.info(f"Removing thread {thread} since it is done")

            time.sleep(1)  # Prevent the loop from consuming too much CPU usage

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

        self.check_manage_background_threads()

        logger.debug("Validation finished 🎉")

    def start_testing(self, prefix):
        self.complete_background_threads()
        self.callback_handler.on_testing_start(
            experiment=self, model=self.model
        )

        self.evaluator.start_testing(
            global_step=self.global_step, prefix=prefix
        )
        logger.debug("Starting testing 🧪")

    def end_testing(self, prefix, model):
        self.callback_handler.on_testing_end(
            experiment=self,
            model=model,
        )

        self.evaluator.end_testing(
            global_step=self.global_step, model=model, prefix=prefix
        )

        self.check_manage_background_threads()

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
        self,
        test_dataloader: List[DataLoader] = None,
        model: nn.Module = None,
        prefix: Optional[str] = None,
        force: Optional[bool] = False,
    ):
        if self.status == ExperimentStatus.COMPLETED and not force:
            return

        if test_dataloader is not None:
            test_dataloader = self.accelerator.prepare(test_dataloader)
            self.test_dataloader = test_dataloader
        base_model = copy.deepcopy(self.model)
        base_evaluator = copy.deepcopy(self.evaluator)

        if model is None:
            for kth in [1, 3, 5]:
                if self.evaluator.model_selection_metric_name is not None:
                    model = self.load_best_model(
                        metric_name=base_evaluator.model_selection_metric_name,
                        higher_is_better=base_evaluator.model_selection_metric_higher_is_better,
                        kth_best=kth,
                        base_model=base_model,
                        evaluator=base_evaluator,
                    )
                model = self.accelerator.prepare(model)

                self._testing_loop(
                    test_dataloader=self.test_dataloader,
                    model=model,
                    prefix=f"ensemble_{kth}",
                )
        else:
            model = self.accelerator.prepare(model)

            self._testing_loop(
                test_dataloader=self.test_dataloader,
                model=model,
                prefix=prefix,
            )

        self.save_checkpoint(
            checkpoint_name=f"ckpt_{self.global_step}",
            status=ExperimentStatus.COMPLETED,
        )

    def _finalize_training(self):
        # self._validation_loop()
        # self.save_checkpoint(checkpoint_name=f"ckpt_{self.global_step}")
        return self.end_training()

    def _training_loop(self, train_dataloader: DataLoader = None):
        if (
            self.status == ExperimentStatus.TESTING
            or self.status == ExperimentStatus.COMPLETED
        ) and self.global_step >= self.train_iters:
            return self._finalize_training()

        if train_dataloader is None:
            train_dataloader = self.train_dataloader

        last_val_step = 0

        if train_dataloader is not None:
            self.start_training()

            if self.train_iters is None:
                self.train_iters = len(train_dataloader)

            if self.limit_train_iters is not None:
                self.train_iters = min(
                    self.train_iters, self.limit_train_iters
                )

            with tqdm(
                initial=self.global_step, total=self.train_iters, smoothing=0.0
            ) as pbar_steps:
                while self.global_step <= self.train_iters:
                    tqdm_iter = self.global_step

                    for batch_idx, batch in enumerate(train_dataloader):
                        if self.global_step > self.train_iters:
                            break
                        if (
                            self.global_step % self.evaluate_every_n_steps == 0
                            or self.global_step == 0
                        ):
                            self._validation_loop()

                        output_list = self.training_step(
                            model=self.model,
                            batch=batch,
                        )

                        if (
                            self.checkpoint_every_n_steps is not None
                            and (
                                self.global_step
                                % self.checkpoint_every_n_steps
                                == 0
                            )
                            and self.global_step > 0
                        ):
                            self.save_checkpoint(
                                checkpoint_name=f"ckpt_{self.global_step}",
                                status=ExperimentStatus.TRAINING
                                if self.global_step < self.train_iters
                                else ExperimentStatus.TESTING,
                            )

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
                        tqdm_update = self.global_step - tqdm_iter
                        tqdm_iter = self.global_step
                        pbar_steps.update(tqdm_update)
                        pbar_steps.set_description(f"Loss: {loss:.4f}")

        self.save_checkpoint(
            checkpoint_name=f"ckpt_{self.global_step}",
            status=ExperimentStatus.TESTING,
        )
        return self._finalize_training()

    def _validation_loop(
        self, val_dataloader: List[DataLoader] = None, model: nn.Module = None
    ):
        if val_dataloader is None:
            val_dataloader = self.val_dataloader

        if model is None:
            model = self.model

        model = model.eval()

        if val_dataloader is not None:
            self.start_validation()

            with tqdm(
                total=len(val_dataloader), smoothing=0.0
            ) as pbar_dataloaders:
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
        prefix: Optional[str] = None,
    ):
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        if model is None:
            model = self.model
        model = model.eval()

        if test_dataloader is not None:
            self.start_testing(prefix=prefix)

            with tqdm(
                total=len(test_dataloader), smoothing=0.0
            ) as pbar_dataloaders:
                for batch_idx, batch in enumerate(test_dataloader):
                    self.testing_step(
                        model=model,
                        batch=batch,
                        prefix=prefix,
                    )
                    pbar_dataloaders.update(1)

            self.end_testing(prefix=prefix, model=model)

    def save_checkpoint(
        self,
        checkpoint_name: str,
        status: ExperimentStatus = ExperimentStatus.TRAINING,
    ):
        ckpt_save_path = self.checkpoints_dir / checkpoint_name

        if not ckpt_save_path.exists():
            ckpt_save_path.mkdir(parents=True)

        experiment_hyperparameters = dict(
            step_idx=self.global_step,
            global_step=self.global_step,
            current_epoch_dict={
                "train": self.trainer.current_epoch_dict,
                "eval": self.evaluator.current_epoch_dict,
            },
            per_epoch_metrics={
                "eval": self.evaluator.per_epoch_metrics,
            },
            status=status,
        )

        torch.save(
            obj=experiment_hyperparameters,
            f=ckpt_save_path / "trainer_state.pt",
        )
        self.accelerator.save_state(ckpt_save_path)

        self.callback_handler.on_save_checkpoint(
            model=self.model,
            optimizer=self.trainer.optimizer,
            experiment=self,
            checkpoint_path=ckpt_save_path,
        )
        self.status = status

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
            return ExperimentStatus.STARTING

        trainer_state = torch.load(
            pathlib.Path(checkpoint_path) / "trainer_state.pt"
        )
        self.global_step = trainer_state["step_idx"]
        self.global_step = trainer_state["global_step"]
        current_epoch_dict = trainer_state["current_epoch_dict"]
        per_epoch_metrics = trainer_state["per_epoch_metrics"]

        if "status" in trainer_state:
            self.status = trainer_state["status"]
            logger.info(f"Found status {self.status}")
        else:
            self.status = ExperimentStatus.STARTING
            logger.info(f"Could not find status, setting to {self.status}")

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

        return self.status

    def load_best_model(
        self,
        metric_name: str,
        higher_is_better: bool,
        kth_best: int,
        base_model: nn.Module,
        evaluator: Evaluator = None,
    ):
        (
            best_global_step,
            best_metric,
        ) = evaluator.get_best_model_global_step_and_metric(
            metric_name, higher_is_better, kth_best=10
        )
        logger.info(
            f"Best {metric_name}: {best_metric} at step {best_global_step}, downloading model..."
        )
        logger.info(
            f"hf_repo_path: {self.hf_repo_path}, hf_cache_dir: {self.hf_cache_dir}, model_name: ckpt_{best_global_step}"
        )

        download_dict_list = []
        for global_step in best_global_step:
            download_dict = download_model_with_name(
                hf_repo_path=self.hf_repo_path,
                hf_cache_dir=self.hf_cache_dir,
                model_name=f"ckpt_{global_step}",
                local_checkpoint_store_dir=self.checkpoints_dir,
            )
            if download_dict["validation_passed"] is True:
                download_dict_list.append(download_dict)

            if len(download_dict_list) == kth_best:
                break

        models = []

        for download_dict in download_dict_list:
            # Create a new instance of the model architecture
            model = copy.deepcopy(base_model)

            # Load the state dictionary
            state_dict = torch.load(
                download_dict["model_filepath"], map_location="cpu"
            )

            # Use Accelerate's state_dict loading
            model.load_state_dict(state_dict)

            models.append(model.model)

        model = GATEModel(
            config=base_model.config,
            model=Ensemble(models=models),
        )
        model = self.accelerator.prepare(model)

        return model
