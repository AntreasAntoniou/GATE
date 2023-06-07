import logging
import os
import sys
import threading
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from huggingface_hub import HfApi
from hydra_zen import instantiate
from torch.utils.data import DataLoader

from .utils import get_logger

logger = get_logger(__name__)
hf_logger = get_logger("huggingface_hub", logging_level=logging.CRITICAL)


class Callback(ABC):
    def __init__(self) -> None:
        pass

    def on_init_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_init_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_phase_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_phase_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_batch_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_batch_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_training_step_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_training_step_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_validation_step_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_validation_step_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_testing_step_start(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_testing_step_end(self, model: nn.Module, batch: Dict) -> None:
        pass

    def on_train_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_train_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_validation_start(
        self,
        experiment: Any,
        model: nn.Module,
    ):
        pass

    def on_validation_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_testing_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_testing_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        pass

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizer: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        pass

    def on_load_checkpoint(
        self,
        model: nn.Module,
        optimizer: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        pass


def instantiate_callbacks(callback_dict: dict) -> List[Callback]:
    callbacks = []
    for cb_conf in callback_dict.values():
        callbacks.append(instantiate(cb_conf))

    return callbacks


class CallbackHandler(Callback):
    def __init__(self, callbacks: List[Callback]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def on_init_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_init_start(
                experiment,
                model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            )

    def on_init_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_init_end(
                experiment,
                model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            )

    def on_phase_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_phase_start(
                experiment,
                model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            )

    def on_phase_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloader: Union[List[DataLoader], DataLoader] = None,
        test_dataloader: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_phase_end(
                experiment,
                model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            )

    def on_batch_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_batch_start(model, batch)

    def on_batch_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(model, batch)

    def on_training_step_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_training_step_start(model, batch)

    def on_training_step_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_training_step_end(model, batch)

    def on_validation_step_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_validation_step_start(model, batch)

    def on_validation_step_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_validation_step_end(model, batch)

    def on_testing_step_start(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_testing_step_start(model, batch)

    def on_testing_step_end(self, model: nn.Module, batch: Dict) -> None:
        for callback in self.callbacks:
            callback.on_testing_step_end(model, batch)

    def on_train_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_start(experiment, model)

    def on_train_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_end(
                experiment,
                model,
            )

    def on_validation_start(
        self,
        experiment: Any,
        model: nn.Module,
    ):
        for callback in self.callbacks:
            callback.on_validation_start(experiment, model)

    def on_validation_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(experiment, model)

    def on_testing_start(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_testing_start(experiment, model)

    def on_testing_end(
        self,
        experiment: Any,
        model: nn.Module,
    ) -> None:
        for callback in self.callbacks:
            callback.on_testing_end(experiment, model)

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizer: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        for callback in self.callbacks:
            callback.on_save_checkpoint(
                model, optimizer, experiment, checkpoint_path
            )

    def on_load_checkpoint(
        self,
        model: nn.Module,
        optimizer: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        for callback in self.callbacks:
            callback.on_load_checkpoint(
                model, optimizer, experiment, checkpoint_path
            )


import time


class UploadCheckpointToHuggingFaceBackground(threading.Thread):
    def __init__(
        self,
        repo_name: str,
        repo_owner: str,
        checkpoint_path: Path,
        timeout: int = 10 * 60,
    ):
        super().__init__()
        self.repo_name = repo_name
        self.repo_owner = repo_owner
        self.checkpoint_path = checkpoint_path
        self.hf_api = HfApi(token=os.environ["HF_TOKEN"])
        self.done = False
        self.should_stop = False  # Flag to indicate the thread should stop
        self.timeout = timeout  # Timeout in seconds
        self.start_time = None

    def run(self):
        try:
            self.hf_api.upload_folder(
                repo_id=f"{self.repo_owner}/{self.repo_name}",
                folder_path=self.checkpoint_path,
                path_in_repo=f"checkpoints/{self.checkpoint_path.name}",
            )

            self.done = True

        except Exception as e:
            logger.info(e)

    def start_with_timeout(self):
        self.start()


class UploadCheckpointsToHuggingFace(Callback):
    def __init__(self, repo_name: str, repo_owner: str):
        from huggingface_hub import HfApi

        super().__init__()
        self.repo_name = repo_name
        self.repo_owner = repo_owner
        self.hf_api = HfApi()

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        background_upload_thread = UploadCheckpointToHuggingFaceBackground(
            repo_name=self.repo_name,
            repo_owner=self.repo_owner,
            checkpoint_path=checkpoint_path,
        )
        experiment.background_threads.append(background_upload_thread)
