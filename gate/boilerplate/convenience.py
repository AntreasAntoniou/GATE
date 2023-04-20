from typing import Any, Dict, Optional

import accelerate
import torch
import torch.nn as nn

from gate.boilerplate.utils import get_logger

logger = get_logger(__name__)


def set_batch_size(batch: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    """
    Sets the batch size for a given input batch.

    Args:
        batch (Dict[str, Any]): The input batch.
        batch_size (int): The desired batch size.

    Returns:
        Dict[str, Any]: The input batch with the specified batch size.
    """
    return {
        key: value[0]
        .unsqueeze(0)
        .repeat([batch_size, *[1] * (value.dim() - 1)])
        if isinstance(value, torch.Tensor)
        else [value[0]] * batch_size
        for key, value in batch.items()
    }


def get_max_supported_batch_size(
    model: nn.Module,
    batch: Dict[str, Any],
    accelerator: Optional[accelerate.Accelerator] = None,
    train_mode: bool = False,
) -> int:
    """
    Finds the maximum supported batch size for the given model and batch.

    Args:
        model (nn.Module): The model to test.
        batch (Dict[str, Any]): The input batch.
        accelerator (Optional[accelerate.Accelerator], optional):
        The accelerator for model and batch preparation. Defaults to None.
        train_mode (bool, optional): If True, use training mode.
        Defaults to False.

    Returns:
        int: The maximum supported batch size.
    """
    # Create accelerator if not provided 🚀
    if accelerator is None:
        accelerator = accelerate.Accelerator()

    # Set model mode based on train_mode flag 🚦
    if train_mode:
        model.train()
    else:
        model.eval()

    batch_size = 16
    crashed = False

    for key, value in batch.items():
        print(key, value.shape if isinstance(value, torch.Tensor) else value)

    # Iterate and test different batch sizes until crash 🛠️
    while not crashed:
        logger.debug(f"Trying batch size {batch_size}... 👾")
        try:
            if batch is not None:
                cur_batch = set_batch_size(batch, batch_size)

                # Perform dummy forward and backward passes with the current batch size 📊
                if not train_mode:
                    with torch.no_grad():
                        dummy_fprop_bprop(model, cur_batch, accelerator)
                else:
                    dummy_fprop_bprop(
                        model, cur_batch, accelerator, do_bprop=True
                    )
        except Exception as e:
            crashed = True
            batch_size = batch_size // 2
            logger.debug(f"Batch size {batch_size} crashed with error: {e} 🤯")
            logger.info(f"Max supported batch size: {batch_size} 🎉")
            return batch_size

        # Double the batch size for the next iteration 📈
        batch_size = batch_size * 2
