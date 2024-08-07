import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def contrastive_accuracy(logits, is_irregular_shape: bool = False):
    if is_irregular_shape:
        logits = logits.reshape(logits.shape[0], -1, 2, 2).reshape(-1, 2, 2)
        targets = (
            torch.arange(2)
            .to(logits.device)
            .unsqueeze(0)
            .repeat([logits.shape[0], 1])
        )
        logits = logits.reshape(-1, 2)
        targets = targets.reshape(-1)
    else:
        targets = torch.arange(logits.shape[0]).to(logits.device)
    return (logits.argmax(dim=-1) == targets).float().mean()


def contrastive_accuracy_top_k(
    logits, k: int = 5, is_irregular_shape: bool = False
):
    if is_irregular_shape:
        logits = logits.reshape(logits.shape[0], -1, 2, 2).reshape(-1, 2, 2)
        targets = (
            torch.arange(2)
            .to(logits.device)
            .unsqueeze(0)
            .repeat([logits.shape[0], 1])
        )
        logits = logits.reshape(-1, 2)
        targets = targets.reshape(-1)
    else:
        targets = torch.arange(logits.shape[0]).to(logits.device)

    num_classes = logits.shape[-1]
    k = min(k, num_classes)  # Adjust k to be within the valid range

    top_k_indices = torch.topk(logits, k, dim=-1).indices
    targets_match = top_k_indices == targets.view(-1, 1)
    accuracy = torch.mean(targets_match.any(dim=-1).float())

    return accuracy


def num_parameters(
    model, only_trainable: bool = False, exclude_embeddings: bool = False
) -> int:
    """
    Get number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
        only_trainable (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of trainable parameters

        exclude_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of non-embeddings parameters

    Returns:
        `int`: The number of parameters.
    """

    if exclude_embeddings:
        embedding_param_names = [
            f"{name}.weight"
            for name, module_type in model.named_modules()
            if isinstance(module_type, nn.Embedding)
        ]
        non_embedding_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name not in embedding_param_names
        ]
        return sum(
            p.numel()
            for p in non_embedding_parameters
            if p.requires_grad or not only_trainable
        )
    else:
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad or not only_trainable
        )


def get_device():
    return torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def contrastive_loss(logits, is_irregular_shape: bool = False):
    if is_irregular_shape:
        logits = logits.reshape(logits.shape[0], -1, 2, 2).reshape(-1, 2, 2)
        targets = (
            torch.arange(2)
            .to(logits.device)
            .unsqueeze(0)
            .repeat([logits.shape[0], 1])
        )
        logits = logits.reshape(-1, 2)
        targets = targets.reshape(-1)
    else:
        targets = torch.arange(len(logits), device=logits.device)

    return nn.functional.cross_entropy(logits, targets)


def get_similarities(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature_parameter: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        tensor_modality_a: Tensor, shape [seq_len, embedding_dim] or [batch_size, seq_len, embedding_dim]
        tensor_modality_b: Tensor, shape [seq_len, embedding_dim] or [batch_size, seq_len, embedding_dim]
    """

    image_features = image_features / image_features.norm(
        p=2, dim=-1, keepdim=True
    )
    text_features = text_features / text_features.norm(
        p=2, dim=-1, keepdim=True
    )

    similarity = F.linear(image_features, text_features) * torch.clamp(
        temperature_parameter.exp(), max=100
    )

    similarities = {f"image_to_text_similarities": similarity}

    similarities[f"text_to_image_similarities"] = similarity.T

    return similarities


def compute_zero_shot_loss_and_metrics(
    similarities: Dict[str, torch.Tensor],
    is_irregular_shape: bool = False,
):
    if isinstance(is_irregular_shape, List):
        is_irregular_shape = is_irregular_shape[0]

    contrastive_losses_dict = {
        f"{key.replace('_similarities', '_loss')}": contrastive_loss(
            value, is_irregular_shape=is_irregular_shape
        )
        for key, value in similarities.items()
    }

    loss = torch.mean(torch.stack(list(contrastive_losses_dict.values())))

    contrastive_accuracy_dict = {
        f"{key.replace('_similarities', '_accuracy')}": contrastive_accuracy(
            value, is_irregular_shape=is_irregular_shape
        )
        for key, value in similarities.items()
    }

    contrastive_accuracy_top_5_dict = {
        f"{key.replace('_similarities', '_accuracy_top_5')}": contrastive_accuracy_top_k(
            value, k=5, is_irregular_shape=is_irregular_shape
        )
        for key, value in similarities.items()
    }

    return (
        similarities
        | contrastive_losses_dict
        | contrastive_accuracy_dict
        | contrastive_accuracy_top_5_dict
        | {"is_irregular_shape": is_irregular_shape, "loss": loss}
    )


def extract_all_possible_pairs(batch_dict):
    from itertools import combinations

    modality_dict = {}
    for key, value in batch_dict.items():
        if isinstance(value, dict) and key != "other":
            modality_dict[key] = list(value.keys())

    pairs_keys = combinations(list(modality_dict.keys()), 2)

    # get all possible pairs of two lists
    pairs = []
    for key1, key2 in pairs_keys:
        for sub_key1, sub_key2 in zip(
            modality_dict[key1], modality_dict[key2]
        ):
            pairs.append((key1, sub_key1, key2, sub_key2))

    return pairs


def reinit(input_module: nn.Module):
    for name, module in input_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose1d):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
