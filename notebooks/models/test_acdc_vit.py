import accelerate
import fire
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import gate.data.medical.segmentation.automated_cardiac_diagnosis as acd
import gate.data.medical.segmentation.medical_decathlon as md
from gate.models.task_adapters.medical_semantic_segmentation import logger
from gate.models.task_specific_models.semantic_segmentation.timm import (
    build_gate_model,
)

logger.setLevel("DEBUG")


def sub_batch_generator(batch_dict, sub_batch_size):
    """
    Generator function to yield sub-batches from a given batch dictionary.

    Parameters:
    - batch_dict (dict): Dictionary containing original batch data. Each key maps to a tensor or list.
    - sub_batch_size (int): Size of each sub-batch to be returned.

    Yields:
    - dict: Dictionary containing sub-batch data.
    """
    batch_size = None

    # Validate input and get original batch size
    for key, value in batch_dict.items():
        # print(f"key: {key}, value.shape: {value.shape}")
        if batch_size is None:
            batch_size = value.shape[0] * value.shape[1]
        elif batch_size != value.shape[0] * value.shape[1]:
            raise ValueError(
                f"Batch sizes for different keys in batch_dict must be the same. Mismatch at key: {key}"
            )

    # Generate and yield sub-batches
    for start_idx in range(0, batch_size, sub_batch_size):
        end_idx = min(start_idx + sub_batch_size, batch_size)
        sub_batch = {}

        for key, value in batch_dict.items():
            sub_batch[key] = value.reshape(-1, *value.shape[2:])[
                start_idx:end_idx
            ]

        yield sub_batch


def build_dataloader(
    dataset_name,
    data_dir,
    image_size,
    target_image_size,
    transforms,
    batch_size=1,
    num_workers=12,
):
    if dataset_name == "md":
        data = md.build_gate_dataset(
            data_dir=data_dir,
            image_size=image_size,
            label_image_size=target_image_size,
            transforms=transforms,
        )
    elif dataset_name == "acdc":
        data = acd.build_gate_dataset(
            data_dir=data_dir,
            image_size=image_size,
            target_image_size=target_image_size,
            transforms=transforms,
        )
    else:
        raise ValueError("Invalid dataset name")

    return DataLoader(
        data["val"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def integrate_output_list(output_list):
    def accumulate_outputs(output_accumulator, output_dict):
        for key, value in output_dict.items():
            if isinstance(value, dict):
                if key not in output_accumulator:
                    output_accumulator[key] = {}
                accumulate_outputs(output_accumulator[key], value)
            else:
                if isinstance(value, torch.Tensor):
                    if key not in output_accumulator:
                        output_accumulator[key] = []
                    output_accumulator[key].append(value.detach().cpu())

    output_accumulator = {}
    for output_dict in output_list:
        accumulate_outputs(output_accumulator, output_dict)

    def concatenate_tensors(nested_dict):
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                concatenate_tensors(value)
            else:
                if isinstance(value, list) and len(value) > 0:
                    if value[0].dim() == 0:
                        # Handle scalar tensors
                        nested_dict[key] = torch.cat(
                            [v.unsqueeze(0) for v in value], dim=0
                        )
                    else:
                        # Handle non-scalar tensors
                        nested_dict[key] = torch.cat(value, dim=0)

    concatenate_tensors(output_accumulator)
    return output_accumulator


def detach_and_move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {
            key: detach_and_move_to_cpu(value) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [detach_and_move_to_cpu(element) for element in obj]
    else:
        return obj


def main(
    dataset_name: str = "md",
    data_dir: str = "/data/",
    image_size: int = 512,
    target_image_size: int = 256,
    batch_size: int = 1,
    sub_batch_size: int = 1,
    num_workers: int = 8,
    eval_mode: bool = False,
):
    model_and_transform = build_gate_model(
        timm_model_name="vit_base_patch16_clip_224.laion2b",
        num_classes=100,
        pretrained=True,
        use_temporal_model=True,
        num_channels=3,
    )

    model = model_and_transform.model
    transforms = model_and_transform.transform
    dataloader = build_dataloader(
        dataset_name,
        data_dir,
        image_size,
        target_image_size,
        transforms,
        batch_size,
        num_workers,
    )

    accelerator = accelerate.Accelerator(
        mixed_precision="fp16",
    )
    model = accelerator.prepare(model)
    dataloader = accelerator.prepare(dataloader)
    optimizer = transformers.AdamW(
        model.parameters(), lr=1e-5, weight_decay=0.0
    )
    optimizer = accelerator.prepare(optimizer)

    with tqdm(total=len(dataloader)) as pbar:
        output_list = []
        for input_dict in dataloader:
            optimizer.zero_grad()
            for batch in sub_batch_generator(input_dict, sub_batch_size):
                if eval_mode:
                    with torch.no_grad():
                        output = model.forward(batch)
                        loss = output["image"]["image"]["loss"]
                else:
                    output = model.forward(batch)
                    loss = output["image"]["image"]["loss"]
                    accelerator.backward(loss)

                output = detach_and_move_to_cpu(output)

                output_list.append(output)

            if not eval_mode:
                optimizer.step()
            output = integrate_output_list(output_list)
            loss = output["image"]["image"]["loss"].mean()
            pbar.set_description(f"loss: {loss.item():.4f}")

            pbar.update(1)


if __name__ == "__main__":
    fire.Fire(main)
