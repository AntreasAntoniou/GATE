# Complete Python code with the requested adjustments to make 'accelerate_launch_path' and 'gate_run_path' accessible at the 'get_commands' level.

from typing import Dict, List, Union


def build_command(
    exp_name: str,
    encoder_name: str,
    adapter_name: str,
    dataset_name: str,
    num_workers: int = 12,
    gpu_ids: Union[str, None] = None,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    trainer="image_classification",
    evaluator="image_classification",
    encoder_args: str = "",
    adapter_args: str = "",
    lr: float = 1e-5,
    seed: int = 42,
    accelerate_launch_path: str = "/opt/conda/envs/main/bin/accelerate-launch",
    gate_run_path: str = "/app/gate/run.py",
    train_iters: int = 10000,
    evaluate_every_n_steps: int = 250,
) -> str:
    """
    Build a command for running an experiment. 🛠️
    """

    accelerate_launch_command = (
        f"{accelerate_launch_path} --mixed_precision=bf16"
    )
    if gpu_ids:
        accelerate_launch_command += f" --gpu_ids={gpu_ids}"

    gate_run_command = f"{gate_run_path}"

    command_template = (
        f"{accelerate_launch_command} {gate_run_command} "
        f"exp_name={exp_name} encoder={encoder_name} {encoder_args} adapter={adapter_name} {adapter_args} dataset={dataset_name} optimizer.lr={lr} optimizer.weight_decay=0.01 "
        f"trainer={trainer} evaluator={evaluator} num_workers={num_workers} "
        f"seed={seed} train_batch_size={train_batch_size} eval_batch_size={eval_batch_size} "
        f"train_iters={train_iters} learner.evaluate_every_n_steps={evaluate_every_n_steps}"
    )
    return command_template
