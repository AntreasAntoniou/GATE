import json
import os
import sys
from typing import Dict, List, Optional, Union

import fire
import pandas as pd
from rich import print
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

import wandb

# 'seed': 7,
# 'test': True,
# 'train': True,
# 'resume': True,
# 'adapter': {'encoder': '???', '_target_': 'gate.models.adapters.classification.BackboneWithLinearClassification', 'num_classes': 5, 'allow_on_model_metric_computation': True},
# 'dataset': {
#     '_target_': 'gate.data.medical.classification.chexpert.build_gate_dataset',
#     'data_dir': '/disk/scratch_fast1/data/',
#     'transforms': None,
#     'num_classes': 5,
#     'label_idx_to_class_name': {'0': 'cardiomegaly', '1': 'edema', '2': 'consolidation', '3': 'atelectasis', '4': 'pleural-effusion'}
# },
# 'encoder': {
#     '_target_': 'gate.models.backbones.timm.TimmCLIPAdapter',
#     'image_size': 224,
#     'pretrained': True,
#     'clip_model_name': 'openai/clip-vit-base-patch16',
#     'timm_model_name': 'vit_base_patch16_siglip_224',
#     'num_projection_features': None
# },


def format_metrics(
    metrics: Dict[str, Dict[str, Union[float, None]]]
) -> Dict[str, str]:
    """
    Format the metrics as "mean ± std".
    """
    for k, v in metrics.items():
        metrics[k] = (
            f'{v["mean"]} ± {v["std"]}'
            if v["mean"] is not None and v["std"] is not None
            else None
        )
    return metrics


def extract_test_results(
    experiments: Union[Dict[str, str], List[str]],
    wandb_exp_name_to_summary_dict: Dict[str, List[Dict[str, float]]],
    wandb_exp_name_to_config_dict: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Dict[str, Union[str, None]]]:
    """
    Extract test results from the experiments.
    """
    exp_data = {}
    with tqdm(total=len(experiments), desc="Checking experiments") as pbar:
        for exp_name in experiments:
            # Check if the experiment exists in wandb and fetch its testing result
            if exp_name in wandb_exp_name_to_summary_dict:
                summary_metrics = wandb_exp_name_to_summary_dict[exp_name]
                configs = wandb_exp_name_to_config_dict[exp_name]

                # Filter out the relevant metrics and group them
                metrics = {}
                for summary_keys in summary_metrics:
                    for k, v in summary_keys.items():
                        if k.startswith("testing/ensemble"):
                            key = k.replace("-/", "/")
                            if key not in metrics:
                                metrics[key] = {"mean": None, "std": None}
                            if "mean" in k:
                                metrics[key]["mean"] = v
                            elif "std" in k:
                                metrics[key]["std"] = v

                # Format the metrics as "mean ± std"
                metrics = format_metrics(metrics)

                # Append the data to the list
                exp_data[exp_name] = metrics
                # Add experiment info from run.config
                exp_data[exp_name].update(
                    {
                        "dataset": configs[0]["dataset"]["_target_"].split(
                            "."
                        )[-2],
                        "model": exp_name.replace("beta", ""),
                        "seed": configs[0]["seed"],
                    }
                )
            else:
                exp_data[exp_name] = {"testing_result": "TBA"}

            pbar.update(1)
    return exp_data


def print_experiment_table(
    exp_data: Dict[str, Dict[str, Union[str, None]]]
) -> None:
    """
    Print the experiment data in a table format.
    Args:
        exp_data: A dictionary containing the experiment data.
    """
    # Create a pandas DataFrame
    df = pd.DataFrame.from_dict(exp_data, orient="index")

    # Fill missing values with 'NA'
    df.fillna("NA", inplace=True)

    # Create a console for rich print
    console = Console()

    # Create a table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Experiment Name", width=50)

    # Add columns for each metric
    for metric in df.columns:
        table.add_column(metric, justify="right")

    # Add rows to the table
    for exp_name, row in df.iterrows():
        values = [str(row[metric]) for metric in df.columns]
        table.add_row(exp_name, *values)

    # Print the table
    console.print(table)


def check_wandb_experiments(
    experiments: Optional[Union[Dict[str, str], List[str]]] = None,
    project: Optional[str | List[str]] = None,
    token: Optional[str] = None,
    print_table: bool = True,
) -> Dict[str, Dict[str, Union[str, None]]]:
    """
    Check if the given experiments exist in a Weights & Biases project and if they have completed the testing stage.
    Args:
        experiments: A list of experiment names or a dict with experiment names as keys and experiment commands as values.
        project: The name of the Weights & Biases project.
        token: The Weights & Biases token.
    Returns:
        None. Prints a pandas DataFrame.
    """
    # If project or token are not provided, use environment variables
    if project is None:
        project = os.getenv("WANDB_PROJECT")
    if token is None:
        token = os.getenv("WANDB_API_KEY")

    # Set your wandb API key
    wandb.login(key=token)

    # Fetch runs from wandb
    api = wandb.Api()
    runs = (
        api.runs(path=project)
        if isinstance(project, str)
        else [api.runs(path=p) for p in project]
    )
    runs = [run for run_list in runs for run in run_list]

    exp_name_to_summary_dict = {}
    exp_name_to_config_dict = {}

    for run in tqdm(runs):
        if "exp_name" in run.config:
            exp_name = run.config["exp_name"].lower()
            if exp_name not in exp_name_to_summary_dict:
                exp_name_to_summary_dict[exp_name] = [run.summaryMetrics]
                exp_name_to_config_dict[exp_name] = [run.config]
            else:
                exp_name_to_summary_dict[exp_name].append(run.summaryMetrics)
                exp_name_to_config_dict[exp_name].append(run.config)

    # Initialize an empty list to store experiment data
    exp_data = {}
    if isinstance(experiments, dict):
        experiments = list(experiments.keys())
    experiments = [exp.lower() for exp in experiments]

    # Iterate through the given experiments
    exp_data = extract_test_results(
        experiments=experiments,
        wandb_exp_name_to_summary_dict=exp_name_to_summary_dict,
        wandb_exp_name_to_config_dict=exp_name_to_config_dict,
    )
    if print_table:
        print_experiment_table(exp_data)

    return exp_data


def main(print_table: bool = False):
    # Read input from stdin
    raw_experiments = sys.stdin.read()

    # Parse the input as JSON
    try:
        experiments = json.loads(raw_experiments)
    except json.JSONDecodeError:
        print(f"Invalid input: {raw_experiments}")
        return

    # Call the function
    exp_dict = check_wandb_experiments(
        experiments=experiments,
        project=["eidf-monitor", "gate-0-9-1"],
        print_table=print_table,
    )

    json.dumps(exp_dict)


# This exposes the function to the command line
if __name__ == "__main__":
    fire.Fire(main)
