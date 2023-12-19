import os

import fire
import pandas as pd

import wandb


def fetch_wandb_data(project: str = None, token: str = None):
    """
    Fetch Weights & Biases data for a given project using the provided token and display it in a pandas DataFrame.
    Args:
        project: The name of the Weights & Biases project.
        token: The Weights & Biases token.
    Returns:
        None. Prints a pandas DataFrame.
    """
    # If project or token are not provided, use environment variables
    if project is None:
        project = os.getenv("WANDB_PROJECT")
    if token is None:
        token = os.getenv("WANDB_TOKEN")

    # Set your wandb API key
    wandb.login(key=token)

    # Initialize an empty list to store experiment data
    exp_data = []

    # Fetch runs from wandb
    api = wandb.Api()
    runs = api.runs(f"{wandb.settings().entity}/{project}")

    # Iterate through the runs and fetch required data
    for run in runs:
        # Fetch summary metrics
        summary_metrics = run.summary_metrics

        # Filter out the relevant metrics and group them
        metrics = {}
        for k, v in summary_metrics.items():
            if "testing/ensemble" in k:
                key = k.replace("-", "").rsplit("epoch", 1)[0] + "epoch"
                if key not in metrics:
                    metrics[key] = {"mean": None, "std": None}
                if "mean" in k:
                    metrics[key]["mean"] = v
                elif "std" in k:
                    metrics[key]["std"] = v

        # Format the metrics as "mean ± std"
        for k, v in metrics.items():
            metrics[k] = (
                f'{v["mean"]} ± {v["std"]}'
                if v["mean"] is not None and v["std"] is not None
                else None
            )

        # Add the experiment name to the metrics
        metrics["exp_name"] = run.name

        # Append to the list
        exp_data.append(metrics)

    # Create a pandas DataFrame
    df = pd.DataFrame(exp_data)

    # Display the DataFrame
    print(df)


# This exposes the function to the command line
fire.Fire(fetch_wandb_data)
