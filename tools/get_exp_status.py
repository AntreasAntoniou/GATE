import json
import os
import sys
from typing import Dict, List, Union

import fire
import pandas as pd
import wandb
from rich import print
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm


def check_wandb_experiments(
    experiments: Union[Dict[str, str], List[str]],
    project: str | List[str] = None,
    token: str = None,
    print_table: bool = False,
    filter_for_non_completed: bool = False,
    filter_for_non_running: bool = False,
):
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
    exp_name_to_state_dict = {}
    for run in tqdm(runs):
        if "exp_name" in run.config:
            exp_name = run.config["exp_name"].lower()

            if exp_name not in exp_name_to_summary_dict:
                exp_name_to_summary_dict[exp_name] = [run.summaryMetrics]
                exp_name_to_state_dict[exp_name] = [run.state]
            else:
                exp_name_to_summary_dict[exp_name].append(run.summaryMetrics)
                exp_name_to_state_dict[exp_name].append(run.state)

    # Initialize an empty list to store experiment data
    exp_data = {}
    exp_to_command = {key.lower(): value for key, value in experiments.items()}
    if isinstance(experiments, dict):
        experiments = list(experiments.keys())
    experiments = [exp.lower() for exp in experiments]
    # Iterate through the given experiments
    with tqdm(total=len(experiments), desc="Checking experiments") as pbar:
        for exp_name in experiments:
            # Check if the experiment exists in wandb and if it has completed the testing stage
            if exp_name in exp_name_to_summary_dict:
                keys = [
                    k
                    for summary_keys in exp_name_to_summary_dict[exp_name]
                    for k in summary_keys.keys()
                ]

                testing_completed = any("testing/ensemble" in k for k in keys)
                if "fs" in exp_name:
                    print(exp_name, list(sorted(keys)))

                currently_running = any(
                    "running" == state.lower()
                    for state in exp_name_to_state_dict[exp_name]
                )
                if "global_step" in keys:
                    current_iter = max(
                        [
                            summary_stats["global_step"]
                            for summary_stats in exp_name_to_summary_dict[
                                exp_name
                            ]
                            if "global_step" in summary_stats.keys()
                        ]
                    )
                else:
                    current_iter = 0

            else:
                testing_completed = False
                currently_running = False
                current_iter = 0

            # Append the data to the list
            exp_data[exp_name] = {
                "testing_completed": testing_completed,
                "currently_running": currently_running,
                "current_iter": current_iter,
                "command": exp_to_command[exp_name],
            }
            pbar.update(1)

    if filter_for_non_completed:
        exp_data = {
            key: value
            for key, value in exp_data.items()
            if not value["testing_completed"]
        }

    if filter_for_non_running:
        exp_data = {
            key: value
            for key, value in exp_data.items()
            if not value["currently_running"]
        }

    if print_table:
        # Create a pandas DataFrame
        df = pd.DataFrame(
            exp_data
        ).T  # Transpose the DataFrame so that each experiment is a row

        # Create a console for rich print
        console = Console()

        # Create a table
        table = Table(
            show_header=True, header_style="bold magenta", style="dim"
        )
        table.add_column("idx", justify="right")
        table.add_column("Experiment Name", width=50)
        table.add_column("Currently Running", justify="right")
        table.add_column("Testing Completed", justify="right")
        table.add_column("Current Iteration", justify="right")

        # Add rows to the table
        for idx, (exp_name, row) in enumerate(df.iterrows()):
            table.add_row(
                str(idx),
                exp_name,
                str(row["currently_running"]),
                str(row["testing_completed"]),
                str(row["current_iter"]),
            )

        # Print the table
        console.print(table)

    return exp_data


def main(
    print_table: bool = True,
    filter_for_non_completed: bool = False,
    filter_for_non_running: bool = False,
    pipe_output: bool = False,
):
    if print_table and pipe_output:
        raise ValueError(
            f"print_table={print_table} and pipe_output={pipe_output}, Cannot print table and pipe output at the same time. Please choose one."
        )
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
        project=["gate-0-9-1"],
        print_table=print_table,
        filter_for_non_completed=filter_for_non_completed,
        filter_for_non_running=filter_for_non_running,
    )

    if pipe_output:
        exp_dict = {
            exp_name: exp_data["command"]
            for exp_name, exp_data in exp_dict.items()
        }
        return json.dumps(exp_dict)


# This exposes the function to the command line
if __name__ == "__main__":
    fire.Fire(main)
