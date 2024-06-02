import pathlib
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from attr import dataclass
from rich import print
from sklearn.model_selection import ShuffleSplit
from tqdm.auto import tqdm

import wandb

task_to_dataset_map = yaml.safe_load(open("notebooks/task_mapping.yaml"))

WANDB_PROJECT_NAME = "gate-evolve-nn-v19"


@dataclass
class EvaluationResult:
    """
    Class to represent the evaluation result.
    """

    avg_score: float
    min_score: float
    max_score: float
    std_score: float
    combination: Tuple[str, ...]


def compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, metrics: List[str]
) -> torch.Tensor:
    """
    Compute the loss function.

    Args:
        logits (torch.Tensor): The predicted logits.
        labels (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: The computed loss.
    """
    metric_dict = defaultdict(list)
    reverse_value_key_dict = {
        item.split(".")[0]: k
        for k, v in task_to_dataset_map.items()
        for item in v
    }

    # Compute the MSE loss for each pair of logit and label
    mse_losses = F.mse_loss(logits, labels, reduction="none")
    mse_losses = mse_losses.permute(1, 0)
    metric_to_losses_dict = {
        key: value for key, value in zip(metrics, mse_losses)
    }
    # Initialize a dictionary to store metrics and their corresponding losses
    metric_dict = defaultdict(list)
    dataset_dict = defaultdict(list)
    dataset_key_dict = defaultdict(list)
    # Get mean losses across a particular datasets metrics
    for metric, mse_loss in metric_to_losses_dict.items():
        dataset_name = metric.split(".")[0]
        dataset_dict[dataset_name].append(mse_loss)
        dataset_key_dict[dataset_name].append(metric)

    dataset_dict = {
        key: torch.stack(value).mean() for key, value in dataset_dict.items()
    }
    # print(dataset_key_dict)
    metric_key_dict = defaultdict(list)
    # Append losses to corresponding metrics
    for dataset_name, mean_loss in dataset_dict.items():
        metric_dict[reverse_value_key_dict[dataset_name]].append(mean_loss)
        metric_key_dict[reverse_value_key_dict[dataset_name]].append(
            dataset_key_dict[dataset_name]
        )
    # print(metric_key_dict)
    # Compute mean per task
    loss_dict = {
        key: torch.stack(value).mean() for key, value in metric_dict.items()
    }

    # Compute overall loss as the mean of all task losses
    loss = torch.stack(list(loss_dict.values())).mean()

    return loss


def mutate_combination(
    combination: Tuple[str, ...], all_metrics: List[str]
) -> Tuple[str, ...]:
    """
    Mutate a combination by removing one metric and adding another.

    Args:
        combination (Tuple[str, ...]): The original combination.
        all_metrics (List[str]): The list of all available metrics.

    Returns:
        Tuple[str, ...]: The mutated combination.
    """
    metric_to_remove = random.choice(combination)
    metric_to_add = random.choice(all_metrics)
    while metric_to_add in combination:
        metric_to_add = random.choice(all_metrics)
    new_combination = list(combination)
    new_combination.remove(metric_to_remove)
    new_combination.append(metric_to_add)
    new_combination = sorted(new_combination)
    new_combination = set(new_combination)
    new_combination = sorted(new_combination)
    return tuple(new_combination)


def reset_parameters(model: nn.Module) -> nn.Module:
    """
    Reset the parameters of a model.

    Args:
        model (nn.Module): The model to reset.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            reset_parameters(layer)
    return model


class SmallTransformer(nn.Module):
    def __init__(
        self,
        output_dim,
        seq_length,
        num_heads=2,
        dim_feedforward=128,
        dropout=0.0,
    ):
        super(SmallTransformer, self).__init__()
        self.embedding = nn.Linear(1, dim_feedforward)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, seq_length, dim_feedforward)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )
        self.layer_norm = nn.LayerNorm(dim_feedforward)
        self.fc_out = nn.Linear(dim_feedforward, output_dim)

    def forward(self, x):
        # Assuming x has shape (batch_size, seq_length, input_dim)
        x = x.unsqueeze(-1)
        x = self.embedding(
            x
        )  # Transform to (batch_size, seq_length, dim_feedforward)
        x += self.positional_embedding  # Add positional embedding
        x = self.layer_norm(x)
        x = x.permute(
            1, 0, 2
        )  # Transformer expects (seq_length, batch_size, dim_feedforward)
        x = self.transformer_encoder(x)
        x = x.permute(
            1, 0, 2
        )  # Back to (batch_size, seq_length, dim_feedforward)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling over sequence length
        x = self.fc_out(x)  # Final output
        return x


class SmallMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout=0.0):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Assuming x has shape (batch_size, input_dim)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.layer_norm1(x)

        # x = F.gelu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.layer_norm2(x)

        x = self.fc3(x)

        return x


torch.set_float32_matmul_precision("high")


def evaluate_combination(
    df: pd.DataFrame,
    input_datasets: Tuple[str, ...],
    epochs: int = 20,
    device: Optional[torch.device] = None,
    target_datasets: Optional[List[str]] = None,
    metrics_to_leave_out: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Evaluate a combination of metrics on a given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        metric_combination (Tuple[str, ...]): The combination of metrics to evaluate.
        device (torch.device): The device to use for computation.
        epochs (int, optional): The number of training epochs. Defaults to 20.

    Returns:
        EvaluationResult: The evaluation result containing average, minimum,
        maximum, and standard deviation scores.
    """
    if device is None:
        device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

    if device is None:
        device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

    input_metrics = [
        item
        for item in df.columns
        if any(metric in item for metric in input_datasets)
        if any(metric in item for metric in input_datasets)
    ]
    if target_datasets is None:
        target_metrics = list(set(df.columns))
    else:
        target_metrics = [
            item
            for item in df.columns
            if any(metric in item for metric in target_datasets)
            if any(metric in item for metric in target_datasets)
        ]
    if metrics_to_leave_out is not None:
        target_metrics = [
            item
            for item in target_metrics
            if all([item != metric for metric in metrics_to_leave_out])
        ]
    x = df[input_metrics].values  # num_models, k
    y = df[target_metrics].values  # num_models, all_metrics

    x = torch.tensor(x).to(device).float()
    y = torch.tensor(y).to(device).float()

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    kf = ShuffleSplit(n_splits=20, random_state=42, test_size=0.3)
    scores = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Split the training set into training and validation sets
        kf_val = ShuffleSplit(n_splits=1, random_state=42, test_size=0.3)
        train_index, val_index = next(kf_val.split(x_train))
        x_train, x_val = x_train[train_index], x_train[val_index]
        y_train, y_val = y_train[train_index], y_train[val_index]

        model = SmallMLP(
            output_dim=y_train.shape[1], input_dim=x_train.shape[1]
        ).to(device)
        model = reset_parameters(model)
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        # try lasso regression
        models = dict()
        # with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = compute_loss(outputs, y_train, metrics=target_metrics)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = model(x_val)
                score = compute_loss(y_pred, y_val, metrics=target_metrics)
                models[score] = model.state_dict()
                # pbar.set_description(
                #     f"Epoch: {epoch}, Loss: {loss}, Score: {score}"
                # )
                # pbar.update(1)

        model.load_state_dict(models[min(models.keys())])
        with torch.no_grad():
            score = compute_loss(model(x_test), y_test, metrics=target_metrics)
        scores.append(score.cpu().detach().numpy())
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    std_score = np.std(scores)
    return EvaluationResult(
        avg_score=avg_score,
        min_score=min_score,
        max_score=max_score,
        std_score=std_score,
        combination=input_datasets,
    )


def get_combinations(
    metrics: List[str], max_length: int = 12
) -> List[Tuple[str, ...]]:
    """
    Generate combinations of metrics.

    Args:
        metrics (List[str]): The list of metrics.
        max_length (int, optional): The maximum length of combinations. Defaults to 12.

    Returns:
        List[Tuple[str, ...]]: The generated combinations.
    """
    from itertools import combinations

    out = [i for i in combinations(metrics, max_length)]
    return out


METRICS = sorted(
    set(
        [
            "acdc",
            "ade20k",
            "aircraft",
            "chexpert",
            "cifar100",
            "clevr",
            "clevr-math",
            "coco-10k",
            "coco-164k",
            "cubirds",
            "diabetic",
            "dtextures",
            "flickr30k",
            "food101",
            "fungi",
            "happy",
            "ham10k",
            "hmdb51",
            "imagenet1k",
            "iwildcam",
            "kinetics",
            "mini",
            "newyorkercaptioncontest",
            "nyu",
            "omniglot",
            "pascal",
            "places365",
            "ucf",
            "vgg",
            "winoground",
        ]
    )
)


def load_data_as_df(filepath):
    df = pd.read_csv(filepath)

    # Removing the first row from the DataFrame
    df = df.drop(df.index[0])

    # Resetting the DataFrame index
    df.reset_index(drop=True, inplace=True)

    columns = df.columns.tolist()[1:]
    model_names = df.columns.tolist()[0]
    # remove all datapoints with less than 50% on imagenet1k.1

    # model_names = df[model_names][1:]
    df = df.fillna(5)
    # Convert all columns (except the first) to numeric
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop first column
    df = df.drop(columns=[model_names])

    return df


def main(
    result_csv_path: str | pathlib.Path,
    combination_size: int = 12,
    device: str = "cuda",
):
    """
    Main function for combination optimization.

    Args:
        result_csv_path (str or pathlib.Path): Path to the CSV file containing the results.
        combination_size (int, optional): Size of the combinations to evaluate. Defaults to 12.
    """
    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        name=f"{WANDB_PROJECT_NAME}-k={combination_size}",
        entity="machinelearningbrewery",
    )
    df = load_data_as_df(result_csv_path)
    # replace NaNs with 1000
    device = torch.device(device)
    epochs = 20
    num_winners = 50
    num_children = 10
    num_generations = 10
    num_init_samples = 1000

    previous_k_top_combination_file = (
        f"top_combinations_k={combination_size-1}.csv"
    )

    previous_k_top_combination_file = pathlib.Path(
        previous_k_top_combination_file
    )
    out = []
    if previous_k_top_combination_file.exists():
        # Load the previous top combinations from combination column
        top_combinations_df = pd.read_csv(previous_k_top_combination_file)
        # Get the top combinations
        top_combinations = [
            tuple(combination.split(", "))
            for combination in top_combinations_df["combination"]
        ]

        # add one metric NOT already in a combination to each combination to make it k+1
        top_combinations = [
            set(combination + (metric,))
            for metric in METRICS
            for combination in top_combinations
        ]
        # keep only combinations of length k+1
        top_combinations = [
            combination
            for combination in top_combinations
            if len(combination) == combination_size
        ]
        out = top_combinations
        print("Loaded previous top combinations", out)
    unique_combinations = get_combinations(
        metrics=METRICS, max_length=combination_size
    )

    out = [tuple(sorted(item)) for item in out]
    out = list(set(sorted(out)))
    num_of_random_samples = min(
        num_init_samples - len(out), len(unique_combinations) - len(out)
    )
    if len(out) < num_init_samples:
        random_combinations = random.sample(
            unique_combinations, num_of_random_samples
        )

        out += random_combinations
        out = [tuple(sorted(item)) for item in out]
        out = list(set(sorted(out)))
        num_of_random_samples = min(
            num_init_samples - len(out), len(unique_combinations) - len(out)
        )

    print(
        f"num of random samples: {num_of_random_samples}, "
        f"num of init samples: {num_init_samples}, "
        f"len of out: {len(out)}, "
        f"num unique combinations: {len(unique_combinations)}"
    )

    score_combinations = []
    for combination in tqdm(out, desc="Evaluating combinations"):
        score = evaluate_combination(
            df=df,
            input_datasets=combination,
            device=device,
            epochs=epochs,
        )
        score_combinations.append(score)
        print(f"Combination: {combination}, Score: {score}")

    top_combinations = sorted(score_combinations)[:num_winners]
    run.log(
        {
            "generation": 0,
            "mean_score": np.mean(
                [item.avg_score for item in top_combinations]
            ),
            "std_score": np.std([item.avg_score for item in top_combinations]),
            "min_score": np.min([item.avg_score for item in top_combinations]),
            "max_score": np.max([item.avg_score for item in top_combinations]),
        }
    )

    top_combinations_df = pd.DataFrame(
        [
            {
                "avg_score": item.avg_score,
                "min_score": item.min_score,
                "max_score": item.max_score,
                "std_score": item.std_score,
                "combination": ", ".join(item.combination),
            }
            for item in top_combinations
        ]
    )
    top_combinations_df.to_csv("top_combinations.csv")
    previous_k_top_combination_file = (
        f"top_combinations_k={combination_size}.csv"
    )

    previous_k_top_combination_file = pathlib.Path(
        previous_k_top_combination_file
    )
    top_combinations_df.to_csv(previous_k_top_combination_file)
    artifact = wandb.Artifact("top_combinations", type="top_combinations")
    artifact.add_file("top_combinations.csv")
    run.log_artifact(artifact)

    for i, item in enumerate(top_combinations):
        print(f"Rank: {i}, Score: {item.avg_score}, Combination: {item}")

    if len(score_combinations) >= num_init_samples:
        for i in range(1, num_generations):
            new_combinations = set()
            for item in top_combinations:
                combination = item.combination
                # make list hashable
                combination = tuple(combination)
                new_combinations.add(combination)
                for _ in range(num_children):
                    mutated_combination = mutate_combination(
                        combination, METRICS
                    )
                    new_combinations.add(tuple(mutated_combination))
            score_combinations: List[EvaluationResult] = []
            new_combinations = list(set(new_combinations))
            for combination in tqdm(
                new_combinations, desc="Evaluating mutated combinations"
            ):
                score_combinations.append(
                    evaluate_combination(
                        df=df,
                        input_datasets=combination,
                        device=device,
                        epochs=epochs,
                    )
                )
            # sort by score given that the items are of type EvaluationResult
            top_combinations = sorted(
                score_combinations, key=lambda x: x.avg_score
            )[:num_winners]

            for j, item in enumerate(top_combinations):
                print(f"Generation: {i}, Rank: {j}, Item: {item}")
            print(
                f"logging the following metrics: mean_score: {np.mean([item.avg_score for item in top_combinations])}, std_score: {np.std([item.avg_score for item in top_combinations])}, min_score: {np.min([item.avg_score for item in top_combinations])}, max_score: {np.max([item.avg_score for item in top_combinations])}"
            )
            run.log(
                {
                    "generation": i,
                    "mean_score": np.mean(
                        [item.avg_score for item in top_combinations]
                    ),
                    "std_score": np.std(
                        [item.avg_score for item in top_combinations]
                    ),
                    "min_score": np.min(
                        [item.avg_score for item in top_combinations]
                    ),
                    "max_score": np.max(
                        [item.avg_score for item in top_combinations]
                    ),
                }
            )

            print(
                f"logging the following metrics: mean_score: {np.mean([item.avg_score for item in top_combinations])}, std_score: {np.std([item.avg_score for item in top_combinations])}, min_score: {np.min([item.avg_score for item in top_combinations])}, max_score: {np.max([item.avg_score for item in top_combinations])}"
            )
            # cast to DataFrame given that top_combinations is a list of EvaluationResult and the keys are the attributes of EvaluationResult
            top_combinations_df = pd.DataFrame(
                [
                    {
                        "avg_score": item.avg_score,
                        "min_score": item.min_score,
                        "max_score": item.max_score,
                        "std_score": item.std_score,
                        "combination": ", ".join(item.combination),
                    }
                    for item in top_combinations
                ]
            )
            top_combinations_df.to_csv("top_combinations.csv")
            artifact = wandb.Artifact(
                "top_combinations", type="top_combinations"
            )
            artifact.add_file("top_combinations.csv")
            previous_k_top_combination_file = (
                f"top_combinations_k={combination_size}.csv"
            )

            previous_k_top_combination_file = pathlib.Path(
                previous_k_top_combination_file
            )
            top_combinations_df.to_csv(previous_k_top_combination_file)
            run.log_artifact(artifact)
    if len(top_combinations[0].combination) > 1:
        remove_one_from_combo_and_reevaluate(
            combinations=top_combinations[0],
            df=df,
            device=device,
            epochs=epochs,
            run=run,
        )
    run.finish()


def remove_one_from_combo_and_reevaluate(
    combinations: EvaluationResult, df, device, epochs, run
):
    combintions_of_one_less = [combinations.combination]

    for feature in combinations.combination:
        new_combination = list(combinations.combination)
        new_combination.remove(feature)
        combintions_of_one_less.append(new_combination)

    results_dict = {}
    for combination in tqdm(
        combintions_of_one_less, desc="Evaluating mutated combinations"
    ):

        missing_feature = list(
            set(combinations.combination) - set(combination)
        )

        missing_feature = (
            missing_feature[0] if len(missing_feature) > 0 else "full"
        )

        scores = evaluate_combination(
            df=df,
            input_datasets=combination,
            device=device,
            epochs=epochs,
            metrics_to_leave_out=[missing_feature],
        )
        exp_name = f"gate-evolve-nn-k={len(combinations.combination)}-loo-{missing_feature}"
        exp_name = f"gate-evolve-nn-k={len(combinations.combination)}-loo-{missing_feature}"

        results_dict[exp_name] = {
            "combination": combination,
            "missing_feature": missing_feature,
            "mean": scores.avg_score,
            "std": scores.std_score,
            "min": scores.min_score,
            "max": scores.max_score,
        }

    run.log(results_dict)


def run_job(combination_size, gpu_id):
    csv_path = "notebooks/08052024.csv"
    device = f"cuda:{gpu_id}"
    main(
        result_csv_path=csv_path,
        combination_size=combination_size,
        device=device,
    )


if __name__ == "__main__":

    combination_sizes = [16, 17, 22]  # 1 to 31
    gpu_ids = list(range(3))  # Adjust this based on available GPUs
    max_workers = 3  # Maximum number of parallel jobs
    jobs_per_worker = int(len(combination_sizes) / max_workers)
    per_worker_combinations_sizes = defaultdict(list)

    for worker_id in range(max_workers):
        start_idx = worker_id * jobs_per_worker
        end_idx = (
            (worker_id + 1) * jobs_per_worker
            if worker_id != max_workers - 1
            else len(combination_sizes)
        )
        jobs = combination_sizes[start_idx:end_idx]
        print(f"Worker {worker_id} will run jobs {jobs}.")
        per_worker_combinations_sizes[worker_id] = jobs

    jobs = []
    for i in range(len(combination_sizes) // max_workers):
        for key, value in per_worker_combinations_sizes.items():
            jobs.append(
                (
                    combination_sizes.pop(0),
                    gpu_ids[np.random.randint(0, len(gpu_ids))],
                )
            )

    if max_workers == 1:
        for combination_size, gpu_id in jobs:
            run_job(combination_size, gpu_id)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(run_job, combination_size, gpu_id): (
                    combination_size,
                    gpu_id,
                )
                for combination_size, gpu_id in jobs
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    future.result()  # You can also capture the return value from `main` if needed
                    print(f"Job {job} completed successfully.")
                except Exception as exc:
                    print(f"Job {job} generated an exception: {exc}.")
