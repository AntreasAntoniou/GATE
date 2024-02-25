import pathlib
import random
from math import comb
from typing import List, Tuple

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from attr import dataclass
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm.auto import tqdm

import wandb


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


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss function.

    Args:
        logits (torch.Tensor): The predicted logits.
        labels (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: The computed loss.
    """
    return F.mse_loss(logits, labels) + F.l1_loss(logits, labels)


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


def evaluate_combination(
    df: pd.DataFrame,
    combination: Tuple[str, ...],
    device: torch.device,
    epochs: int = 20,
) -> EvaluationResult:
    """
    Evaluate a combination of metrics on a given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        combination (Tuple[str, ...]): The combination of metrics to evaluate.
        device (torch.device): The device to use for computation.
        epochs (int, optional): The number of training epochs. Defaults to 20.

    Returns:
        EvaluationResult: The evaluation result containing average, minimum,
        maximum, and standard deviation scores.
    """
    input_metrics = [
        item
        for item in df.columns
        if any(metric in item for metric in combination)
    ]
    target_metrics = list(set(df.columns) - set(input_metrics))
    x = df[input_metrics].values
    y = df[target_metrics].values
    x = torch.tensor(x).to(device).float()
    y = torch.tensor(y).to(device).float()
    kf = ShuffleSplit(n_splits=10, random_state=42, test_size=0.4)
    scores = []
    for train_index, test_index in kf.split(x):
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = nn.Linear(x_train.shape[1], y_train.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
        model.reset_parameters()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = compute_loss(outputs, y_train)
            loss.backward()
            optimizer.step()
            y_pred = model(x_test)
            score = compute_loss(y_pred, y_test)
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
        combination=combination,
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
            "ade20k",
            "aircraft",
            "chexpert",
            "cifar100",
            "cifar100",
            "cifar100",
            "clevr",
            "clevr-math",
            "clevr-math",
            "coco-10k",
            "coco-10k",
            "coco-164k",
            "coco-164k",
            "cubirds",
            "diabetic",
            "dtextures",
            "flickr30k",
            "flickr30k",
            "flickr30k",
            "flickr30k",
            "food101",
            "food101",
            "food101",
            "fungi",
            "ham10k",
            "hmdb51",
            "hmdb51",
            "imagenet1k",
            "imagenet1k",
            "imagenet1k",
            "iwildcam",
            "kinetics",
            "kinetics",
            "mini",
            "newyorkercaptioncontest",
            "newyorkercaptioncontest",
            "newyorkercaptioncontest",
            "newyorkercaptioncontest",
            "nyu",
            "nyu",
            "omniglot",
            "pascal",
            "pascal",
            "places365",
            "places365",
            "places365",
            "pokemonblipcaptions",
            "pokemonblipcaptions",
            "pokemonblipcaptions",
            "pokemonblipcaptions",
            "ucf",
            "ucf",
            "vgg",
            "winoground",
            "winoground",
        ]
    )
)


def main(result_csv_path: str | pathlib.Path, combination_size: int = 12):
    """
    Main function for combination optimization.

    Args:
        result_csv_path (str or pathlib.Path): Path to the CSV file containing the results.
        combination_size (int, optional): Size of the combinations to evaluate. Defaults to 12.
    """
    run = wandb.init(
        project=f"combination-optimization",
        name=f"combination-optimization-{combination_size}",
        entity="machinelearningbrewery",
    )
    df = pd.read_csv(result_csv_path)
    df = df.iloc[:, 1:]
    df = df.iloc[1:, :]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    population_size = 25
    mutation_rounds = 10
    num_generations = 10
    out = get_combinations(metrics=METRICS, max_length=combination_size)
    random_combinations = random.sample(out, min(10, len(out)))
    score_combinations = []
    for combination in tqdm(
        random_combinations, desc="Evaluating combinations"
    ):
        score_combinations.append(
            evaluate_combination(
                df=df, combination=combination, device=device, epochs=epochs
            )
        )
    top_combinations = sorted(score_combinations)[:population_size]
    for i, item in enumerate(top_combinations):
        print(f"Rank: {i}, Score: {item.avg_score}, Combination: {item}")
    for i in range(num_generations):
        new_combinations = set()
        for item in top_combinations:
            combination = item.combination
            new_combinations.add(combination)
            for _ in range(mutation_rounds):
                mutated_combination = mutate_combination(combination, METRICS)
                new_combinations.add(mutated_combination)
        score_combinations: List[EvaluationResult] = []
        new_combinations = list(set(new_combinations))
        for combination in tqdm(
            new_combinations, desc="Evaluating mutated combinations"
        ):
            score_combinations.append(
                evaluate_combination(
                    df=df,
                    combination=combination,
                    device=device,
                    epochs=epochs,
                )
            )
        # sort by score given that the items are of type EvaluationResult
        top_combinations = sorted(
            score_combinations, key=lambda x: x.avg_score
        )[:population_size]

        for j, item in enumerate(top_combinations):
            print(f"Generation: {i}, Rank: {j}, Item: {item}")

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
                    "combination": item.combination,
                }
                for item in top_combinations
            ]
        )
        top_combinations_df.to_csv("top_combinations.csv")
        artifact = wandb.Artifact("top_combinations", type="top_combinations")
        artifact.add_file("top_combinations.csv")
        run.log_artifact(artifact)
    remove_one_from_combo_and_reevaluate(
        combinations=top_combinations[0],
        df=df,
        device=device,
        epochs=epochs,
    )
    run.finish()


def remove_one_from_combo_and_reevaluate(
    combinations: EvaluationResult, df, device, epochs
):
    combintions_of_one_less = [combinations.combination]

    for feature in combinations.combination:
        new_combination = list(combinations.combination)
        new_combination.remove(feature)
        combintions_of_one_less.append(new_combination)

    for combination in tqdm(
        combintions_of_one_less, desc="Evaluating mutated combinations"
    ):
        scores = evaluate_combination(
            df=df,
            combination=combination,
            device=device,
            epochs=epochs,
        )
        missing_feature = list(
            set(combinations.combination) - set(combination)
        )

        missing_feature = (
            missing_feature[0] if len(missing_feature) > 0 else "full"
        )

        run = wandb.init(
            project=f"combination-optimization",
            name=f"combination-optimization-loo-{len(combinations.combination)}-{missing_feature}",
            entity="machinelearningbrewery",
            reinit=True,
        )

        run.log(
            {
                "combination": combination,
                "missing_feature": missing_feature,
                "mean": scores.avg_score,
                "std": scores.std_score,
                "min": scores.min_score,
                "max": scores.max_score,
            }
        )


if __name__ == "__main__":
    csv_path = "notebooks/21022024-processed.csv"
    for i in range(24):
        main(csv_path, combination_size=i + 2)
