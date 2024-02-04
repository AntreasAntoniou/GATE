import pathlib
import random

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

import wandb

# Initialize Weights and Biases
run = wandb.init(
    project="combination-optimization", entity="machinelearningbrewery"
)


# Define the loss function
def compute_loss(logits, labels):
    return F.mse_loss(logits, labels) + F.l1_loss(logits, labels)


# Function to mutate a combination
def mutate_combination(combination, all_metrics):
    metric_to_remove = random.choice(combination)
    metric_to_add = random.choice(all_metrics)
    while metric_to_add in combination:
        metric_to_add = random.choice(all_metrics)
    new_combination = list(combination)
    new_combination.remove(metric_to_remove)
    new_combination.append(metric_to_add)
    new_combination = sorted(new_combination)
    new_combination = set(new_combination)
    return tuple(new_combination)


# Function to evaluate a combination
def evaluate_combination(df, combination, device, epochs=20):
    input_metrics = [
        item
        for item in df.columns
        if any(metric in item for metric in combination)
    ]
    target_metrics = list(set(df.columns) - set(input_metrics))
    X = df[input_metrics].values
    y = df[target_metrics].values
    kf = KFold(n_splits=10)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_t = torch.tensor(X_train).to(device).float()
        X_test_t = torch.tensor(X_test).to(device).float()
        y_train_t = torch.tensor(y_train).to(device).float()
        y_test_t = torch.tensor(y_test).to(device).float()
        model = nn.Linear(X_train_t.shape[1], y_train_t.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
        model.reset_parameters()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = compute_loss(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            y_pred = model(X_test_t)
            score = compute_loss(y_pred, y_test_t)
        scores.append(score.cpu().detach().numpy())
    avg_score = np.mean(scores)
    return avg_score, combination


def get_combinations(metrics: list, max_length: int = 12):
    from itertools import combinations

    # List of metrics

    metrics = list(metrics)
    print(len(metrics))
    print(metrics)

    # Calculate combinations of 12 datasets
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
            "cityscapes",
            "cityscapes",
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


def main(result_csv_path: str | pathlib.Path):

    # Load the CSV file
    df = pd.read_csv(result_csv_path)
    df = df.iloc[:, 1:]
    df = df.iloc[1:, :]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    population_size = 25
    mutation_rounds = 10
    num_generations = 100
    out = get_combinations(metrics=METRICS, max_length=12)
    random_combinations = random.sample(out, 500)
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

    for i, (score, combination) in enumerate(top_combinations):
        print(f"Rank: {i}, Score: {score}, Combination: {combination}")

    for i in range(num_generations):
        new_combinations = []
        for score, combination in top_combinations:
            new_combinations.append(combination)
            for _ in range(mutation_rounds):
                mutated_combination = mutate_combination(combination, METRICS)
                new_combinations.append(mutated_combination)
        score_combinations = []
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
        top_combinations = sorted(score_combinations)[:population_size]
        for i, (score, combination) in enumerate(top_combinations):
            print(
                f"Generation: {i}, Rank: {i}, Score: {score}, Combination: {combination}"
            )

        # Log the mean and standard deviation of the scores at this generation
        scores = [score for score, _ in top_combinations]
        wandb.log(
            {
                "generation": i,
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
            }
        )

        # Save the top combinations as an artifact
        top_combinations_df = pd.DataFrame(
            top_combinations, columns=["score", "combination"]
        )
        top_combinations_df.to_csv("top_combinations.csv")
        artifact = wandb.Artifact("top_combinations", type="top_combinations")
        artifact.add_file("top_combinations.csv")
        run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    fire.Fire(main)
