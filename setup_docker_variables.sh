#!/bin/bash
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOTFjMTY5Zi03ZGUwLTQ4ODYtYWI0Zi1kZDEzNjlkMGI5ZjQifQ=="
export NEPTUNE_PROJECT=MachineLearningBrewery/gate-dev-0-8-0
export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'

export WANDB_API_KEY="821661c6ee1657a2717093701ab76574ae1a9be0"
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=gate-dev-0-8-0

export KAGGLE_USERNAME="antreasantoniou"
export KAGGLE_KEY="d14aab63e71334cfa118bd5251bf85da"

export PYTEST_DIR="/data/"

export EXPERIMENT_NAME=gate-dev-0-8-0
export HF_USERNAME="Antreas"
export HF_TOKEN=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
export HF_CACHE_DIR=$PYTEST_DIR

export TOKENIZERS_PARALLELISM=False

export CODE_DIR=/devcode/GATE-private/
export PROJECT_DIR="/data/"
export EXPERIMENT_NAME_PREFIX="gate-dev-0"
export EXPERIMENTS_ROOT_DIR=$PROJECT_DIR
export EXPERIMENTS_DIR=$PROJECT_DIR
export EXPERIMENT_DIR=$PROJECT_DIR
export DATASET_DIR=$PYTEST_DIR
export MODEL_DIR=$PROJECT_DIR

