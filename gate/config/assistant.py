import os
import pathlib
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown

console = Console()


def ask_user(
        question: str, default: Optional[str] = None, advice: Optional[str] = None
) -> str:
    if advice:
        console.print(Markdown(advice))
    value = console.input(
        f"[bold cyan]{question} (Default is {default}):[/bold cyan] "
    )
    return value.strip() if value.strip() else default


def assistance():
    console.print(
        Markdown(
            """
    This script will guide you to set up the environment variables needed for your machine learning project. 
    Here's an overview of the services we'll be setting up:

    - **Weights & Biases (wandb)**: It's a tool for experiment tracking, dataset versioning, and model management. 
    It helps to keep track of your experiments in your machine learning projects. 
    It logs your hyperparameters and metrics, and allows you to visualize them in a web dashboard.

    - **Hugging Face**: It's a platform to host transformer models and datasets. It has a vast array of 
    pre-trained models contributed by the community. It's used to store model checkpoints to retrieve them later, 
    making the model training code fully stateless.

    - **Kaggle**: It's a platform for predictive modelling and analytics competitions. 
    It allows users to find and publish datasets, explore and build models. 
    The Kaggle API is used here to fetch various key datasets for our experiments.
    """
        )
    )

    api_details = {
        "WANDB_API_KEY": {
            "question": "What's your Weights & Biases API key?",
            "advice": "You can find this in your Weights & Biases settings: "
                      "[link=https://wandb.ai/settings#api](https://wandb.ai/settings#api)",
        },
        "KAGGLE_USERNAME": {
            "question": "What's your Kaggle username?",
            "advice": "This is the username you use to log into Kaggle.",
        },
        "KAGGLE_KEY": {
            "question": "What's your Kaggle API key?",
            "advice": "You can generate this from your Kaggle account settings.",
        },
        "HF_USERNAME": {
            "question": "What's your Hugging Face username?",
            "advice": "This is the username you use to log into Hugging Face.",
        },
        "HF_TOKEN": {
            "question": "What's your Hugging Face token?",
            "advice": "You can generate this from your Hugging Face account settings: "
                      "[link=https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)",
        },
    }

    wandb_details = {
        "WANDB_ENTITY": {
            "question": "What's your Weights & Biases entity?",
            "advice": "This is the team name in Weights & Biases under which your project resides.",
        },
        "WANDB_PROJECT": {
            "question": "What's your Weights & Biases project?",
            "advice": "This is the name of your project in Weights & Biases.",
        },
        "EXPERIMENT_NAME": {
            "question": "What's the name of your experiment?",
            "advice": "Choose a name that reflects the focus of your experiment.",
        },
    }

    directories = {
        "DATASET_DIR": {
            "question": "Where is your dataset directory located?",
            "advice": "Please provide the full path to the directory where your datasets are stored.",
        },
        "PROJECT_DIR": {
            "question": "Where is your project directory located?",
            "advice": "Please provide the full path to your main project directory.",
        },
        "CODE_DIR": {
            "question": "Where is your code directory located?",
            "advice": "Please provide the full path to the directory where your code files are stored.",
        },
        "EXPERIMENTS_ROOT_DIR": {
            "question": "Where is your experiments root directory located?",
            "advice": "Please provide the full path to the root directory for your experiments.",
        },
        "EXPERIMENTS_DIR": {
            "question": "Where is your experiments directory located?",
            "advice": "Please provide the full path to the directory where you keep your experiment files.",
        },
        "MODEL_DIR": {
            "question": "Where is your model directory located?",
            "advice": "Please provide the full path to the directory where your model files are stored.",
        },
        "PYTEST_DIR": {
            "question": "Where is your pytest directory located?",
            "advice": "Please provide the full path to the directory where your pytest files are stored.",
        },
        "HF_CACHE_DIR": {
            "question": "Where is your Hugging Face cache directory located?",
            "advice": "Please provide the full path to the directory where your Hugging Face cache files are stored.",
        },
    }

    groups = [api_details, wandb_details, directories]
    defaults = {
        "WANDB_API_KEY": None,
        "KAGGLE_USERNAME": None,
        "KAGGLE_KEY": None,
        "HF_USERNAME": None,
        "HF_TOKEN": None,
        "WANDB_ENTITY": None,
        "WANDB_PROJECT": None,
        "EXPERIMENT_NAME": "$WANDB_PROJECT",
        "DATASET_DIR": "/data/",
        "PROJECT_DIR": "/data/experiments/",
        "TOKENIZERS_PARALLELISM": False,
        "CODE_DIR": "$PWD",
        "EXPERIMENTS_ROOT_DIR": "$PROJECT_DIR",
        "EXPERIMENTS_DIR": "$PROJECT_DIR",
        "MODEL_DIR": "$PROJECT_DIR",
        "PYTEST_DIR": "$DATASET_DIR",
        "HF_CACHE_DIR": "$PYTEST_DIR",
    }
    home_dir = pathlib.Path.home()
    config_path = home_dir / ".gate" / "config.sh"

    if config_path.exists():
        console.print(
            f"[bold yellow]Warning![/bold yellow] A configuration file already exists at {config_path}."
        )
        overwrite = console.input(
            "Would you like to overwrite the existing configuration? (yes/no): "
        )
        if overwrite.lower() != "yes":
            # source from the file to make the changes effective
            os.system(f"bash -c 'source {config_path}'")
            return

    if not config_path.parent.exists():
        config_path.parent.mkdir(exist_ok=True, parents=True)

    with open(config_path, "w") as f:
        for group in groups:
            for var, details in group.items():
                question = details["question"]
                advice = details.get("advice")
                default = defaults[var]
                val = ask_user(question, default, advice)
                if val is not None:
                    f.write(f"export {var}={val}\n")

    # source the file to make the changes effective
    os.system(f"source {config_path}")  # does not work

    console.print(
        f"[bold green]Done![/bold green] Your settings have been saved to {config_path}"
    )
