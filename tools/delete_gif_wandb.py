import wandb
from rich import print
from tqdm.auto import tqdm

# Initialize the wandb API
api = wandb.Api()

# Get all runs from the project
runs = api.runs("gate-0-9-1")
# runs = list(runs)
# shuffle(runs)
# Iterate over each run in the project
for run in tqdm(runs):
    # Iterate over each artifact used in the run
    try:
        if (
            "BackboneWithTemporalTransformerAndLinear"
            in run.config["adapter"]["_target_"]
        ):
            for file in run.files():
                if file.name.endswith(".gif"):
                    file.delete()
                    print(f"Deleted {file.name} from {run.name}")
    except Exception as e:
        print(e)
        continue
