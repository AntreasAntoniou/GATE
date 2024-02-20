import wandb

run = wandb.init()
artifact = run.use_artifact(
    "machinelearningbrewery/combination-optimization/top_combinations:v99",
    type="top_combinations",
)
artifact_dir = artifact.download()
print(artifact_dir)
