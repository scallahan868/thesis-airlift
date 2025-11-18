import wandb
import pandas as pd

# Connect to wandb API
api = wandb.Api()

# Load your run (replace with your entity and project names)
run = api.run("socallahan-air-force-institute-of-technology/new_thesis/feedf_00000")

# Fetch the history (all metrics logged per step)
history = run.history(keys=["env_runners/episode_return_mean", "training_iteration"])

# Keep just the columns you want
df = history[["training_iteration", "env_runners/episode_return_mean"]].dropna()

# Show results
print(df.head())

# Save to CSV if needed
df.to_csv("episode_returns.csv", index=False)