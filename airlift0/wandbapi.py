import wandb
import pandas as pd

# Connect to wandb API
api = wandb.Api()

# Load your run (replace with your entity and project names)
run = api.run("socallahan-air-force-institute-of-technology/new_thesis/36679_00000")

# Fetch the history (all metrics logged per step)
history = run.history(keys=["training_iteration", "env_runners/episode_return_mean", "env_runners/custom_metrics/proportion_deliveries_missed_mean"
                            , "env_runners/custom_metrics/env/number_of_initial_cargo_mean", "env_runners/custom_metrics/env/number_of_agents_mean",
                            "env_runners/custom_metrics/env/number_of_airports_mean"])

# Keep just the columns you want
df = history[["training_iteration", "env_runners/episode_return_mean", "env_runners/custom_metrics/proportion_deliveries_missed_mean"
                            , "env_runners/custom_metrics/env/number_of_initial_cargo_mean", "env_runners/custom_metrics/env/number_of_agents_mean",
                            "env_runners/custom_metrics/env/number_of_airports_mean"]].dropna()

# Show results
print(df.head())

# Save to CSV if needed
df.to_csv("episode_returns.csv", index=False)