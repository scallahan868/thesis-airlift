import wandb
from ray.rllib.callbacks.callbacks import RLlibCallback
import json
import os

import requests


class AlgorithmTrainingCallback(RLlibCallback):
    """Enhanced callback with curriculum progression and detailed logging."""
    
    def __init__(self, algorithm_name="PPO", starting_iteration=0):
        super().__init__()
        self.algorithm_name = algorithm_name
        self.iteration_count = starting_iteration  # Initialize with resume iteration
        

    def on_episode_end(self, *, worker, base_env, episode, env_index=None, **kwargs):
        CURRICULUM_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../curriculum1.json"))
        with open(CURRICULUM_JSON_PATH, "r") as f:
            read_data = json.load(f)
        test_id = read_data.get("test_id")
        
        # Prefer the env RLlib passes. If you wrap your env, use `.unwrapped`
        # e = getattr(env, "unwrapped", env)
        e = base_env._unwrapped_env.get_sub_environments

        # Read your @property values
        metrics = e.metrics          
        env_info = e.env_info        

        # If they’re namedtuples, convert to dicts so Tune can log them cleanly
        md = metrics._asdict()  if hasattr(metrics,  "_asdict") else dict(metrics)
        ei = env_info._asdict() if hasattr(env_info, "_asdict") else dict(env_info)

        ei["test_id"] = test_id

        # Put into Tune’s result stream via episode.custom_metrics
        # (These will appear under `custom_metrics/` in training results.)
        for k, v in md.items():
            episode.custom_metrics[k] = v

        # If you also want the static env_info in the results:
        for k, v in ei.items():
            episode.custom_metrics[f"env/{k}"] = v

        if getattr(wandb, "run", None):
            wandb.log({
                **{f"m/{k}": v for k, v in md.items()},
                **{f"env/{k}": v for k, v in ei.items()},
                "episode_len": episode.length,
                "episode_reward": episode.total_reward,
                "episode_id": episode.episode_id,
            })


    def on_train_result(self, *, algorithm, result, **kwargs):
        self.iteration_count += 1

        CURRICULUM_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../curriculum1.json"))
        with open(CURRICULUM_JSON_PATH, "r") as f:
            read_data = json.load(f)

        # if self.iteration_count > 0 and (self.iteration_count + 1) % (read_data["max_iterations"] // 15) == 0:
        #     read_data["test_id"] = read_data["test_id"] + 1

        # iterations_per_level = read_data["max_iterations"] // 15
        # read_data["test_id"] = min(self.iteration_count // iterations_per_level, 14)

        base_test_id = read_data["test_id"]
        reset_iter = read_data["reset_iter"]

        env_runners = result.get("env_runners", {})
        custom = env_runners.get("custom_metrics", {})

        prop_val = custom.get("proportion_deliveries_missed_mean", None)

        if prop_val is not None and prop_val < 0.3 and reset_iter > 20:
            read_data["test_id"] = base_test_id + 1
            reset_iter = 0
            # data = {
            #     "token": "aem37vgi2ahyjj13uas1rant8tm4e4",
            #     "user": "u9w3eapuf49w2oijy3y7hiimab3d1f",
            #     "title": "Airlift Challenge Notification",
            #     "message": f"Primary Airlift Challenge Training Has Progressed to Test ID {read_data['test_id']}",
            # }
            # response = requests.post("https://api.pushover.net/1/messages.json", data=data)
            # try:
            #     response.raise_for_status()
            #     print("Notification sent! Response:", response.json())
            # except requests.exceptions.HTTPError as e:
            #     print("Error sending notification:", e)
            #     print("Response content:", response.text)
        else:
            read_data["test_id"] = base_test_id
            reset_iter += 1

        data_to_write = {
            "current_iteration": self.iteration_count,
            "max_iterations": read_data["max_iterations"],
            "test_id": read_data["test_id"],
            "reset_iter": reset_iter
            }
        with open(CURRICULUM_JSON_PATH, 'w') as f:
                json.dump(data_to_write, f, indent=4)

        # 2) Optionally also log to W&B
        if getattr(wandb, "run", None):
            wandb.log({
                "iteration": self.iteration_count,
                "test_id": read_data["test_id"],
            })