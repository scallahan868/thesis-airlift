"""
Debug script for RLlib callbacks in Airlift using the legacy API that supports local mode.
This is specifically for allowing VSCode debugging with breakpoints in your AirliftEnv.
"""

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
from airlift.envs.airlift_env_rl import AirliftEnv
from curriculum_maps import DifficultyProgressionMap
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from airlift_callbacks import (
    MissedDeliveriesCallback,
    TotalRewardsCallback,
    AgentRewardsCallback,
    CargoMetricsCallback,
    CurriculumCallback,
)
from train import _wrap_to_gymnasium_if_gym
from simple_flat_wrapper import AirliftSimpleFlattenWrapper
from single_callback import AlgorithmTrainingCallback

from centralized_critic_model import CentralizedCriticModel
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model(
    "centralized_critic_model", CentralizedCriticModel
)
print(ray.__version__)
import time
import json
import os
from airlift.envs.airlift_env_rl import AirliftEnv


CURRICULUM_JSON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../curriculum.json")
)
curriculum_map = DifficultyProgressionMap(CURRICULUM_JSON_PATH)

training_iteration = 45
data_to_write = {
        "current_iteration": 0,
        "max_iterations": training_iteration,
        "test_id": 0,
        "reset_iter": 0,
    }
CURRICULUM_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../curriculum.json"))
with open(CURRICULUM_JSON_PATH, 'w') as f:
    json.dump(data_to_write, f, indent=4)

def env_creator(env_config):
    curriculum_map.refresh()
    world_generator = curriculum_map.get_world_generator()
    world_generator.max_cycles = 5000
    world_generator.test_id = getattr(curriculum_map, "current_testid", 0)

    base_env = AirliftEnv(world_generator=world_generator)
    flat_env = AirliftSimpleFlattenWrapper(base_env)

    # >>> Add this line so wrapper.reset() can read & swap the world each episode
    flat_env.curriculum_map = curriculum_map

    gymnasium_env = _wrap_to_gymnasium_if_gym(flat_env)
    return ParallelPettingZooEnv(gymnasium_env)



register_env("AirliftEnv-v0", env_creator)

if __name__ == "__main__":
    # Use local mode for debugging
    ray.init(ignore_reinit_error=True, local_mode=True)

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"

    from ray.rllib.policy.policy import PolicySpec
    policies = {"shared_policy": PolicySpec()}

    print(ray.__version__)

    # Build PPOConfig - using the legacy approach
    config = (
        PPOConfig()
        .environment(env="AirliftEnv-v0")
        .framework("torch")
        .env_runners(
            # <<< FAST + DEBUG-FRIENDLY SETTINGS >>>
            num_env_runners=0,                 # run env on the driver process
            num_envs_per_env_runner=1,
            rollout_fragment_length=16,        # small fragments => quick sampling
            batch_mode="truncate_episodes"     # don’t wait for episode ends
        )
        .training(
            lr=5e-5,
            train_batch_size=256,              # small batch => quick iteration
            num_sgd_iter=1,                    # a single pass over the batch
            model={
                "custom_model": "centralized_critic_model",
                "custom_model_config": {
                   "local_obs_dim": 48,
                    "central_obs_dim": 414,
                },
            },
        )
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .callbacks(AlgorithmTrainingCallback)
        .debugging(log_level="DEBUG")  # Enable debug logging
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        
    )

    # Create the algorithm directly (not using Tune)
    algo = PPO(config=config)

    # Training loop with debugging
    for i in range(45):
        print(f"\n--- Training iteration {i+1} ---")
        print(ray.__version__)
        result = algo.train()

        # Print some key metrics
        env_runners = result.get("env_runners", {})
        reward_mean = env_runners.get("episode_reward_mean", "N/A")
        length_mean = env_runners.get("episode_len_mean", "N/A")
        print(f"Episode reward mean: {reward_mean}")
        print(f"Episode length mean: {length_mean}")

        # Print custom metrics from callbacks
        # custom_metrics = result.get("env_runners", {}).get("custom_metrics", {})
        # if custom_metrics:
        #     print("Custom metrics:")
        #     for key, value in custom_metrics.items():
        #         print(f"  {key}: {value}")
              # safe now (num_resets>0)

        # You can set breakpoints here to debug
        if i == 0:
            # This is a good place to set a breakpoint and examine the result
            pass

    # Save the final policy
    checkpoint_dir = algo.save("./debug_checkpoint_airlift")
    print(f"Checkpoint saved to: {checkpoint_dir}")

    algo.stop()
    ray.shutdown()
