"""
Train a policy for the AirliftEnv using Ray RLlib PPO and Ray Tune
Tuner API with Weights & Biases logging. Multi-agent (PettingZoo) version.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import pathlib
import time

import ray
from ray import train, tune

# Import your custom spaces so we can `isinstance`-check them.
from airlift.envs.spaces import List as AirList, DiGraph as AirDiGraph

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

#from pz_space_compat import PZSpaceCompatWrapper
from simple_flat_wrapper import AirliftSimpleFlattenWrapper

# NEW: compatibility shims for old Gym -> Gymnasium
# Prefer Shimmy (covers Gym v0.21 and v0.26). Falls back gracefully if not installed.
try:
    from shimmy.openai_gym_compatibility import (
        GymV21CompatibilityV0 as GymV21ToGymnasium,
    )
    try:
        from shimmy.openai_gym_compatibility import (
            GymV26CompatibilityV0 as GymV26ToGymnasium,
        )
    except Exception:
        GymV26ToGymnasium = None
    _HAVE_SHIMMY = True
except Exception:
    _HAVE_SHIMMY = False
    GymV21ToGymnasium = None
    GymV26ToGymnasium = None

# Optional: direct Gymnasium wrapper (rarely needed if Shimmy is available)
try:
    from gymnasium.wrappers.compatibility import Compatibility as GymCompatibility
except Exception:
    GymCompatibility = None

from airlift.envs.airlift_env_rl import AirliftEnv
from curriculum_maps import DifficultyProgressionMap

from single_callback import AlgorithmTrainingCallback
from ray.rllib.utils.framework import try_import_torch  # just to ensure torch is present
_, _ = try_import_torch()

# ---- training config ----
training_iteration = 100000

# Create curriculum map object - this gets shared across all environment instances
curriculum_map = DifficultyProgressionMap(seed=int(time.time()) % 10000)

from centralized_critic_model import CentralizedCriticModel
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model(
    "centralized_critic_model", CentralizedCriticModel
)

import json
import os
data_to_write = {
        "current_iteration": 0,
        "max_iterations": training_iteration,
        "test_id": 0,
        "reset_iter": 0
    }
CURRICULUM_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../curriculum.json"))
with open(CURRICULUM_JSON_PATH, 'w') as f:
    json.dump(data_to_write, f, indent=4)

def _wrap_to_gymnasium_if_gym(old_env):
    """Wrap a legacy Gym env so it presents the Gymnasium API.

    - If it's PettingZoo (AEC/Parallel), return as-is (RLlib handles via PettingZooEnv).
    - If it's Gym v0.21: use Shimmy's GymV21CompatibilityV0.
    - If it's Gym v0.26: use Shimmy's GymV26CompatibilityV0 (if available).
    - Otherwise (no Shimmy), try Gymnasium's Compatibility wrapper as a fallback.
    """
    # Heuristic: PettingZoo envs have `possible_agents` attribute.
    if hasattr(old_env, "possible_agents"):
        return old_env

    # If the object looks like a Gym env (older API), adapt it.
    if _HAVE_SHIMMY:
        # Prefer the v26 wrapper if present; otherwise v21.
        if GymV26ToGymnasium is not None:
            try:
                return GymV26ToGymnasium(old_env)
            except Exception:
                pass
        if GymV21ToGymnasium is not None:
            try:
                return GymV21ToGymnasium(old_env)
            except Exception:
                pass

    # Fallback: Gymnasium's built-in compatibility wrapper (0.29 has this).
    if GymCompatibility is not None:
        try:
            return GymCompatibility(old_env)
        except Exception:
            pass

    # If nothing matched, just return original (RLlib will error if the API is old).
    return old_env


# def env_creator(env_config):
#     curriculum_map.refresh()
#     world_generator = curriculum_map.get_world_generator()
#     world_generator.max_cycles = 5000
#     world_generator.test_id = getattr(curriculum_map, "current_testid", 0)

#     base_env = AirliftEnv(world_generator=world_generator)
#     flat_env = AirliftSimpleFlattenWrapper(base_env)

#     gymnasium_env = _wrap_to_gymnasium_if_gym(flat_env)
#     return ParallelPettingZooEnv(gymnasium_env)

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

def globalstate_observation_fn(agent_obs, env, *args, **kwargs):
    return {agent_id: obs.get("globalstate", {}) for agent_id, obs in agent_obs.items()}

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    log_dir = pathlib.Path("./rllib_logs").absolute()
    os.makedirs(log_dir, exist_ok=True)

    # Multi-agent policy setup
    # All agents get mapped to the same policy
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"

    policies = {"shared_policy": PolicySpec()}

    config = (
        PPOConfig()
        .environment(env="AirliftEnv-v0", env_config={})
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .env_runners(
        num_env_runners=16,
        env_to_module_connector=None,
        )
        .training(
            lr=5e-5,
            grad_clip = 0.5,  # or 1.0
            num_epochs = 10,  # instead of 30
            model={
            "custom_model": "centralized_critic_model",
            "custom_model_config": {
                "local_obs_dim": 93, #126
                "central_obs_dim": 1608,  # Adjust based on actual global
            }
            }
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .callbacks(AlgorithmTrainingCallback)
        .debugging(log_level="INFO")
    )

    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=train.RunConfig(
            storage_path=log_dir,
            stop={"training_iteration": training_iteration},
            callbacks=[
                WandbLoggerCallback(
                    project="new_thesis",
                    log_config=True,
                    save_checkpoints=True,
                    upload_checkpoints=True,
                )
            ],
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )

    tuner.fit()
    ray.shutdown()