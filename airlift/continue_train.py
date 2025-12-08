"""
continue_train_wandb.py

Continue PPO training for AirliftEnv from an existing RLlib/Tune checkpoint,
using the JSON configuration (params.json) stored alongside that checkpoint,
and log metrics to Weights & Biases (wandb).

This script does **not** use Ray Tune's Tuner; instead it restores an
RLlib PPO algorithm directly and runs a manual training loop for full control.
"""

import os
import sys
import pathlib
import argparse
import json
import time

sys.path.append(os.path.dirname(__file__))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector

import wandb

# Ensure torch is present (same pattern as train.py)
_, _ = try_import_torch()

# --- Local project imports (mirroring train.py) ---
from airlift.envs.airlift_env_rl import AirliftEnv
from curriculum_maps import DifficultyProgressionMap
from simple_flat_wrapper import AirliftSimpleFlattenWrapper
from centralized_critic_model import CentralizedCriticModel
from single_callback import AlgorithmTrainingCallback

# ===== Shimmy / Gymnasium compatibility helpers (copied from train.py) =====
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

try:
    from gymnasium.wrappers.compatibility import Compatibility as GymCompatibility
except Exception:
    GymCompatibility = None


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
        # Prefer v26 wrapper if present; otherwise v21.
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


# ===== Curriculum + env registration (DOES NOT overwrite curriculum.json) =====

# Reuse whatever curriculum.json currently says; don't reset it here.
curriculum_map = DifficultyProgressionMap(seed=int(time.time()) % 10000)


def env_creator(env_config):
    """Environment creator matching train.py, but without resetting curriculum.json."""
    curriculum_map.refresh()
    world_generator = curriculum_map.get_world_generator()
    world_generator.max_cycles = 5000
    world_generator.test_id = getattr(curriculum_map, "current_testid", 0)

    base_env = AirliftEnv(world_generator=world_generator)
    flat_env = AirliftSimpleFlattenWrapper(base_env)

    # Allow wrapper.reset() to read & swap the world each episode.
    flat_env.curriculum_map = curriculum_map

    gymnasium_env = _wrap_to_gymnasium_if_gym(flat_env)
    return ParallelPettingZooEnv(gymnasium_env)


# Register env and model up-front so config.build()/restore can find them.
register_env("AirliftEnv-v0", env_creator)
ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCriticModel)


# ===== Helpers for loading config + checkpoint =====

def find_params_json(checkpoint_path: str) -> pathlib.Path:
    """
    Given a checkpoint path (file or directory), walk up the directory tree
    looking for a params.json (the RLlib/Tune config).

    Handles layouts like:
      trial_dir/
        params.json
        result.json
        checkpoint_000042/
          ...
    """
    cp = pathlib.Path(checkpoint_path)

    # Start from checkpoint dir (or its parent if a file)
    start_dir = cp if cp.is_dir() else cp.parent

    # Walk upwards until filesystem root
    for d in [start_dir, *start_dir.parents]:
        cfg_path = d / "params.json"
        if cfg_path.exists():
            print(f"[find_params_json] Using params.json at: {cfg_path}")
            return cfg_path

    raise FileNotFoundError(
        f"Could not find params.json in {start_dir} or any of its parent "
        f"directories. Start dir was: {start_dir}"
    )


def load_config_from_params_json(params_path: pathlib.Path) -> PPOConfig:
    with params_path.open("r") as f:
        cfg_dict = json.load(f)

    # --- Clean up bad serialized callback class path ---
    # Some Ray/Tune versions will store custom callbacks as
    # "<class 'single_callback.AlgorithmTrainingCallback'>",
    # which breaks deserialize_type. We'll drop it here and
    # reattach the real class in code later.
    bad_cb = cfg_dict.get("callbacks")
    if isinstance(bad_cb, str) and "single_callback.AlgorithmTrainingCallback" in bad_cb:
        print(f"[load_config_from_params_json] Removing bad callbacks entry from params.json: {bad_cb}")
        cfg_dict.pop("callbacks", None)

    # Now safely create PPOConfig from the cleaned dict.
    return PPOConfig().from_dict(cfg_dict)


# ===== Main training loop for continuing from checkpoint with wandb =====

def continue_training(
    checkpoint_path: str,
    num_iterations: int,
    save_every: int,
    wandb_project: str = "new_thesis",
    wandb_run_name: str | None = None,
):
    ray.init(ignore_reinit_error=True)

    # Multi-agent mapping: all agents share one policy (same as train.py)
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"

    policies = {"shared_policy": PolicySpec()}

    params_path = find_params_json(checkpoint_path)
    config = load_config_from_params_json(params_path)
    
    # Some configs serialize sample_collector as a string path, which can
    # survive deserialization and cause `'str' object is not callable`.
    if isinstance(config.sample_collector, str):
        print(f"[continue_training] sample_collector is a string ({config.sample_collector}); "
              f"resetting to SimpleListCollector.")
        config.sample_collector = SimpleListCollector

    # Ensure callbacks and multi-agent bits are set (in case they weren't fully
    # serialized, or you want to be explicit).
    config = (
        config
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .callbacks(AlgorithmTrainingCallback)
    )

    algo = config.build()
    algo.restore(checkpoint_path)

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Using config from:      {params_path}")
    print(f"Continuing for {num_iterations} iterations\n")

    # --- Start Weights & Biases run ---
    wandb_config = {
        "checkpoint_path": str(checkpoint_path),
        "params_json": str(params_path),
        "num_iterations": num_iterations,
        "save_every": save_every,
    }
    # Add RLlib config for convenience (keep it lightweight).
    try:
        wandb_config["rllib_config"] = config.to_dict()
    except Exception:
        pass

    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=wandb_config,
    )

    try:
        for i in range(1, num_iterations + 1):
            result = algo.train()
            # Basic logging; customize as needed.
            ep_rew_mean = result.get("episode_reward_mean", None)
            timesteps_total = result.get("timesteps_total", None)
            training_iteration = result.get("training_iteration", i)

            print(f"[Iter {training_iteration}] "
                  f"episode_reward_mean={ep_rew_mean}, timesteps_total={timesteps_total}")

            # Log a subset of metrics to wandb.
            log_dict = {
                "training_iteration": training_iteration,
                "episode_reward_mean": ep_rew_mean,
                "timesteps_total": timesteps_total,
            }
            for k in ["episode_len_mean", "episodes_this_iter", "episodes_total"]:
                if k in result:
                    log_dict[k] = result[k]

            wandb.log(log_dict)

            if save_every and i % save_every == 0:
                ckpt = algo.save()
                ckpt_path = ckpt.checkpoint.path if hasattr(ckpt, "checkpoint") else ckpt
                print(f"  Saved checkpoint at: {ckpt_path}\n")
                wandb.log({"checkpoint_path": str(ckpt_path)})

        # Final checkpoint at the end
        final_ckpt = algo.save()
        final_ckpt_path = final_ckpt.checkpoint.path if hasattr(final_ckpt, "checkpoint") else final_ckpt
        print(f"\nFinal checkpoint saved at: {final_ckpt_path}")
        wandb.log({"final_checkpoint_path": str(final_ckpt_path)})

    finally:
        run.finish()
        ray.shutdown()


if __name__ == "__main__":
    CHECKPOINT_PATH = "/opt/project/airlift/rllib_logs/PPO_2025-12-03_15-53-02/PPO_AirliftEnv-v0_266ce_00000_0_2025-12-03_15-53-02/checkpoint_000042"
    NUM_ITERATIONS = 10000          # How many additional training iterations to run
    SAVE_EVERY = 10               # Save checkpoint every N iterations (0 = never)
    WANDB_PROJECT = "new_thesis"  # W&B project name
    WANDB_RUN_NAME = "resume_run_266ce" # Optional W&B run name (None for auto)

    continue_training(
        checkpoint_path=CHECKPOINT_PATH,
        num_iterations=NUM_ITERATIONS,
        save_every=SAVE_EVERY,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
    )
