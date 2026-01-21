import sys
import os

sys.path.append(os.path.dirname(__file__))

import pathlib
import time

import ray
from ray import train, tune

# IMPORTANT: Import your custom spaces module so its @flatten/@flatdim
# registrations are active before RLlib inspects spaces.
from airlift.envs import spaces as airlift_spaces

# Import your custom spaces so we can `isinstance`-check them.
from airlift.envs.spaces import List as AirList, DiGraph as AirDiGraph

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.connectors.env_to_module import FlattenObservations

#from pz_space_compat import PZSpaceCompatWrapper
from simple_flat_wrapper import AirliftSimpleFlattenWrapper

from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


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

from airlift.envs.airlift_env_rl import AirliftEnv
from curriculum_maps import DifficultyProgressionMap

from pettingzoo.test import parallel_api_test

from ray.rllib.utils.framework import try_import_torch  # just to ensure torch is present
_, _ = try_import_torch()

# ---- training config ----
training_iteration = 1500

# Create curriculum map object - this gets shared across all environment instances
curriculum_map = DifficultyProgressionMap(seed=int(time.time()) % 10000)


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


def env_creator(env_config):
    world_generator = curriculum_map.get_world_generator()
    world_generator.max_cycles = 5000
    world_generator.test_id = getattr(curriculum_map, "current_testid", 0)

    base_env = AirliftEnv(world_generator=world_generator)
    
    # Normalize spaces to Gymnasium before RLlib wraps the env
    compat_env = AirliftSimpleFlattenWrapper(base_env)

    gymnasium_env = _wrap_to_gymnasium_if_gym(compat_env)

    return gymnasium_env

register_env("AirliftEnv-v0", lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Create the environment instance
    env = env_creator({})    # Pass an empty config or your desired config

    # Run the parallel API test
    parallel_api_test(env, num_cycles=10)