# debug_wrapper.py
import time
import numpy as np

# --- imports should mirror your train.py imports ---
from airlift.envs.airlift_env import AirliftEnv          # adjust if your path differs
from airlift.simple_flat_wrapper import AirliftSimpleFlattenWrapper
from airlift.train import _wrap_to_gymnasium_if_gym
from curriculum_maps import DifficultyProgressionMap

curriculum_map = DifficultyProgressionMap(seed=int(time.time()) % 10000)


def make_env(seed: int = 0):
    world_generator = curriculum_map.get_world_generator()
    world_generator.max_cycles = 5000
    world_generator.test_id = getattr(curriculum_map, "current_testid", 0)

    base_env = AirliftEnv(world_generator=world_generator)
    base_env = _wrap_to_gymnasium_if_gym(base_env)   # no-op if already Gymnasium
    env = AirliftSimpleFlattenWrapper(base_env)      # your manual flattener
    return env


def describe_space(space, indent="  "):
    """Compact printer for Gymnasium spaces."""
    try:
        import gymnasium as gym
    except Exception:
        import gym as gym  # fallback, if needed

    if isinstance(space, gym.spaces.Box):
        return f"Box(low={np.min(space.low):.3g}, high={np.max(space.high):.3g}, shape={space.shape}, dtype={space.dtype})"
    elif isinstance(space, gym.spaces.Discrete):
        return f"Discrete(n={space.n})"
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return f"MultiDiscrete(nvec={space.nvec})"
    elif isinstance(space, gym.spaces.MultiBinary):
        return f"MultiBinary(n={space.n})"
    elif isinstance(space, gym.spaces.Dict):
        lines = ["Dict("]
        for k, v in space.spaces.items():
            lines.append(f"{indent}{k}: {describe_space(v, indent + '  ')}")
        lines.append(")")
        return "\n".join(lines)
    elif isinstance(space, gym.spaces.Tuple):
        return "Tuple(" + ", ".join(describe_space(s, indent + "  ") for s in space.spaces) + ")"
    else:
        return f"{type(space).__name__} (unhandled)"

def build_sample_actions(env):
    """
    Build a valid multi-agent action dict by sampling each agent's action_space.
    Works because the wrapper proxies the base env's action spaces.
    """
    actions = {}
    for agent in env.agents:
        try:
            actions[agent] = env.action_space(agent).sample()
        except Exception as e:
            raise RuntimeError(f"Failed to sample action for {agent}: {e}")
    return actions

def main():
    env = make_env(seed=0)

    # Reset and get first observation batch
    obs = env.reset(seed=0)

    print("\n=== Agents ===")
    print(env.possible_agents)

    print("\n=== Per-agent observation spaces (AFTER wrapper) ===")
    for agent_id in env.possible_agents:
        try:
            space = env.observation_space(agent_id)
        except Exception as e:
            print(f"  {agent_id}: ERROR reading observation_space -> {e}")
            continue
        print(f"\nAgent: {agent_id}\n{describe_space(space)}")

    # Show one sample observation (sizes/dtypes) for each agent
    print("\n=== Sample observation snapshot ===")
    for agent_id in env.possible_agents:
        agent_obs = obs.get(agent_id, None)
        if agent_obs is None:
            print(f"  {agent_id}: no observation on reset()")
            continue
        if isinstance(agent_obs, dict):
            print(f"\n  {agent_id}: Dict with {len(agent_obs)} keys")
            for k, v in agent_obs.items():
                try:
                    arr = np.asarray(v)
                    print(f"    {k:20s} shape={arr.shape} dtype={arr.dtype} "
                          f"nan_count={np.isnan(arr).sum() if arr.dtype.kind=='f' else 'n/a'}")
                except Exception as e:
                    print(f"    {k:20s} <non-arrayable> -> {type(v)} ({e})")
        else:
            arr = np.asarray(agent_obs)
            print(f"\n  {agent_id}: shape={arr.shape} dtype={arr.dtype} "
                  f"nan_count={np.isnan(arr).sum() if arr.dtype.kind=='f' else 'n/a'}")

    # Take a single "no-op" step if possible to ensure step path works.
    print("\n=== One dummy step ===")
    try:
        actions = build_sample_actions(env)
        step_obs, step_rew, step_term, step_trunc, step_info = env.step(actions)
        # Print something small to confirm it worked
        any_agent = env.agents[0]
        ao = step_obs.get(any_agent, {})
        print(f"  Step ok. Example agent={any_agent}, keys={list(ao.keys())[:5]} ...")
    except Exception as e:
        print("Step raised an error:", e)

    try:
        obs2, rews, terms, truncs, infos = env.step(actions)
        print("Step ok:",
              f"obs_keys={list(obs2.keys())[:3]}{'...' if len(obs2)>3 else ''},",
              f"rewards_len={len(rews)}, terms_any={any(terms.values()) if terms else False},",
              f"truncs_any={any(truncs.values()) if truncs else False}")
    except Exception as e:
        print("Step raised an error:", e)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
