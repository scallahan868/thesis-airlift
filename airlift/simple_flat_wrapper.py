from __future__ import annotations

import numpy as np
import networkx as nx

from typing import Dict, Any, Iterable

from gym.utils import seeding

# Gym/Gymnasium flatten utils (single-dispatch) so we can register our custom handlers.
# Either gym or gymnasium provides these under gym.spaces.utils
from gym.spaces.utils import flatten as gym_flatten, flatdim as gym_flatdim

# Airlift custom space types
import airlift.envs.spaces as airliftspaces

List = airliftspaces.List
DiGraph = airliftspaces.DiGraph
from airlift.envs.airlift_env_rl import AirliftEnv


# -------------------------------
# Manual flatdim/flatten handlers
# -------------------------------

@gym_flatdim.register(List)
def _flatdim_list(space: List) -> int:
    # Max size times element flatdim
    return space.maxsize * gym_flatdim(space.space)


@gym_flatten.register(List)
def _flatten_list(space: List, x: Iterable) -> np.ndarray:
    """
    Returns a 1D array of length (maxsize * flatdim(element_space)).
    Unused slots are filled with NaNs (so shape stays static).
    """
    out = np.empty((gym_flatdim(space),), dtype=float)
    out[:] = np.nan

    # Nothing to do?
    if x is None:
        return out

    # Concatenate flattened elements we actually have
    elems = list(x)
    elem_flat = [gym_flatten(space.space, xi) for xi in elems]
    if elem_flat:
        flat = np.concatenate(elem_flat).astype(float, copy=False)
        out[: flat.size] = flat
    return out

_PAD_VALUE = -1.0

def _sanitize_obs(obs_dict):
    """Replace NaNs, fix shapes, cast to float32 for RLlib."""
    out = {}
    for k, v in obs_dict.items():
        arr = np.asarray(v)
        # Ensure 1D shape for scalars (Gymnasium Box expects matching shapes)
        if arr.shape == ():
            arr = arr.reshape(1)
        # Replace NaNs in all numeric arrays
        if np.issubdtype(arr.dtype, np.number):
            arr = np.nan_to_num(arr, nan=_PAD_VALUE, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
            arr = arr.astype(np.float32, copy=False)
        else:
            # If any non-numeric slipped in, drop/encode it (shouldn't happen after your transforms)
            # Simplest is to represent as a single -1.0 token:
            arr = np.array([_PAD_VALUE], dtype=np.float32)
        out[k] = arr
    return out


@gym_flatdim.register(DiGraph)
def _flatdim_digraph(space: DiGraph) -> int:
    # For each attribute (node + edge), we export a full nnodes x nnodes matrix (flattened)
    n_attr = len(space.node_attributes) + len(space.edge_attributes)
    return space.nnodes * space.nnodes * n_attr


@gym_flatten.register(DiGraph)
def _flatten_digraph(space: DiGraph, g: nx.DiGraph) -> np.ndarray:
    """
    Build a big feature vector by concatenating adjacency-like matrices
    for every requested attribute in node_attributes + edge_attributes.
    Missing/non-numeric values are coerced to 0.0. Nodes are 0..nnodes-1.
    """
    # Ensure the node set is exactly [0..nnodes-1]
    for n in range(space.nnodes):
        if n not in g:
            g.add_node(n)

    mats = []
    for attr in list(space.node_attributes) + list(space.edge_attributes):
        if not isinstance(attr, str):
            continue
        # sanitize edge attributes to numeric
        for u, v, data in g.edges(data=True):
            val = data.get(attr, 0.0)
            if not isinstance(val, (int, float)):
                try:
                    data[attr] = float(val)
                except Exception:
                    data[attr] = 0.0
        mats.append(
            nx.to_numpy_array(g, weight=attr, nodelist=range(space.nnodes)).astype(float).ravel()
        )
    if mats:
        return np.concatenate(mats, dtype=float)
    # still return the correct static size (all zeros) if no attrs
    return np.zeros((_flatdim_digraph(space),), dtype=float)

def _pad_with_nans(values, maxlen, dtype=float):
    arr = np.zeros((maxlen,), dtype=dtype)          # <- pad with 0.0
    arr[:len(values)] = np.asarray(values, dtype=dtype)
    return arr

# --------------------------------------------
# A minimal ParallelEnv observation “fixer-upper”
# --------------------------------------------

class AirliftSimpleFlattenWrapper:
    """
    Wrap a ParallelEnv-like Airlift environment and manually flatten ONLY
    the custom List and DiGraph parts inside the observations/state.
    Everything else (ints/floats/dicts) is passed through.

    Usage:
        base_env = AirliftEnv(... )
        env = AirliftSimpleFlattenWrapper(base_env)
    """
    def __init__(self, env):
        self.env = env

        # Grab world generator for sizing hints (fall back to safe constants)
        wg = getattr(self.env, "world_generator", None)

        # Fixed, numeric limits (store as attributes — not callables)
        # self.max_cargo_per_plane = int(
        #     getattr(wg, "max_cargo_per_plane",
        #             getattr(wg, "max_cargo_per_episode", 64)) or 64
        # )
        self.max_cargo_per_plane = int(23)
        # If env doesn't expose a per-airport cap, use per-episode as a safe upper bound.
        # self.max_cargo_per_airport = int(
        #     getattr(wg, "max_cargo_per_airport",
        #             getattr(wg, "max_cargo_per_episode", 64)) or 64
        # )
        self.max_cargo_per_airport = int(20)
        # If env doesn't expose this, use max_airports as a conservative cap.
        # self.max_routes_per_airport = int(
        #     getattr(wg, "max_routes_per_airport",
        #             getattr(wg, "max_airports", 32)) or 32
        # )
        self.max_routes_per_airport = int(14)

        self.max_agents = int(24)

        # For action list padding (used by debug scaffolding / future action adapters)
        self._action_maxlens = {
            "cargo_to_load": self.max_cargo_per_plane,
            "cargo_to_unload": self.max_cargo_per_plane,
        }

        # self.max_cargo_per_episode = int(getattr(wg, "max_cargo_per_episode", 256) or 256)
        # self.max_airports = int(getattr(wg, "max_airports", 64) or 64)
        # self.num_possible_agents = len(getattr(self.env, "possible_agents", []))
        # self._last_central_state = None 

        self.max_cargo_per_episode = int(72)
        self.max_airports = int(12)
        self.num_possible_agents = int(24)
        self._last_central_state = None 

        # Total number of logits / action entries
        self.mask_dim = (
            2 * self.max_cargo_per_airport
            + 2 * self.max_cargo_per_plane
            + self.max_routes_per_airport
            + 1
        )

        self.prev_action_per_plane_dim = (
            self.max_cargo_per_airport
            + self.max_cargo_per_plane
            + self.max_routes_per_airport
            + 1
        )

        # Total size of previous_action vector: one mask_dim per plane slot
        self.prev_action_dim = int(self.num_possible_agents) * int(self.prev_action_per_plane_dim)

        # Buffer for previous actions (concatenated over all planes)
        self._last_actions_for_all_agents = None

        # --- NEW: caches so we return the SAME space objects each call ---
        self._action_space_cache = {}         # agent_id -> gymnasium.Space
        self._observation_space_cache = {}    # agent_id -> gymnasium.Space

    # PettingZoo Parallel API passthroughs
    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents
    
    @property
    def unwrapped(self):
        # If the underlying env itself is a wrapper, defer to its .unwrapped;
        # otherwise just return the base env.
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, *, seed=None, options=None):
        cm = getattr(self, "curriculum_map", None) or getattr(self.env, "curriculum_map", None)
        changed_test = False

        if cm is not None:
            try:
                cm.refresh()
                new_wg = cm.get_world_generator()

                # Preserve/seed as you already do
                try:
                    old_wg = getattr(self.env, "world_generator", None)
                    if old_wg is not None and hasattr(old_wg, "max_cycles"):
                        new_wg.max_cycles = old_wg.max_cycles
                except Exception:
                    pass

                # robust seeding for new_wg + sub-generators (keep your block here)
                base_seed = (
                    seed if seed is not None
                    else getattr(cm, "current_seed", None) if getattr(cm, "current_seed", None) is not None
                    else 0
                )
                if hasattr(new_wg, "seed"):
                    try: new_wg.seed(base_seed)
                    except TypeError: pass
                sg = getattr(new_wg, "airport_generator", None)
                if sg is not None:
                    if hasattr(sg, "seed"): sg.seed(base_seed + 1)
                    elif not hasattr(sg, "_np_random") or sg._np_random is None:
                        rng, _ = seeding.np_random(base_seed + 1); sg._np_random = rng
                sg = getattr(new_wg, "route_generator", None)
                if sg is not None:
                    if hasattr(sg, "seed"): sg.seed(base_seed + 2)
                    elif not hasattr(sg, "_np_random") or sg._np_random is None:
                        rng, _ = seeding.np_random(base_seed + 2); sg._np_random = rng
                sg = getattr(new_wg, "cargo_generator", None)
                if sg is not None:
                    if hasattr(sg, "seed"): sg.seed(base_seed + 3)
                    elif not hasattr(sg, "_np_random") or sg._np_random is None:
                        rng, _ = seeding.np_random(base_seed + 3); sg._np_random = rng

                # If test_id changed (or plane count likely changed), rebuild env
                old_id = getattr(getattr(self.env, "world_generator", None), "test_id", None)
                new_id = getattr(new_wg, "test_id", None)
                # changed_test = (old_id != new_id)
                changed_test = True

                if changed_test:
                    # Cleanly replace the base env so possible_agents are rebuilt
                    try:
                        if hasattr(self.env, "close"):
                            self.env.close()
                    except Exception:
                        pass
                    # create a fresh AirliftEnv with the new generator
                    new_env = AirliftEnv(world_generator=new_wg)
                    # keep reference to curriculum_map on the new env (optional)
                    setattr(new_env, "curriculum_map", cm)
                    self.env = new_env
                else:
                    # Same test id: in-place swap is fine
                    self.env.world_generator = new_wg

            except Exception as e:
                if getattr(self, "_debug_curriculum", False):
                    print(f"[Curriculum] refresh/swap failed: {e}")

        # Clear previous action history at episode start
        self._last_actions_for_all_agents = None

        # Delegate to underlying env reset (handle legacy API w/o options)
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset(seed=seed)

        # --- your existing post-processing ---
        if isinstance(result, tuple) and len(result) == 2:
            base_obs, base_info = result
        else:
            base_obs, base_info = result, {}
        obs = self.flatten_obs(base_obs)
        self._last_raw_obs = base_obs
        infos = {aid: dict(base_info.get(aid, {})) if isinstance(base_info, dict) else {}
                for aid in obs.keys()}
        return obs, infos

    def step(self, action_dict):
        """
        Normalize to Gymnasium's 5-tuple:
        (obs, rewards, terminations, truncations, infos)
        """

        # --- NEW: encode flat actions for previous_action feature ---
        try:
            self._update_previous_actions_vector(action_dict)
        except Exception as e:
            # Fail-safe: if encoding breaks, just clear history so we don't
            # crash training. You can add logging here if you want.
            self._last_actions_for_all_agents = None


        # Decode flat actions into actual cargo IDs (map slot indices -> cargo ids)
        decoded_action_dict = {
            aid: self._decode_action_from_flat(aid, action)
            for aid, action in action_dict.items()
        }

        result = self.env.step(decoded_action_dict)
        if not isinstance(result, tuple):
            raise TypeError(f"Underlying env.step() returned non-tuple: {type(result)}")

        n = len(result)
        if n == 5:
            # New API (what AirliftEnv implements) – trust it verbatim.
            obs, rewards, terminations, truncations, infos = result

        elif n == 4:
            # Legacy API: (obs, rewards, dones, infos)
            # We cannot reconstruct truncations (time limits) reliably here,
            # so keep them False and treat dones as hard terminations.
            obs, rewards, dones, infos = result

            # Build per-agent dicts; prefer wrapper/env's agent list if present
            if hasattr(self, "agents") and self.agents:
                agent_ids = list(self.agents)
            else:
                agent_ids = set()
                if isinstance(obs, dict):
                    agent_ids.update(obs.keys())
                if isinstance(dones, dict):
                    agent_ids.update([k for k in dones.keys() if k != "__all__"])
                agent_ids = list(agent_ids)

            terminations = (
                {aid: bool(dones.get(aid, False)) for aid in agent_ids}
                if isinstance(dones, dict) else
                {aid: bool(dones) for aid in agent_ids}
            )
            truncations = {aid: False for aid in agent_ids}

        else:
            raise ValueError(f"Expected env.step() to return 4 or 5 values, got {n}.")

        self._last_raw_obs = obs

        obs = self.flatten_obs(obs)
         
        # print("[DEBUG] Flattened obs:", obs)

        return obs, rewards, terminations, truncations, infos

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def action_space(self, agent_id):
        """
        Return a pure Gymnasium action space for one agent.

        We encode variable-length cargo ID lists as fixed-length MultiDiscrete arrays:
        - 0     => "no cargo" (padding)
        - 1..N  => cargo_id + 1 (to stay non-negative for MultiDiscrete)
        """
        if agent_id in self._action_space_cache:
            return self._action_space_cache[agent_id]

        import numpy as np
        from gymnasium import spaces as gspaces

        space = gspaces.Dict({
            "cargo_to_load":   gspaces.MultiDiscrete([2] * self.max_cargo_per_airport),
            "cargo_to_unload": gspaces.MultiDiscrete([2] * self.max_cargo_per_plane),
            "destination":     gspaces.Discrete(self.max_routes_per_airport + 1),
        })
        self._action_space_cache[agent_id] = space
        return space

    def observation_space(self, agent_id: str):
        """
        Observation Dict that matches the NEW flatten_obs() output.
        All sub-entries are float32 Box vectors with fixed shapes.
        """
        # lazy cache so PettingZoo can query per-agent spaces
        if not hasattr(self, "_observation_space_cache"):
            self._observation_space_cache = {}

        if agent_id in self._observation_space_cache:
            return self._observation_space_cache[agent_id]

        from gymnasium.spaces import Box, Dict as GymDict

        mask_dim = 2*self.max_cargo_per_airport + 2*self.max_cargo_per_plane + self.max_routes_per_airport + 1  # num_airports=12 here

        space = GymDict({
            # --- per-agent compact fields ---
            "state":                    Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "current_airport":          Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "available_routes":         Box(-np.inf, np.inf, shape=(self.max_routes_per_airport,), dtype=np.float32),
            # "onboard_count":            Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "cargo_destinations":       Box(-np.inf, np.inf, shape=(self.max_cargo_per_plane,), dtype=np.float32),
            "cargo_at_current_airport": Box(-np.inf, np.inf, shape=(self.max_cargo_per_airport,), dtype=np.float32),
            "cargo_at_current_airport_urgency": Box(-np.inf, np.inf, shape=(self.max_cargo_per_airport,), dtype=np.float32),
            "cargo_onboard":            Box(-np.inf, np.inf, shape=(self.max_cargo_per_plane,), dtype=np.float32),
            "cargo_onboard_urgency":    Box(-np.inf, np.inf, shape=(self.max_cargo_per_plane,), dtype=np.float32),
            # "is_moving":                Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "load_frac":                Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "plane_node_feats":        Box(-np.inf, np.inf, shape=(self.max_airports, 4), dtype=np.float32),
            "cargo_node_feats":        Box(-np.inf, np.inf, shape=(self.max_airports, 4), dtype=np.float32),
            "edge_index":              Box(-np.inf, np.inf, shape=(2, self.max_airports * self.max_routes_per_airport), dtype=np.float32),
            "edge_attr":               Box(-np.inf, np.inf, shape=(self.max_airports * self.max_routes_per_airport, 3), dtype=np.float32),

            # --- centralized critic input (shared vector you attach to each agent) ---
            "globalstate":              Box(-np.inf, np.inf, shape=((3 + self.max_routes_per_airport + 2*self.max_cargo_per_plane + 2*self.max_cargo_per_airport + self.max_cargo_per_plane)*24,), dtype=np.float32),

            # --- NEW: previous actions for all planes ---
            # Values will be in {-1, 0, 1}: -1 for padding / "no action yet",
            # 0/1 for one-hot action choices.
            "previous_action": Box(
                low=-1.0,
                high=1.0,
                shape=(self.prev_action_dim,),
                dtype=np.float32,
            ),

            # -- action mask --
            "action_mask": Box(low=0.0, high=1.0, shape=(mask_dim,), dtype=np.float32),
        })

        self._observation_space_cache[agent_id] = space
        return space

    def close(self):
        return self.env.close()
    def _build_plane_node_feats(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Build per-airport plane node features.

        Returns:
            plane_node_feats: [max_airports, 4] float32
                For each airport i, features are:
                    [ num_planes,
                      total_capacity,
                      total_load,
                      load_fraction ]
        """
        num_airports = int(getattr(self, 'max_airports', 0) or 0)
        if num_airports <= 0:
            return np.zeros((0, 0), dtype=np.float32)

        num_planes = np.zeros((num_airports,), dtype=np.float32)
        total_cap  = np.zeros((num_airports,), dtype=np.float32)
        total_load = np.zeros((num_airports,), dtype=np.float32)

        # Aggregate per-airport from raw agent observations
        for aobs in (obs or {}).values():
            cur_ap = aobs.get('current_airport', -1)
            try:
                ap = int(cur_ap)
            except Exception:
                ap = -1
            if ap < 0 or ap >= num_airports:
                continue

            cw = float(aobs.get('current_weight', 0.0) or 0.0)
            mw = float(aobs.get('max_weight', 0.0) or 0.0)

            num_planes[ap] += 1.0
            total_cap[ap]  += mw
            total_load[ap] += cw

        plane_node_feats = np.zeros((num_airports, 4), dtype=np.float32)
        for ap in range(num_airports):
            cap = total_cap[ap]
            load = total_load[ap]
            load_frac = float(load / cap) if cap > 0.0 else 0.0

            plane_node_feats[ap, :] = [
                num_planes[ap],
                cap,
                load,
                load_frac,
            ]

        return plane_node_feats

    def _build_cargo_node_feats(self, active_cargo: Iterable[Any]) -> np.ndarray:
        """
        Build per-airport cargo node features from the global active cargo list.

        Args:
            active_cargo: iterable of CargoObservation or dict-like objects
                          (typically state['active_cargo']).

        Returns:
            cargo_node_feats: [max_airports, 4] float32
                For each airport i, features are:
                    [ num_cargo,
                      total_weight,
                      total_urgency,
                      avg_urgency ]
        """
        num_airports = int(getattr(self, 'max_airports', 0) or 0)
        if num_airports <= 0:
            return np.zeros((0, 0), dtype=np.float32)

        num_cargo     = np.zeros((num_airports,), dtype=np.float32)
        total_weight  = np.zeros((num_airports,), dtype=np.float32)
        total_urgency = np.zeros((num_airports,), dtype=np.float32)

        # Current time-step (for urgency)
        current_step = float(getattr(self.env, '_elapsed_steps', 0) or 0.0)

        for cg in (active_cargo or []):
            # Robust attribute / dict access
            loc = getattr(cg, 'location', None)
            if loc is None and isinstance(cg, dict):
                loc = cg.get('location', None)

            try:
                ap = int(loc)
            except Exception:
                ap = -1
            if ap < 0 or ap >= num_airports:
                continue

            w = getattr(cg, 'weight', None)
            if w is None and isinstance(cg, dict):
                w = cg.get('weight', 0.0)
            w = float(w or 0.0)

            hard_deadline = getattr(cg, 'hard_deadline', None)
            if hard_deadline is None and isinstance(cg, dict):
                hard_deadline = cg.get('hard_deadline', None)

            urgency = 0.0
            if hard_deadline is not None:
                try:
                    time_left = float(hard_deadline) - current_step
                    urgency = max(0.0, 1.0 / max(time_left, 1.0))
                except Exception:
                    urgency = 0.0

            num_cargo[ap]     += 1.0
            total_weight[ap]  += w
            total_urgency[ap] += float(urgency)

        cargo_node_feats = np.zeros((num_airports, 4), dtype=np.float32)
        for ap in range(num_airports):
            n = num_cargo[ap]
            avg_urg = float(total_urgency[ap] / n) if n > 0.0 else 0.0

            cargo_node_feats[ap, :] = [
                num_cargo[ap],
                total_weight[ap],
                total_urgency[ap],
                avg_urg,
            ]

        return cargo_node_feats

    def _build_route_edge_index(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Build padded edge_index for the route graph from the global state.

        Returns:
            edge_index: [2, max_edges] int64
                edge_index[:, e] = [src, dst] for real edges, or [-1, -1] for padding.
        """
        max_airports = int(getattr(self, 'max_airports', 0) or 0)
        max_routes_per_airport = int(getattr(self, 'max_routes_per_airport', 0) or 0)
        max_edges = max_airports * max_routes_per_airport
        if max_airports <= 0 or max_edges <= 0:
            return np.zeros((2, 0), dtype=np.int64)

        route_map_dict = (state or {}).get('route_map', {}) or {}
        if not route_map_dict:
            # Fill with -1 to denote "no edge"
            edge_index = -1 * np.ones((2, max_edges), dtype=np.int64)
            return edge_index

        # Use the first plane type's DiGraph as the canonical route map
        first_key = sorted(route_map_dict.keys())[0]
        g = route_map_dict[first_key]

        edges = list(g.edges())
        edge_index = -1 * np.ones((2, max_edges), dtype=np.int64)

        for eid, (u, v) in enumerate(edges):
            if eid >= max_edges:
                break
            edge_index[0, eid] = int(u)
            edge_index[1, eid] = int(v)

        return edge_index

    def _build_route_edge_attr(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Build padded edge_attr for the route graph from the global state.

        Returns:
            edge_attr: [max_edges, 3] float32
                For each real edge, features are:
                    [ time, cost, route_available ]
                Remaining rows are zeros.
        """
        max_airports = int(getattr(self, 'max_airports', 0) or 0)
        max_routes_per_airport = int(getattr(self, 'max_routes_per_airport', 0) or 0)
        max_edges = max_airports * max_routes_per_airport
        if max_airports <= 0 or max_edges <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        route_map_dict = (state or {}).get('route_map', {}) or {}
        if not route_map_dict:
            return np.zeros((max_edges, 3), dtype=np.float32)

        first_key = sorted(route_map_dict.keys())[0]
        g = route_map_dict[first_key]

        edges = list(g.edges(data=True))
        edge_attr = np.zeros((max_edges, 3), dtype=np.float32)

        for eid, (u, v, data) in enumerate(edges):
            if eid >= max_edges:
                break

            # Robust dict-like access for attributes
            time_val = data.get('time', 0.0)
            cost_val = data.get('cost', 0.0)
            avail    = data.get('route_available', 1.0)

            try:
                time_val = float(time_val)
            except Exception:
                time_val = 0.0
            try:
                cost_val = float(cost_val)
            except Exception:
                cost_val = 0.0
            try:
                avail = float(avail)
            except Exception:
                avail = 0.0

            edge_attr[eid, :] = [time_val, cost_val, avail]

        return edge_attr


    def flatten_obs(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        import numpy as np

        def pad_to(values, length, fill=-1.0):
            out = np.full((length,), fill, dtype=np.float32)
            if values is None:
                return out
            arr = np.asarray(list(values), dtype=np.float32).ravel()
            n = min(len(arr), length)
            out[:n] = arr[:n]
            return out

        # ---- Build cargo_id -> destination (object-safe) ----
        # ---- Build cargo_id -> destination (object-safe) ----
        # Grab active_cargo from any agent's globalstate
        any_agent = next(iter(obs)) if obs else None
        gs = {}
        active_list = []
        if any_agent is not None:
            gs = (obs[any_agent] or {}).get('globalstate', {}) or {}
            active_list = gs.get('active_cargo', []) or []

        # --- Build graph features for EGAT (shared across agents) ---
        plane_node_feats = self._build_plane_node_feats(obs)
        cargo_node_feats = self._build_cargo_node_feats(active_list)
        edge_index = self._build_route_edge_index(gs)
        edge_attr = self._build_route_edge_attr(gs)


        # CargoObservation objects -> use attribute access; dicts would still work via getattr fallback
        id_to_dest = {}
        for cg in active_list:
            cid = getattr(cg, "id", None)
            dest = getattr(cg, "destination", None)
            if cid is None and isinstance(cg, dict):
                cid = cg.get("id")
                dest = cg.get("destination", dest)
            if cid is not None:
                id_to_dest[cid] = -1 if dest is None else dest

        # ---- Enum value for MOVING (be robust) ----
        PS_MOVING = None
        try:
            PS_MOVING = int(self.env.PlaneState.MOVING)  # if available
        except Exception:
            pass

        flattened_obs: Dict[str, Dict[str, np.ndarray]] = {}

        for aid, aobs in obs.items():
            # raw
            cur_ap   = aobs.get("current_airport", -1)
            routes_l = list(aobs.get("available_routes", []) or [])
            onboard_l= list(aobs.get("cargo_onboard", []) or [])
            at_here_l= list(aobs.get("cargo_at_current_airport", []) or [])
            state    = aobs.get("state", 0)
            cw = float(aobs.get("current_weight", 0.0))
            mw = float(aobs.get("max_weight", 1.0))

            # vectors (DON’T overwrite before copying)
            v_state = np.array([float(state)], dtype=np.float32)
            v_current_airport  = np.array([float(cur_ap)], dtype=np.float32)
            v_available_routes = pad_to(routes_l, self.max_routes_per_airport, fill=-1.0)
            v_onboard_count    = np.array([float(len(onboard_l))], dtype=np.float32)
            v_at_current       = pad_to(at_here_l, self.max_cargo_per_airport, fill=-1.0)
            v_onboard          = pad_to(onboard_l, self.max_cargo_per_plane, fill=-1.0)

            # moving flag
            if PS_MOVING is not None:
                is_moving = float(int(state) == PS_MOVING)
            else:
                is_moving = float(bool(aobs.get("is_in_flight", False)))
            v_is_moving = np.array([is_moving], dtype=np.float32)

            # load frac
            load_frac = cw / max(mw, 1e-6)
            v_load_frac = np.array([float(load_frac)], dtype=np.float32)

            # onboard cargo destinations via id_to_dest map
            onboard_dests = [id_to_dest.get(cid, -1) for cid in onboard_l]
            v_cargo_dests = pad_to(onboard_dests, self.max_cargo_per_plane, fill=-1.0)

            # --- Urgency lists ---
            # Estimate urgency using remaining time to hard deadline if available
            # (access cargo info from globalstate's active_cargo)
            def get_urgency(cid):
                cargo = next((cg for cg in active_list if getattr(cg, "id", None) == cid or (isinstance(cg, dict) and cg.get("id") == cid)), None)
                if cargo is None:
                    return 0.0
                hard_deadline = getattr(cargo, "hard_deadline", None)
                if hard_deadline is None and isinstance(cargo, dict):
                    hard_deadline = cargo.get("hard_deadline")
                if hard_deadline is None:
                    return 0.0
                try:
                    time_left = float(hard_deadline) - float(getattr(self.env, "_elapsed_steps", 0))
                    return max(0.0, 1.0 / max(time_left, 1.0))
                except Exception:
                    return 0.0

            at_here_urgency = [get_urgency(cid) for cid in at_here_l]
            onboard_urgency = [get_urgency(cid) for cid in onboard_l]
            v_at_here_urgency = pad_to(at_here_urgency, self.max_cargo_per_airport, fill=0.0)
            v_onboard_urgency = pad_to(onboard_urgency, self.max_cargo_per_plane, fill=0.0)

            flattened_obs[aid] = {
                "state":                   v_state,
                "current_airport":         v_current_airport,
                "available_routes":        v_available_routes,
                "cargo_destinations":      v_cargo_dests,
                "cargo_at_current_airport":v_at_current,
                "cargo_at_current_airport_urgency": v_at_here_urgency,
                "cargo_onboard":           v_onboard,
                "cargo_onboard_urgency":   v_onboard_urgency,
                # "is_moving":               v_is_moving,
                "load_frac":               v_load_frac,
                "plane_node_feats":        plane_node_feats,
                "cargo_node_feats":        cargo_node_feats,
                "edge_index":              edge_index,
                "edge_attr":               edge_attr,
            }

         # Build shared globalstate by concatenating each agent’s compact vector (use flattened_obs, not outer vars)
        # We pad up to the max number of planes with -1 so globalstate has a fixed size.
        parts = []

        # Max number of planes (slots) we ever want in the centralized state
        max_planes = getattr(self, "num_possible_agents", len(obs))

        # Use a stable ordering: env.possible_agents if available, otherwise current obs keys
        ordered_agents = list(getattr(self.env, "possible_agents", [])) or list(obs.keys())

        # Ensure we have exactly max_planes slots: truncate extra, pad missing with None
        if len(ordered_agents) > max_planes:
            ordered_agents = ordered_agents[:max_planes]
        elif len(ordered_agents) < max_planes:
            ordered_agents = ordered_agents + [None] * (max_planes - len(ordered_agents))

        # Template for an "empty plane" filled with -1s
        empty_plane = {
            "state": np.full((1,), -1.0, dtype=np.float32),
            "current_airport": np.full((1,), -1.0, dtype=np.float32),
            "available_routes": np.full((self.max_routes_per_airport,), -1.0, dtype=np.float32),
            "cargo_destinations": np.full((self.max_cargo_per_plane,), -1.0, dtype=np.float32),
            "cargo_at_current_airport": np.full((self.max_cargo_per_airport,), -1.0, dtype=np.float32),
            "cargo_at_current_airport_urgency": np.full((self.max_cargo_per_airport,), -1.0, dtype=np.float32),
            "cargo_onboard": np.full((self.max_cargo_per_plane,), -1.0, dtype=np.float32),
            "cargo_onboard_urgency": np.full((self.max_cargo_per_plane,), -1.0, dtype=np.float32),
            "load_frac": np.full((1,), -1.0, dtype=np.float32),
        }

        for aid in ordered_agents:
            # If this slot has no real agent in this world, use the empty_plane vector
            fa = flattened_obs.get(aid, empty_plane)
            parts.extend([
                fa["state"],
                fa["current_airport"],
                fa["available_routes"],
                fa["cargo_destinations"],
                fa["cargo_at_current_airport"],
                fa["cargo_at_current_airport_urgency"],
                fa["cargo_onboard"],
                fa["cargo_onboard_urgency"],
                fa["load_frac"],
            ])

        globalstate = np.concatenate(parts, dtype=np.float32)

        # --- NEW: build previous_action vector ---
        if self._last_actions_for_all_agents is None:
            prev_action_vec = -1.0 * np.ones((self.prev_action_dim,), dtype=np.float32)
        else:
            prev_action_vec = np.asarray(self._last_actions_for_all_agents, dtype=np.float32)
            # Safety: ensure correct length; if not, reset to -1
            if prev_action_vec.size != self.prev_action_dim:
                prev_action_vec = -1.0 * np.ones((self.prev_action_dim,), dtype=np.float32)

        for aid in flattened_obs:
            flattened_obs[aid]["globalstate"] = globalstate
            flattened_obs[aid]["previous_action"] = prev_action_vec
            flattened_obs[aid]["action_mask"] = self._create_action_mask(obs[aid]).astype(np.float32)

        for aid in flattened_obs:
            for k, v in flattened_obs[aid].items():
                arr = np.asarray(v, dtype=np.float32)

                if not np.all(np.isfinite(arr)):
                    # Log the bad array
                    with open("bad_obs_log.txt", "a") as f:
                        f.write("\n=== Non-finite observed in flatten_obs ===\n")
                        f.write(f"Agent ID: {aid}\n")
                        f.write(f"Key: {k}\n")
                        f.write(f"Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}\n")
                        f.write(f"Array contents:\n{arr}\n")
                        f.write(f"{'='*50}\n")

                    # **Fix** it before returning to RLlib
                    arr = np.nan_to_num(
                        arr,
                        nan=-1.0,   # our padding/sentinel value
                        posinf=-1.0,
                        neginf=-1.0,
                    )

                # Ensure everything is float32 and store back
                flattened_obs[aid][k] = arr.astype(np.float32, copy=False)

        return flattened_obs

    def _flatten_action_list(self, name: str, values, maxlen: int = None) -> np.ndarray:
        """
        Flatten a per-agent action list (e.g., cargo_to_load / cargo_to_unload) to a fixed 1D array,
        padding with NaNs. Assumes values is None or a list of ints/floats.
        """
        # infer a default max length if you already have a per-field limit cached
        if maxlen is None:
            maxlen = self._action_maxlens.get(name, 0) if hasattr(self, "_action_maxlens") else 0

        return _pad_with_nans(values, maxlen)

    def _compact_per_agent_dim(self) -> int:
        """
        Must match EXACTLY what flatten_obs() concatenates per agent into `globalstate`:
        1 (current_airport)
        + max_routes_per_airport (available_routes)
        + 1 (onboard_count)
        + max_cargo_per_airport (cargo_at_current_airport)
        + 1 (is_moving)
        + 1 (load_frac)
        + max_cargo_per_plane (cargo_destinations)   # include if you appended it to globalstate
        """
        return (
            1
            + self.max_routes_per_airport
            + 1
            + self.max_cargo_per_airport
            + 1
            + 1
            + self.max_cargo_per_plane
        )

    def _central_state_dim_compact(self) -> int:
        # Prefer the env’s possible_agents if available; fall back to a configured value.
        n_agents = len(getattr(self.env, "possible_agents", [])) or getattr(self, "num_possible_agents", 0)
        return n_agents * self._compact_per_agent_dim()

    def _decode_action_from_flat(self, agent_id, flat_action):
        """
        Decode flat action for `agent_id`. Map per-slot logits -> actual cargo ids
        from the last raw observation (cargo_at_current_airport / cargo_onboard).
        Destination decoding:
          - choice 0 -> destination = 0
          - choice 1 -> first entry of available_routes
          - choice 2 -> second entry of available_routes, etc.
        """
        def _selected_indices(arr):
            if arr is None:
                return []
            if isinstance(arr, np.ndarray):
                arr = arr.tolist()
            return [i for i, x in enumerate(arr) if int(x) == 1]

        # Raw lists from last observation (fall back to empty lists)
        raw = getattr(self, "_last_raw_obs", {}).get(agent_id, {}) or {}
        at_here_list = list(raw.get("cargo_at_current_airport", []) or [])
        onboard_list = list(raw.get("cargo_onboard", []) or [])
        routes_raw = list(raw.get("available_routes", []) or [])

        load_slots = _selected_indices(flat_action.get("cargo_to_load"))
        unload_slots = _selected_indices(flat_action.get("cargo_to_unload"))

        # Map slot indices to actual cargo ids, guard out-of-range and padding values (-1)
        def _slots_to_ids(slots, source_list):
            ids = []
            for idx in slots:
                try:
                    if idx is None:
                        continue
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < len(source_list):
                    cid = source_list[i]
                    # skip padding markers
                    if cid is None or (isinstance(cid, (int, float)) and int(cid) == -1):
                        continue
                    ids.append(cid)
            return ids

        cargo_to_load = _slots_to_ids(load_slots, at_here_list)
        cargo_to_unload = _slots_to_ids(unload_slots, onboard_list)

        try:
            dest_choice = int(flat_action.get("destination", 0))
        except Exception:
            dest_choice = 0

        if dest_choice == 0:
            destination = 0
        else:
            idx = dest_choice - 1
            if 0 <= idx < len(routes_raw):
                destination = int(routes_raw[idx])
            else:
                # Fallback to noop destination if index out of range
                destination = 0

        return {
            "cargo_to_load":   cargo_to_load,
            "cargo_to_unload": cargo_to_unload,
            "destination":     int(destination),
            "priority":        int(1),
        }
    
    def _create_action_mask(self, aobs) -> np.ndarray:
        """
        Flat mask layout (float32):
        [ load (2*max_cargo_per_airport),
            unload (2*max_cargo_per_plane),
            destination (max_routes_per_airport + 1) ]

        If plane state != 3 (READY_FOR_TAKEOFF), enforce a pure noop:
        - all load/unload bits must be 0
        - destination must be 0 ("stay")
        Otherwise (state == 3), use the normal mask based on available cargo/routes.
        """
        import numpy as np

        # --- sizes that MUST match your action_space ---
        k_load   = int(self.max_cargo_per_airport)
        k_unload = int(self.max_cargo_per_plane)
        dest_len = int(self.max_routes_per_airport) + 1

        # --- plane state check ---
        try:
            state = int(aobs.get("state", 0))
        except Exception:
            state = 0

        # NOOP mask for all non-READY_FOR_TAKEOFF states
        if state != 3:
            # load/unload pairs: [1,0] everywhere (can only choose 0)
            load_pairs   = np.zeros((k_load, 2), dtype=np.float32)
            unload_pairs = np.zeros((k_unload, 2), dtype=np.float32)
            load_pairs[:, 0]   = 1.0
            unload_pairs[:, 0] = 1.0
            load_mask_pairs   = load_pairs.reshape(-1)
            unload_mask_pairs = unload_pairs.reshape(-1)

            # destination: only index 0 ("stay") allowed
            dest_mask = np.zeros((dest_len,), dtype=np.float32)
            dest_mask[0] = 1.0

            mask = np.concatenate([load_mask_pairs, unload_mask_pairs, dest_mask], axis=0).astype(np.float32)
            return mask

        # ---------------- READY_FOR_TAKEOFF (state == 3) ----------------
        # normal masking logic (unchanged from before)

        # --- pull raw fields (may be padded with -1) ---
        at_here    = list(aobs.get("cargo_at_current_airport", []) or [])
        onboard    = list(aobs.get("cargo_onboard", []) or [])
        routes_raw = list(aobs.get("available_routes", []) or [])

        # count only real (non -1 / None) entries
        def _valid_count(arr):
            c = 0
            for x in arr:
                try:
                    xi = int(x)
                except Exception:
                    xi = None
                if xi is None or xi == -1:
                    continue
                c += 1
            return c

        n_load_ok   = min(_valid_count(at_here),   k_load)
        n_unload_ok = min(_valid_count(onboard),   k_unload)
        n_routes    = _valid_count(routes_raw)

        # helper: first m binary slots allow "1" (as [1,1]); others forced to 0 (as [1,0])
        def _pairs_first_m_are_1(m: int, k: int) -> np.ndarray:
            out = np.zeros((k, 2), dtype=np.float32)
            out[:, 0] = 1.0             # always allow choosing 0
            if m > 0:
                out[:min(m, k), 1] = 1.0
            return out.reshape(-1)       # length 2*k

        load_mask_pairs   = _pairs_first_m_are_1(n_load_ok,   k_load)
        unload_mask_pairs = _pairs_first_m_are_1(n_unload_ok, k_unload)

        # destination head
        dest_mask = np.zeros((dest_len,), dtype=np.float32)
        dest_mask[0] = 1.0
        upto = min(n_routes, dest_len - 1)
        if upto > 0:
            dest_mask[1:1+upto] = 1.0

        mask = np.concatenate([load_mask_pairs, unload_mask_pairs, dest_mask], axis=0).astype(np.float32)
        return mask

    def _encode_single_action_for_history(self, flat_action) -> np.ndarray:
        """
        Encode one agent's action into a 1D vector of length mask_dim, matching the
        logits/mask layout:

        [ load(2 * max_cargo_per_airport),
          unload(2 * max_cargo_per_plane),
          destination(max_routes_per_airport + 1) ]

        For load/unload, we one-hot each binary decision (0 -> [1,0], 1 -> [0,1]).
        For destination, we one-hot over (max_routes_per_airport + 1) choices.

        Returned values are in {0, 1}.
        """
        import numpy as np

        k_load   = int(self.max_cargo_per_airport)
        k_unload = int(self.max_cargo_per_plane)
        dest_len = int(self.max_routes_per_airport) + 1

        # --- load head ---
        load_raw = flat_action.get("cargo_to_load", None)
        if load_raw is None:
            load_raw = np.zeros((k_load,), dtype=np.int64)
        load_raw = np.asarray(load_raw, dtype=np.int64).ravel()
        if load_raw.size < k_load:
            pad = np.zeros((k_load - load_raw.size,), dtype=np.int64)
            load_raw = np.concatenate([load_raw, pad], axis=0)
        else:
            load_raw = load_raw[:k_load]

        load_pairs = np.zeros((k_load, 2), dtype=np.float32)
        for i, val in enumerate(load_raw):
            val = int(val)
            if val == 0:
                load_pairs[i, 0] = 1.0
            else:
                load_pairs[i, 1] = 1.0
        load_vec = load_pairs.reshape(-1)  # length 2*k_load

        # --- unload head ---
        unload_raw = flat_action.get("cargo_to_unload", None)
        if unload_raw is None:
            unload_raw = np.zeros((k_unload,), dtype=np.int64)
        unload_raw = np.asarray(unload_raw, dtype=np.int64).ravel()
        if unload_raw.size < k_unload:
            pad = np.zeros((k_unload - unload_raw.size,), dtype=np.int64)
            unload_raw = np.concatenate([unload_raw, pad], axis=0)
        else:
            unload_raw = unload_raw[:k_unload]

        unload_pairs = np.zeros((k_unload, 2), dtype=np.float32)
        for i, val in enumerate(unload_raw):
            val = int(val)
            if val == 0:
                unload_pairs[i, 0] = 1.0
            else:
                unload_pairs[i, 1] = 1.0
        unload_vec = unload_pairs.reshape(-1)  # length 2*k_unload

        # --- destination head ---
        try:
            dest_choice = int(flat_action.get("destination", 0))
        except Exception:
            dest_choice = 0
        if dest_choice < 0 or dest_choice >= dest_len:
            dest_choice = 0
        dest_vec = np.zeros((dest_len,), dtype=np.float32)
        dest_vec[dest_choice] = 1.0

        # Concatenate into [load, unload, dest]
        return np.concatenate([load_vec, unload_vec, dest_vec], axis=0).astype(np.float32)

    def _update_previous_actions_vector(self, action_dict: Dict[str, Any]) -> None:
        """
        Build the big previous_action vector by concatenating encoded actions
        for all plane slots (up to num_possible_agents). Missing agents / slots
        are filled with -1.0.
        """
        import numpy as np

        max_planes = getattr(self, "num_possible_agents", len(action_dict))
        ordered_agents = list(getattr(self.env, "possible_agents", [])) or list(action_dict.keys())

        if len(ordered_agents) > max_planes:
            ordered_agents = ordered_agents[:max_planes]
        elif len(ordered_agents) < max_planes:
            ordered_agents = ordered_agents + [None] * (max_planes - len(ordered_agents))

        empty = -1.0 * np.ones((self.mask_dim,), dtype=np.float32)
        parts = []

        for aid in ordered_agents:
            if aid is None or aid not in action_dict:
                parts.append(empty)
            else:
                encoded = self._encode_single_action_for_history(action_dict[aid])
                parts.append(encoded)

        self._last_actions_for_all_agents = np.concatenate(parts, axis=0).astype(np.float32)








