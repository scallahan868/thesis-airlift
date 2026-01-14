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

        self.env = env

        # Grab world generator for sizing hints (fall back to safe constants)
        wg = getattr(self.env, "world_generator", None)

        # Fixed, numeric limits (store as attributes — not callables)
        self.max_cargo_per_plane = int(7)
        self.max_cargo_per_airport = int(10)
        self.max_routes_per_airport = int(3)
        self.max_agents = int(6)
        self.max_cargo_per_episode = int(15)
        self.max_airports = int(6)
        self.max_edges = int(self.max_airports * self.max_routes_per_airport)

        # For action list padding (used by debug scaffolding / future action adapters)
        self._action_maxlens = {
            "cargo_to_load": self.max_cargo_per_plane,
            "cargo_to_unload": self.max_cargo_per_plane,
        }

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
        self.prev_action_dim = int(self.max_agents) * int(self.prev_action_per_plane_dim)

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
        # self._debug_check_obs(obs)
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
        # self._debug_check_obs(obs)

        return obs, rewards, terminations, truncations, infos

    def _debug_check_obs(self, obs):
        """
        Robust observer-space checker for multi-agent observations.

        - Calls self.observation_space(agent_id) (the method) rather than using it
        as an attribute.
        - Handles missing spaces, non-dict observations, and prints helpful diagnostics.
        - Raises AssertionError on failure so existing control flow remains unchanged.
        """
        import numpy as np

        if obs is None:
            raise AssertionError("obs is None")

        # Expect a mapping of agent_id -> per-agent observation
        if not hasattr(obs, "items"):
            raise AssertionError(f"obs must be a mapping of agent_id->obs, got {type(obs)}")

        for aid, aobs in obs.items():
            # Obtain the per-agent observation space by calling the method
            try:
                space = self.observation_space(aid)
            except Exception as e:
                raise AssertionError(f"failed to obtain observation space for agent {aid}: {e}")

            if not hasattr(space, "contains"):
                raise AssertionError(f"observation space for agent {aid} has no 'contains' method ({type(space)})")

            # If the space check fails, give detailed diagnostics
            if not space.contains(aobs):
                print(f"OBS SPACE VIOLATION for agent {aid}: expected {space}")

                # If the space is a Gym Dict, inspect each sub-space
                subspaces = getattr(space, "spaces", None)
                if subspaces and isinstance(aobs, dict):
                    for k, sp in subspaces.items():
                        v = aobs.get(k, None)
                        if v is None:
                            print(f"  MISSING KEY: {k}")
                            continue
                        arr = np.asarray(v)
                        if np.issubdtype(arr.dtype, np.number):
                            try:
                                vmin = np.nanmin(arr)
                                vmax = np.nanmax(arr)
                            except Exception:
                                vmin = vmax = None
                        else:
                            vmin = vmax = None
                        print(f"  {k}: got shape={arr.shape} dtype={arr.dtype} min={vmin} max={vmax} expected={sp}")
                    extra = set(aobs.keys()) - set(subspaces.keys())
                    if extra:
                        print("  EXTRA KEYS:", extra)
                else:
                    # Non-dict obs or non-Dict space: print a short representation
                    try:
                        arr = np.asarray(aobs)
                        print(f"  value shape={arr.shape} dtype={arr.dtype}")
                    except Exception:
                        print(f"  value (non-array): {type(aobs)}")

                raise AssertionError("obs not in space")

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

            # --- centralized critic input (shared vector you attach to each agent) ---
            "globalstate":              Box(-np.inf, np.inf, shape=((4+self.max_routes_per_airport+3*self.max_cargo_per_plane+2*self.max_cargo_per_airport)*self.max_agents,), dtype=np.float32),

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
            
            # --- graph inputs for EGAT/GAT (per-agent plane_type graph) ---
            # node_features: [max_airports, 7] = [num_aircraft_any_type, num_packages, cumulative_urgency, airport_x, airport_y, degree, is_destination]
            "node_features": Box(-np.inf, np.inf, shape=(self.max_airports, 7), dtype=np.float32),
            # edge_index: [2, max_edges] padded with -1 (PyG-style)
            "edge_index":    Box(-1.0, np.inf, shape=(2, self.max_edges), dtype=np.float32),
            # edge_features: [max_edges, 2] = [distance, is_available]
            "edge_features": Box(-np.inf, np.inf, shape=(self.max_edges, 2), dtype=np.float32),
            # current airport row-index into node_features
            "current_airport_idx": Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            # plane_type included for debugging / optional conditioning
            "plane_type": Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),

            "action_mask": Box(low=0.0, high=1.0, shape=(mask_dim,), dtype=np.float32),
                    })

        self._observation_space_cache[agent_id] = space
        return space

    def close(self):
        return self.env.close()


    # ----------------------------
    # Graph feature extraction
    # ----------------------------
    def _select_route_graph(self, aobs: dict, globalstate: dict):
        """Select the correct NetworkX DiGraph for this agent's plane_type."""
        route_map = (globalstate or {}).get("route_map", {}) or {}
        plane_type = (aobs or {}).get("plane_type", 0)
        try:
            plane_type = int(plane_type)
        except Exception:
            plane_type = 0

        G = None
        if isinstance(route_map, dict):
            G = route_map.get(plane_type, None)

        # fall back: if route_map itself is a graph
        if G is None:
            if hasattr(route_map, "nodes") and hasattr(route_map, "edges"):
                G = route_map
            else:
                # last resort: pick any available graph (first value)
                try:
                    G = next(iter(route_map.values()))
                except Exception:
                    G = None

        return G, plane_type

    def _get_destination_set(self, globalstate: dict) -> set[int]:
        """Best-effort: return set of destination airport node-IDs."""
        gsd = globalstate or {}

        # 1) Explicit destination lists/sets
        for key in (
            "destination_airports",
            "destinations",
            "destination_nodes",
            "sink_airports",
            "sink_nodes",
        ):
            if key in gsd:
                try:
                    return set(int(x) for x in (gsd.get(key) or []) if x is not None)
                except Exception:
                    pass

        # 2) Scenario info object(s) may carry destination metadata
        scenario_info = gsd.get("scenario_info", None)
        if scenario_info is not None:
            try:
                if isinstance(scenario_info, (list, tuple)) and len(scenario_info) > 0:
                    scenario_info = scenario_info[0]
                for attr in (
                    "destination_airports",
                    "destinations",
                    "destination_nodes",
                    "sink_airports",
                    "sink_nodes",
                ):
                    if hasattr(scenario_info, attr):
                        vals = getattr(scenario_info, attr)
                        try:
                            return set(int(x) for x in (vals or []) if x is not None)
                        except Exception:
                            pass
            except Exception:
                pass

        # 3) Fallback: infer destinations from cargo objects (active + newly spawned)
        dest_set: set[int] = set()
        for cargo_list_key in ("active_cargo", "event_new_cargo"):
            cargo_list = gsd.get(cargo_list_key, []) or []
            for cargo in cargo_list:
                dest = getattr(cargo, "destination", None)
                if isinstance(cargo, dict):
                    dest = cargo.get("destination", dest)
                try:
                    if dest is not None:
                        dest_set.add(int(dest))
                except Exception:
                    pass
        return dest_set



    def get_node_features(
        self,
        obs: dict,
        aid: str,
        aobs: dict,
        globalstate: dict,
        G,
        plane_type: int,
        node_list,
        node_id_to_row,
        get_urgency_fn,
    ):
        """
        Node features per airport node (row order = node_list / node_id_to_row):

          0) num_aircraft_any_type: Number of aircraft currently at airport (any plane type)
          1) num_packages:          Number of packages currently at airport
          2) cumulative_urgency:    Sum of urgency for packages currently at airport
          3) airport_x:             X position from graph node attr 'pos' (fallback 0)
          4) airport_y:             Y position from graph node attr 'pos' (fallback 0)
          5) degree:                Airport degree (in/out are the same in your graphs; we use total degree)
          6) is_destination:        1 if airport is a destination, else 0

        Output shape: (self.max_airports, 7) padded with zeros.
        """
        feats = np.zeros((self.max_airports, 7), dtype=np.float32)
        if G is None or not node_list:
            return feats

        # --- 0) aircraft counts at each airport (ANY plane type) ---
        aircraft_counts = {int(n): 0 for n in node_list}
        for _, other_obs in (obs or {}).items():
            cur = (other_obs or {}).get("current_airport", None)
            try:
                cur = int(cur)
            except Exception:
                continue
            if cur in aircraft_counts:
                aircraft_counts[cur] += 1

        # --- 1-2) packages + urgency at each airport (from globalstate active_cargo) ---
        pkg_counts = {int(n): 0 for n in node_list}
        urg_sums = {int(n): 0.0 for n in node_list}
        active_cargo = (globalstate or {}).get("active_cargo", []) or []
        for cargo in active_cargo:
            loc = getattr(cargo, "location", None)
            cid = getattr(cargo, "id", None)
            if isinstance(cargo, dict):
                loc = cargo.get("location", loc)
                cid = cargo.get("id", cid)
            try:
                loc = int(loc)
            except Exception:
                continue
            if loc in pkg_counts:
                pkg_counts[loc] += 1
                try:
                    urg_sums[loc] += float(get_urgency_fn(cid))
                except Exception:
                    pass

        # destination set (robust fallbacks)
        dest_set = self._get_destination_set(globalstate)

        for n in node_list[: self.max_airports]:
            nid = int(n)
            i = node_id_to_row.get(nid, None)
            if i is None or i >= self.max_airports:
                continue

            # --- 3-4) position ---
            x = 0.0
            y = 0.0
            try:
                nd = G.nodes[nid]
                pos = None
                if isinstance(nd, dict):
                    pos = nd.get("pos", None)
                    # sometimes stored as {'x':..., 'y':...}
                    if pos is None and ("x" in nd or "y" in nd):
                        x = float(nd.get("x", 0.0))
                        y = float(nd.get("y", 0.0))
                if pos is not None and isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x = float(pos[0])
                    y = float(pos[1])
            except Exception:
                pass

            # --- 5) degree ---
            # In your route graphs, in/out are symmetric; we use total degree for safety.
            try:
                deg = float(G.degree(nid))
            except Exception:
                deg = 0.0

            # --- 6) destination flag ---
            is_dest = 1.0 if nid in dest_set else 0.0

            feats[i, 0] = float(aircraft_counts.get(nid, 0))
            feats[i, 1] = float(pkg_counts.get(nid, 0))
            feats[i, 2] = float(urg_sums.get(nid, 0.0))
            feats[i, 3] = float(x)
            feats[i, 4] = float(y)
            feats[i, 5] = float(deg)
            feats[i, 6] = float(is_dest)

        return feats

    def get_edge_features(self, aobs: dict, globalstate: dict, G, node_id_to_row):
        """
        Edge features: [distance(cost), is_available]
        Output:
          edge_index shape (2, self.max_edges) padded with -1
          edge_features shape (self.max_edges, 2) padded with 0
        """
        edge_features = np.zeros((self.max_edges, 2), dtype=np.float32)
        edge_index = -1.0 * np.ones((2, self.max_edges), dtype=np.float32)

        if G is None:
            return edge_index, edge_features

        e = 0
        for u, v, data in G.edges(data=True):
            if e >= self.max_edges:
                break
            try:
                ur = node_id_to_row[int(u)]
                vr = node_id_to_row[int(v)]
            except Exception:
                continue

            edge_index[0, e] = float(ur)
            edge_index[1, e] = float(vr)

            dist = 0.0
            for k in ("cost", "distance", "dist", "time", "length"):
                if isinstance(data, dict) and k in data:
                    try:
                        dist = float(data[k])
                        break
                    except Exception:
                        pass

            avail = None
            if isinstance(data, dict):
                if "route_available" in data:
                    avail = data.get("route_available")
                elif "is_available" in data:
                    avail = data.get("is_available")
            if avail is None:
                avail = True

            edge_features[e, 0] = float(dist)
            edge_features[e, 1] = 1.0 if bool(avail) else 0.0
            e += 1

        return edge_index, edge_features

    def get_graph_data(self, aobs: dict, globalstate: dict):
        """
        Select the correct route graph for this agent and return
        graph + indexing metadata needed by node/edge feature builders.

        Returns:
            G               : nx.DiGraph or None
            plane_type      : int
            node_list       : List[int]
            node_id_to_row  : Dict[int, int]
        """
        # 1) Select graph based on plane_type
        G, plane_type = self._select_route_graph(aobs, globalstate)

        if G is None:
            return None, plane_type, [], {}

        # 2) Stable node ordering
        node_list = list(G.nodes())

        # 3) Node-id → row index mapping
        node_id_to_row = {
            int(nid): i for i, nid in enumerate(node_list)
            if i < self.max_airports
        }

        return G, plane_type, node_list, node_id_to_row

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
        # Grab active_cargo from any agent's globalstate
        any_agent = next(iter(obs)) if obs else None
        active_list = []
        if any_agent is not None:
            gs = (obs[any_agent] or {}).get("globalstate", {}) or {}
            active_list = gs.get("active_cargo", []) or []

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


            # --- Graph features for EGAT/GAT (select graph by plane_type) ---
            gsd = (aobs.get("globalstate", {}) or {})
            G, plane_type, node_list, node_id_to_row = self.get_graph_data(aobs, gsd)

            # current airport row-index (defaults to 0 if missing)
            try:
                cur_ap_int = int(cur_ap)
            except Exception:
                cur_ap_int = -1
            cur_idx = float(node_id_to_row.get(cur_ap_int, 0))

            v_plane_type = np.array([float(plane_type)], dtype=np.float32)
            v_current_airport_idx = np.array([cur_idx], dtype=np.float32)

            v_node_features = self.get_node_features(
                obs=obs,
                aid=aid,
                aobs=aobs,
                globalstate=gsd,
                G=G,
                plane_type=plane_type,
                node_list=node_list,
                node_id_to_row=node_id_to_row,
                get_urgency_fn=get_urgency,
            )

            v_edge_index, v_edge_features = self.get_edge_features(
                aobs=aobs,
                globalstate=gsd,
                G=G,
                node_id_to_row=node_id_to_row,
            )

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

                # graph inputs
                "node_features":          v_node_features,
                "edge_index":             v_edge_index,
                "edge_features":          v_edge_features,
                "current_airport_idx":    v_current_airport_idx,
                "plane_type":             v_plane_type,
            }

         # Build shared globalstate by concatenating each agent’s compact vector (use flattened_obs, not outer vars)
        # We pad up to the max number of planes with -1 so globalstate has a fixed size.
        parts = []

        # Max number of planes (slots) we ever want in the centralized state
        max_planes = self.max_agents

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
            "plane_type": np.full((1,), -1.0, dtype=np.float32),
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
                fa["plane_type"],
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

        max_planes = self.max_agents
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
















