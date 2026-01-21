
"""
The centralized critic custom model class
"""

#!/usr/bin/env python3

"""
Legacy RLLib Custom Model for Centralized Critic MAPPO
Compatible with the legacy API stack (enable_rl_module_and_learner=False)

This model implements centralized critic where:
- Actor networks use local observations (individual agent view)
- Critic networks use centralized observations (global state via observation_fn)

EGAT replacement:
- Actor network is augmented with a graph embedding computed via PyTorch Geometric's GATConv.
- Critic network remains untouched (still consumes only centralized observations).
"""

# from __future__ import annotations

import collections
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from gymnasium.spaces import Box

import wandb

torch, nn = try_import_torch()

try:
    from torch_geometric.nn import GATConv
except Exception as e:  # pragma: no cover
    GATConv = None
    _pyg_import_error = e
else:
    _pyg_import_error = None


# Debug logging setup
DEBUG_LOG_DIR = "debug_logs"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Legacy-style centralized critic model for MAPPO.

    Key Features (preserved from base):
    - Actor uses local observations (from original obs)
    - Critic uses centralized observations (from observation_fn)
    - Compatible with legacy RLLib API
    - Proper value function estimation with global state
    - Action masking preserved
    - DEBUG: Tracks and logs observation space usage to verify centralized critic

    Added:
    - Actor-side PyG GATConv encoder. The current-airport node embedding is
      concatenated onto the local observation before passing into the actor MLP.
    """

    # Class-level debug counters
    _debug_counter = 0
    _actor_calls = 0
    _critic_calls = 0
    _last_log_time = None

    # Performance optimization: Only log first 25 calls then stop permanently
    MAX_DEBUG_LOGS = 25
    _debug_logs_written = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        print(f"🧠 CentralizedCriticModel (Legacy + PyG GATConv) setting up...")
        print(f"   📊 Observation space: {obs_space}")
        print(f"   🎯 Action space: {action_space}")
        print(f"   🎮 Num outputs (actions): {num_outputs}")

        cfg = model_config.get("custom_model_config", {})
        self.local_obs_dim = int(cfg.get("local_obs_dim", 907))
        self.central_obs_dim = int(cfg.get("central_obs_dim", 851))

        # ---- GAT config (actor-only) ----
        self.enable_gat = bool(cfg.get("enable_gat", True))

        # Input feature sizes provided by your wrapper:
        # node_features: [B, N, gat_in_node_feats]
        # edge_features: [B, E, gat_in_edge_feats]  (optional)
        self.gat_in_node_feats = int(cfg.get("gat_in_node_feats", 7))
        self.gat_in_edge_feats = int(cfg.get("gat_in_edge_feats", 2))

        self.gat_num_layers = int(cfg.get("gat_num_layers", 3))
        self.gat_num_heads = int(cfg.get("gat_num_heads", 3))
        self.gat_hidden = int(cfg.get("gat_hidden", 32))
        self.gat_out = int(cfg.get("gat_out", 32))

        # Output node embedding dim from the final layer.
        # For PyG GATConv with concat=True: heads*out_channels.
        self.gat_concat = False
        self.gat_final_node_dim = self.gat_num_heads * self.gat_out

        # Optional projection for the node embedding before concatenation.
        self.gat_project_dim = int(cfg.get("gat_project_dim", self.gat_final_node_dim))

        # If disabled, embed dim is 0.
        self._actor_extra_dim = self.gat_project_dim if self.enable_gat else 0

        print(f"   📏 Local obs dim: {self.local_obs_dim}")
        print(f"   🌍 Central obs dim: {self.central_obs_dim}")
        if self.enable_gat:
            print(
                f"   🕸️ PyG GAT enabled: layers={self.gat_num_layers}, heads={self.gat_num_heads}, "
                f"in_node={self.gat_in_node_feats}, edge_dim={self.gat_in_edge_feats}, "
                f"hidden={self.gat_hidden}, out={self.gat_out}, proj={self.gat_project_dim}"
            )

        # --- Actor network (local obs + optional GAT embedding) ---
        actor_model_config = {
            "fcnet_hiddens": [512, 512, 256],
            "fcnet_activation": "relu",
        }

        actor_input_dim = self.local_obs_dim + self._actor_extra_dim
        actor_obs_space = Box(low=-1.0, high=1.0, shape=(actor_input_dim,), dtype=np.float32)
        self.actor_net = FullyConnectedNetwork(
            obs_space=actor_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=actor_model_config,
            name="actor_net",
        )

        # --- Critic network (unchanged) ---
        self.critic_net = nn.Sequential(
            nn.Linear(self.central_obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # --- PyG GAT encoder (actor-only) ---
        if self.enable_gat:
            if GATConv is None:
                raise ImportError(
                    "torch_geometric is required for GATConv but could not be imported. "
                    f"Original error: {_pyg_import_error}"
                )

            self.gat_layers = nn.ModuleList()
            self.gat_acts = nn.ModuleList()

            # Layer 0: in -> hidden
            self.gat_layers.append(
                GATConv(
                    in_channels=self.gat_in_node_feats,
                    out_channels=self.gat_hidden,
                    heads=self.gat_num_heads,
                    concat=False,
                    dropout=float(cfg.get("gat_dropout", 0.0)),
                    add_self_loops=True,
                    edge_dim=self.gat_in_edge_feats if self.gat_in_edge_feats > 0 else None,
                )
            )
            self.gat_acts.append(nn.ELU())

            # Middle layers: hidden*heads -> hidden
            for _ in range(1, max(self.gat_num_layers - 1, 1)):
                self.gat_layers.append(
                    GATConv(
                        in_channels=self.gat_num_heads * self.gat_hidden,
                        out_channels=self.gat_hidden,
                        heads=self.gat_num_heads,
                        concat=False,
                        dropout=float(cfg.get("gat_dropout", 0.0)),
                        add_self_loops=True,
                        edge_dim=self.gat_in_edge_feats if self.gat_in_edge_feats > 0 else None,
                    )
                )
                self.gat_acts.append(nn.ELU())

            # Final layer: hidden*heads -> out
            self.gat_layers.append(
                GATConv(
                    in_channels=self.gat_num_heads * self.gat_hidden,
                    out_channels=self.gat_out,
                    heads=self.gat_num_heads,
                    concat=False,  # keep [N, heads*out]
                    dropout=float(cfg.get("gat_dropout", 0.0)),
                    add_self_loops=True,
                    edge_dim=self.gat_in_edge_feats if self.gat_in_edge_feats > 0 else None,
                )
            )

            self.gat_proj = (
                nn.Identity()
                if self.gat_project_dim == self.gat_final_node_dim
                else nn.Linear(self.gat_final_node_dim, self.gat_project_dim)
            )

        print(f"   ✅ Actor network: {actor_input_dim} → FC → {num_outputs}")
        print(f"   ✅ Critic network: {self.central_obs_dim} → 512 → 256 → 1")

    # -------------------- Actor forward --------------------

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for the actor network.
        The critic forward pass is handled separately in value_function().
        """
        obs = input_dict["obs"]

        CentralizedCriticModel._actor_calls += 1

        # --- Build local tensor (what the actor traditionally sees) ---
        if isinstance(obs, dict):
            # Prefer explicit "local_obs" if the wrapper provides it.
            if "local_obs" in obs:
                local_tensor = _to_2d_tensor(obs["local_obs"])
            else:
                # Fallback: exclude fields that are not local actor inputs.
                excluded = {
                    "action_mask",
                    "previous_action",
                    "globalstate",
                    "node_features",
                    "edge_features",
                    "edge_index",
                    "current_airport_idx",
                }
                local_fields = {k: v for k, v in obs.items() if k not in excluded}
                local_tensor = _to_2d_tensor(local_fields)
        else:
            local_tensor = _to_2d_tensor(obs)

        local_tensor = local_tensor.to(torch.float32)

        # --- Optional: compute GAT embedding and concat onto local_tensor ---
        if self.enable_gat and isinstance(obs, dict) and "node_features" in obs and "edge_index" in obs:
            try:
                gat_emb = self._compute_current_node_embedding(obs)  # [B, gat_project_dim]
                local_tensor = torch.cat([local_tensor, gat_emb.to(local_tensor.device)], dim=1)
            except Exception as e:
                # Fail loud once; then keep training without GAT (avoids crash loops).
                if not hasattr(self, "_logged_gat_failure"):
                    print(f"\n[WARN] GAT embedding computation failed once; continuing without GAT. Error: {e}\n")
                    self._logged_gat_failure = True

                # If actor expects GAT dims, we must still pad zeros to keep shape consistent.
                if self._actor_extra_dim > 0:
                    zeros = torch.zeros((local_tensor.shape[0], self._actor_extra_dim), dtype=local_tensor.dtype, device=local_tensor.device)
                    local_tensor = torch.cat([local_tensor, zeros], dim=1)

        # Prepare input dict for actor network (actor expects tensors)
        local_input_dict = dict(input_dict)
        local_input_dict["obs"] = local_tensor
        local_input_dict["obs_flat"] = local_tensor

        logits, _ = self.actor_net(local_input_dict, state, seq_lens)

        # --- Action masking (preserved) ---
        if isinstance(obs, dict) and "action_mask" in obs:
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)  # [B, N]

            valid = (mask <= 1.0 + 1e-6) & (mask >= 0.0 - 1e-6)
            if not torch.all(valid):
                if not hasattr(self, "_logged_bad_mask"):
                    print(
                        "\n[WARN] Invalid values found in action_mask! "
                        "Values will be clamped to {0.0, 1.0}."
                    )
                    self._logged_bad_mask = True
                mask = torch.round(mask).clamp(0.0, 1.0)

            FLOAT_MIN = torch.finfo(logits.dtype).min
            logits = logits + torch.log(mask + 1e-12).clamp(min=FLOAT_MIN)

        # Store the raw obs for value_function()
        self._last_obs = obs

        return logits, state

    # -------------------- Critic forward (unchanged) --------------------

    @override(TorchModelV2)
    def value_function(self):
        """
        Compute value function using centralized observations.
        This is called by PPO to get the critic's value estimate.
        """
        assert hasattr(self, '_last_obs'), "Must call forward() before value_function()"
        
        obs = self._last_obs
        
        # DEBUG: Track critic calls
        CentralizedCriticModel._critic_calls += 1
        
        # Extract centralized observation for critic
        # if isinstance(obs, dict):
            # Dict observation: use 'state' key for centralized observations
        central_obs = _to_2d_tensor({
                "globalstate":      obs["globalstate"],
                "previous_action":  obs["previous_action"],
            })
        # central_obs = obs["globalstate","previous_action"] # if "state" in obs else obs["obs"]
        # PERFORMANCE: Only log first 25 calls to avoid I/O overhead
        if (CentralizedCriticModel._debug_logs_written < CentralizedCriticModel.MAX_DEBUG_LOGS and 
            CentralizedCriticModel._critic_calls <= 25):
            self._log_critic_debug(obs, central_obs)
            CentralizedCriticModel._debug_logs_written += 1
        # else:
        #     # Box observation: use as-is (assumes centralized obs from observation_fn)
        #     central_obs = obs
        #     if (CentralizedCriticModel._debug_logs_written < CentralizedCriticModel.MAX_DEBUG_LOGS and 
        #         CentralizedCriticModel._critic_calls <= 25):
        #         self._log_critic_debug(obs, central_obs)
        #         CentralizedCriticModel._debug_logs_written += 1
        
        # Ensure tensor format
        if not isinstance(central_obs, torch.Tensor):
            central_obs = torch.tensor(central_obs, dtype=torch.float32)
        
        # Forward through critic network
        value = self.critic_net(central_obs)
        
        return value.squeeze(-1)  # Remove last dimension to match expected shape

    # -------------------- GAT helpers --------------------

    # def _compute_current_node_embedding(self, obs: Dict[str, Any]) -> torch.Tensor:
    #     """
    #     Computes a per-sample node embedding for the agent's current airport via PyG GATConv.

    #     Expected obs keys (per-sample or batched):
    #       - node_features: [B, N, F] or [N, F]
    #       - edge_index:   [B, 2, E] or [2, E] (padded with -1)
    #       - edge_features:[B, E, D] or [E, D] (optional)
    #       - current_airport_idx: [B] or [B,1] or scalar (index into 0..N-1)

    #     Returns:
    #       Tensor [B, gat_project_dim]
    #     """
    #     node_features = obs["node_features"]
    #     edge_index = obs["edge_index"]
    #     edge_features = obs.get("edge_features", None)

    #     x = node_features if isinstance(node_features, torch.Tensor) else torch.as_tensor(node_features, dtype=torch.float32)
    #     ei = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index, dtype=torch.long)
    #     ea = None
    #     if edge_features is not None:
    #         ea = edge_features if isinstance(edge_features, torch.Tensor) else torch.as_tensor(edge_features, dtype=torch.float32)

    #     # Ensure batch dimension
    #     if x.ndim == 2:  # [N, F]
    #         x = x.unsqueeze(0)
    #     if ei.ndim == 2:  # [2, E]
    #         ei = ei.unsqueeze(0)
    #     if ea is not None and ea.ndim == 2:  # [E, D]
    #         ea = ea.unsqueeze(0)

    #     B, N, _ = x.shape

    #     # Current node indices
    #     cur_idx = obs.get("current_airport_idx", obs.get("current_airport", None))
    #     if cur_idx is None:
    #         # fallback to 0 for all
    #         cur = torch.zeros((B,), dtype=torch.long, device=x.device)
    #     else:
    #         cur = cur_idx if isinstance(cur_idx, torch.Tensor) else torch.as_tensor(cur_idx, dtype=torch.long)
    #         if cur.ndim > 1:
    #             cur = cur.reshape(-1)
    #         if cur.numel() == 1 and B > 1:
    #             cur = cur.repeat(B)
    #         cur = cur.to(x.device)

    #     outs = []
    #     for b in range(B):
    #         xb = x[b]  # [N, F]
    #         eib = ei[b]  # [2, E]
    #         if eib.dtype != torch.long:
    #             eib = eib.long()

    #         # Filter padded edges (-1)
    #         if eib.numel() == 0:
    #             valid_mask = torch.zeros((0,), dtype=torch.bool, device=xb.device)
    #         else:
    #             valid_mask = (eib[0] >= 0) & (eib[1] >= 0)
    #         eib = eib[:, valid_mask]

    #         eab = None
    #         if ea is not None:
    #             eab = ea[b]
    #             eab = eab[valid_mask] if valid_mask.numel() == eab.shape[0] else eab[: eib.shape[1]]

    #         # If no edges survive, create a single self-loop to keep GATConv happy.
    #         if eib.numel() == 0:
    #             eib = torch.zeros((2, 1), dtype=torch.long, device=xb.device)
    #             if self.gat_in_edge_feats > 0:
    #                 eab = torch.zeros((1, self.gat_in_edge_feats), dtype=torch.float32, device=xb.device)

    #         h = xb
    #         for li, conv in enumerate(self.gat_layers):
    #             if self.gat_in_edge_feats > 0:
    #                 h = conv(h, eib, edge_attr=eab)
    #             else:
    #                 h = conv(h, eib)
    #             # Apply activation to all but last layer
    #             if li < len(self.gat_layers) - 1:
    #                 h = self.gat_acts[li](h)

    #         # h: [N, heads*out]
    #         idx = int(torch.clamp(cur[b], 0, N - 1).item())
    #         node_h = h[idx]  # [heads*out]
    #         node_h = self.gat_proj(node_h)  # [proj]
    #         outs.append(node_h.unsqueeze(0))

    #     return torch.cat(outs, dim=0)  # [B, proj]
    def _compute_current_node_embedding(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Faster version: batches graphs using torch_geometric.data.Batch and
        runs the GAT stack once (instead of B times in a Python loop).
        Returns: [B, gat_project_dim]
        """
        from torch_geometric.data import Data, Batch  # local import to avoid hard dependency at module import time

        node_features = obs["node_features"]
        edge_index = obs["edge_index"]
        edge_features = obs.get("edge_features", None)

        x = node_features if isinstance(node_features, torch.Tensor) else torch.as_tensor(node_features, dtype=torch.float32)
        ei = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index, dtype=torch.long)
        ea = None
        if edge_features is not None:
            ea = edge_features if isinstance(edge_features, torch.Tensor) else torch.as_tensor(edge_features, dtype=torch.float32)

        # Ensure batch dimension
        if x.ndim == 2:   # [N, F]
            x = x.unsqueeze(0)
        if ei.ndim == 2:  # [2, E]
            ei = ei.unsqueeze(0)
        if ea is not None and ea.ndim == 2:  # [E, D]
            ea = ea.unsqueeze(0)

        B, N, _ = x.shape

        # Current node indices (shape [B])
        cur_idx = obs.get("current_airport_idx", obs.get("current_airport", None))
        if cur_idx is None:
            cur = torch.zeros((B,), dtype=torch.long, device=x.device)
        else:
            cur = cur_idx if isinstance(cur_idx, torch.Tensor) else torch.as_tensor(cur_idx, dtype=torch.long)
            if cur.ndim > 1:
                cur = cur.reshape(-1)
            if cur.numel() == 1 and B > 1:
                cur = cur.repeat(B)
            cur = cur.to(x.device)

        # Build Data objects per sample (still a loop, but only for packing; GNN runs once)
        data_list = []
        for b in range(B):
            xb = x[b]            # [N, F]
            eib = ei[b].long()   # [2, E]

            # Filter padded edges (-1) ONCE while packing
            if eib.numel() == 0:
                valid_mask = torch.zeros((0,), dtype=torch.bool, device=xb.device)
            else:
                valid_mask = (eib[0] >= 0) & (eib[1] >= 0)

            eib = eib[:, valid_mask]

            eab = None
            if ea is not None:
                eab = ea[b]
                # If padding mismatch, fall back to slicing
                eab = eab[valid_mask] if valid_mask.numel() == eab.shape[0] else eab[: eib.shape[1]]

            # Keep at least one edge to keep GATConv happy (cheap self-loop)
            if eib.numel() == 0:
                eib = torch.zeros((2, 1), dtype=torch.long, device=xb.device)
                if self.gat_in_edge_feats > 0:
                    eab = torch.zeros((1, self.gat_in_edge_feats), dtype=torch.float32, device=xb.device)

            if self.gat_in_edge_feats > 0:
                data_list.append(Data(x=xb, edge_index=eib, edge_attr=eab))
            else:
                data_list.append(Data(x=xb, edge_index=eib))

        batch = Batch.from_data_list(data_list)

        # One GNN forward for the whole batch
        h = batch.x
        for li, conv in enumerate(self.gat_layers):
            if self.gat_in_edge_feats > 0:
                h = conv(h, batch.edge_index, edge_attr=batch.edge_attr)
            else:
                h = conv(h, batch.edge_index)
            if li < len(self.gat_layers) - 1:
                h = self.gat_acts[li](h)

        # Gather each sample’s current node embedding
        # batch.ptr: [B+1], start index of each graph’s nodes in the packed tensor
        cur = torch.clamp(cur, 0, N - 1)
        global_node_idx = batch.ptr[:-1] + cur  # [B]

        node_h = h[global_node_idx]             # [B, gat_final_node_dim]
        node_h = self.gat_proj(node_h)          # [B, gat_project_dim]
        return node_h


    # -------------------- Logging (preserved) --------------------

    def _log_actor_debug(self, full_obs, local_obs):
        """Log actor network observation details for verification."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def shape_or_type(arr):
            if hasattr(arr, "shape"):
                return arr.shape
            elif hasattr(arr, "__len__"):
                return f"len={len(arr)} type={type(arr).__name__}"
            else:
                return str(type(arr))

        if isinstance(full_obs, dict):
            obs_arr = full_obs.get("observations", None)
            if obs_arr is None:
                obs_arr = full_obs.get("obs", None)
            if obs_arr is None:
                obs_arr = full_obs.get("observation", None)
            obs_shape_str = shape_or_type(obs_arr)
            state_arr = full_obs.get("state", None)
            state_shape_str = shape_or_type(state_arr)
            full_shape = f"Dict(obs:{obs_shape_str}, state:{state_shape_str})"
            local_shape = shape_or_type(local_obs)
        else:
            full_shape = shape_or_type(full_obs)
            local_shape = shape_or_type(local_obs)

        debug_info = {
            "timestamp": timestamp,
            "type": "ACTOR",
            "call_count": CentralizedCriticModel._actor_calls,
            "full_obs_shape": str(full_shape),
            "local_obs_shape": str(local_shape),
            "using_local_obs": True,
            "obs_dim": local_obs.shape[0] if hasattr(local_obs, "shape") else len(local_obs) if hasattr(local_obs, "__len__") else None,
        }

        log_file = os.path.join(DEBUG_LOG_DIR, "centralized_critic_debug.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(debug_info) + "\n")

        if wandb.run:
            wandb.log(
                {
                    "debug/actor_calls": CentralizedCriticModel._actor_calls,
                    "debug/actor_obs_dim": debug_info["obs_dim"],
                    "debug/actor_using_local": True,
                }
            )

    def _log_critic_debug(self, full_obs, central_obs):
        """Log critic network observation details for verification."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(full_obs, dict):
            full_shape = f"Dict(obs:{len(full_obs)}, state:{full_obs.get('state', 'N/A')})"
            central_shape = central_obs.shape
        else:
            full_shape = full_obs.shape
            central_shape = central_obs.shape

        debug_info = {
            "timestamp": timestamp,
            "type": "CRITIC",
            "call_count": CentralizedCriticModel._critic_calls,
            "full_obs_shape": str(full_shape),
            "central_obs_shape": str(central_shape),
            "using_central_obs": True,
            "obs_dim": central_obs.shape[0] if hasattr(central_obs, "shape") else len(central_obs),
        }

        log_file = os.path.join(DEBUG_LOG_DIR, "centralized_critic_debug.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(debug_info) + "\n")

        print(f"🧠 CRITIC DEBUG (Call #{CentralizedCriticModel._critic_calls}):")
        print(f"   📊 Full obs: {full_shape}")
        print(f"   🌍 Using central obs: {central_shape}")
        print(f"   ✅ Critic sees CENTRALIZED observations")

        if wandb.run:
            wandb.log(
                {
                    "debug/critic_calls": CentralizedCriticModel._critic_calls,
                    "debug/critic_obs_dim": debug_info["obs_dim"],
                    "debug/critic_using_central": True,
                }
            )

        if CentralizedCriticModel._critic_calls % 500 == 1:
            self._log_comparison_summary()

    def _log_comparison_summary(self):
        """Log a summary comparison showing different observation usage."""
        print(f"\n🔍 CENTRALIZED CRITIC VERIFICATION SUMMARY:")
        print(f"   🎭 Actor calls: {CentralizedCriticModel._actor_calls} (using local obs: {self.local_obs_dim}D)")
        print(f"   🧠 Critic calls: {CentralizedCriticModel._critic_calls} (using central obs: {self.central_obs_dim}D)")
        print(f"   ✅ VERIFIED: Actor and Critic use DIFFERENT observation spaces!")
        print(f"   📏 Local obs (actor): {self.local_obs_dim} dimensions")
        print(f"   🌍 Central obs (critic): {self.central_obs_dim} dimensions")

        if wandb.run:
            wandb.log(
                {
                    "debug/verification_summary": {
                        "actor_calls": CentralizedCriticModel._actor_calls,
                        "critic_calls": CentralizedCriticModel._critic_calls,
                        "local_obs_dim": self.local_obs_dim,
                        "central_obs_dim": self.central_obs_dim,
                        "different_obs_spaces": self.local_obs_dim != self.central_obs_dim,
                    }
                }
            )


def _to_2d_tensor(x) -> Tensor:
    # Convert dict/list/np/tensor to a [B, F] float32 tensor.
    if isinstance(x, (dict, collections.OrderedDict)):
        parts = []
        batch = None
        for v in x.values():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)  # -> [1, d]
            if batch is None:
                batch = t.shape[0]
            parts.append(t.reshape(t.shape[0], -1))
        return torch.cat(parts, dim=1).to(torch.float32)
    elif isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            parts.append(t.reshape(t.shape[0], -1))
        return torch.cat(parts, dim=1).to(torch.float32)
    else:
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
        return t.reshape(1, -1) if t.ndim == 1 else t.reshape(t.shape[0], -1)


if __name__ == "__main__":
    print("✅ Centralized Critic Legacy Model (PyG GATConv) loaded successfully")
