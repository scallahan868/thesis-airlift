"""centralized_critic_model.py

Legacy (TorchModelV2) centralized-critic model for RLlib MAPPO/PPO.

This variant:
- Initializes an EGAT encoder (edge-aware GAT) using EGATConv.
- Builds a DGLGraph internally from `edge_index` found in the observation.
- Concatenates the EGAT node embedding for the agent's current airport into the
  actor's local observation.
- Leaves the critic network untouched (still uses only centralized inputs).

Expected observation keys when dict obs is used:
- node_features: float tensor/array shaped [N, F_n] or [B, N, F_n]
- edge_features: float tensor/array shaped [E, F_e] or [B, E, F_e]
- edge_index: float/int tensor/array shaped [2, E] or [B, 2, E]
    * padded edges should use -1 for u or v (they will be ignored)
- current_airport_idx: int/float tensor/array shaped [B] or scalar
    (fallback keys supported)

Note:
- For RLlib compatibility, avoid passing raw DGLGraph objects in the observation.
  This model constructs the graph from `edge_index` every forward call.
"""

#!/usr/bin/env python3

import os
import json
import collections
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from gymnasium.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

import wandb

from edge_gat_parallel import EGATConv

# RLlib helper (kept for legacy compatibility)
torch, nn = try_import_torch()

# DGL is required for EGAT
try:
    import dgl  # type: ignore
except Exception as e:  # pragma: no cover
    dgl = None
    _dgl_import_error = e
else:
    _dgl_import_error = None

# Debug logging setup
DEBUG_LOG_DIR = "debug_logs"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """Centralized critic model with actor-side EGAT augmentation."""

    # Class-level debug counters
    _debug_counter = 0
    _actor_calls = 0
    _critic_calls = 0

    # Performance optimization: Only log first N critic calls then stop permanently
    MAX_DEBUG_LOGS = 25
    _debug_logs_written = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        print(f"🧠 CentralizedCriticModel (Legacy) setting up...")
        print(f"   📊 Observation space: {obs_space}")
        print(f"   🎯 Action space: {action_space}")
        print(f"   🎮 Num outputs (actions): {num_outputs}")

        cfg = model_config.get("custom_model_config", {})
        self.local_obs_dim = int(cfg.get("local_obs_dim", 907))
        self.central_obs_dim = int(cfg.get("central_obs_dim", 851))

        # -------------------------
        # EGAT (edge-aware GAT) config
        # -------------------------
        self.enable_egat = bool(cfg.get("enable_egat", True))

        # These MUST match your env's provided node_features/edge_features last dims
        self.gat_in_node_feats = int(cfg.get("gat_in_node_feats", 32))
        self.gat_in_edge_feats = int(cfg.get("gat_in_edge_feats", 8))

        # 3-layer, 3-head EGAT as requested
        self.gat_num_layers = int(cfg.get("gat_num_layers", 3))
        self.gat_num_heads = int(cfg.get("gat_num_heads", 3))

        # Per-head dims
        self.gat_hidden_node_feats = int(cfg.get("gat_hidden_node_feats", 32))
        self.gat_hidden_edge_feats = int(cfg.get("gat_hidden_edge_feats", 16))
        self.gat_out_node_feats = int(cfg.get("gat_out_node_feats", 32))
        self.gat_out_edge_feats = int(cfg.get("gat_out_edge_feats", 16))

        if self.enable_egat:
            if dgl is None:
                raise ImportError(
                    "dgl is required to use EGAT, but could not be imported. "
                    "Install dgl or set custom_model_config.enable_egat=False. "
                    f"Import error: {_dgl_import_error}"
                )

            self.egat_layers = nn.ModuleList()

            # Layer 0: in -> hidden
            self.egat_layers.append(
                EGATConv(
                    in_node_feats=self.gat_in_node_feats,
                    in_edge_feats=self.gat_in_edge_feats,
                    out_node_feats=self.gat_hidden_node_feats,
                    out_edge_feats=self.gat_hidden_edge_feats,
                    num_heads=self.gat_num_heads,
                )
            )

            # Middle layers: hidden -> hidden
            for _ in range(max(0, self.gat_num_layers - 2)):
                self.egat_layers.append(
                    EGATConv(
                        in_node_feats=self.gat_hidden_node_feats,
                        in_edge_feats=self.gat_hidden_edge_feats,
                        out_node_feats=self.gat_hidden_node_feats,
                        out_edge_feats=self.gat_hidden_edge_feats,
                        num_heads=self.gat_num_heads,
                    )
                )

            # Final layer: hidden -> out
            if self.gat_num_layers > 1:
                self.egat_layers.append(
                    EGATConv(
                        in_node_feats=self.gat_hidden_node_feats,
                        in_edge_feats=self.gat_hidden_edge_feats,
                        out_node_feats=self.gat_out_node_feats,
                        out_edge_feats=self.gat_out_edge_feats,
                        num_heads=self.gat_num_heads,
                    )
                )

            # Actor receives flattened [H * D_out] for selected node
            self.gat_embed_dim = self.gat_num_heads * self.gat_out_node_feats

            # Optional projection
            proj_dim = int(cfg.get("gat_project_dim", self.gat_embed_dim))
            self.gat_project_dim = proj_dim
            if proj_dim != self.gat_embed_dim:
                self.gat_project = nn.Linear(self.gat_embed_dim, proj_dim)
            else:
                self.gat_project = None
        else:
            self.egat_layers = None
            self.gat_embed_dim = 0
            self.gat_project_dim = 0
            self.gat_project = None

        # -------------------------
        # Actor network (local obs + EGAT embedding)
        # -------------------------
        actor_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        }

        self.actor_input_dim = self.local_obs_dim + (self.gat_project_dim if self.enable_egat else 0)
        actor_obs_space = Box(low=-1.0, high=1.0, shape=(self.actor_input_dim,), dtype=np.float32)
        self.actor_net = FullyConnectedNetwork(
            obs_space=actor_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=actor_model_config,
            name="actor_net",
        )

        # -------------------------
        # Critic network (UNCHANGED)
        # -------------------------
        self.critic_net = nn.Sequential(
            nn.Linear(self.central_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        print(f"   📏 Local obs dim (base): {self.local_obs_dim}")
        if self.enable_egat:
            print(
                f"   🧩 EGAT enabled: {self.gat_num_layers} layers, "
                f"{self.gat_num_heads} heads, actor +{self.gat_project_dim} dims"
            )
        else:
            print("   🧩 EGAT disabled")

        print(f"   ✅ Actor network: {self.actor_input_dim} → FC → {num_outputs}")
        print(f"   ✅ Critic network: {self.central_obs_dim} → 256 → 1")

        # One-time warning flags
        self._warned_missing_current_idx = False
        self._warned_missing_graph_inputs = False

    # -------------------------
    # RLlib hooks
    # -------------------------
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Forward pass for the actor network."""
        obs = input_dict["obs"]
        CentralizedCriticModel._actor_calls += 1

        # Build local tensor (exclude centralized + graph fields) and ensure shape [B, F]
        if isinstance(obs, dict):
            exclude = {
                "action_mask",
                "globalstate",
                "previous_action",
                # graph inputs that should NOT be flattened into local obs
                "node_features",
                "edge_features",
                "edge_index",
                "current_airport_idx",
                "current_airport_id",
                "current_node_idx",
                "current_node_id",
                "current_airport",
                "current_node",
            }
            local_fields = {k: v for k, v in obs.items() if k not in exclude}
            local_tensor = _to_2d_tensor(local_fields)
        else:
            local_tensor = _to_2d_tensor(obs)

        local_tensor = local_tensor.to(torch.float32)

        # EGAT node embedding for current airport (concatenate into actor input)
        if self.enable_egat and isinstance(obs, dict):
            gat_emb = self._compute_current_node_embedding(obs, batch_size=local_tensor.shape[0]).to(local_tensor.device)
            actor_in = torch.cat([local_tensor, gat_emb], dim=1)
        else:
            actor_in = local_tensor

        local_input_dict = dict(input_dict)
        local_input_dict["obs"] = actor_in
        local_input_dict["obs_flat"] = actor_in

        logits, _ = self.actor_net(local_input_dict, state, seq_lens)

        # Apply action mask if present
        if isinstance(obs, dict) and "action_mask" in obs:
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=logits.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)

            valid = (mask <= 1.0 + 1e-6) & (mask >= 0.0 - 1e-6)
            if not torch.all(valid):
                if not hasattr(self, "_logged_bad_mask"):
                    print("\n[WARN] Invalid values found in action_mask! Values will be clamped to {0.0, 1.0}.")
                    self._logged_bad_mask = True
                mask = torch.round(mask).clamp(0.0, 1.0)

            FLOAT_MIN = torch.finfo(logits.dtype).min
            logits = logits + torch.log(mask + 1e-12).clamp(min=FLOAT_MIN)

        self._last_obs = obs
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        """Compute value function using centralized observations (unchanged)."""
        assert hasattr(self, "_last_obs"), "Must call forward() before value_function()"

        obs = self._last_obs
        CentralizedCriticModel._critic_calls += 1

        central_obs = _to_2d_tensor(
            {
                "globalstate": obs["globalstate"],
                "previous_action": obs["previous_action"],
            }
        )

        if (
            CentralizedCriticModel._debug_logs_written < CentralizedCriticModel.MAX_DEBUG_LOGS
            and CentralizedCriticModel._critic_calls <= CentralizedCriticModel.MAX_DEBUG_LOGS
        ):
            self._log_critic_debug(obs, central_obs)
            CentralizedCriticModel._debug_logs_written += 1

        if not isinstance(central_obs, torch.Tensor):
            central_obs = torch.tensor(central_obs, dtype=torch.float32)

        value = self.critic_net(central_obs)
        return value.squeeze(-1)

    # -------------------------
    # EGAT helpers
    # -------------------------
    def _get_current_node_idx(self, obs: dict, batch_size: int) -> torch.Tensor:
        """Return tensor of shape [B] with current node indices."""
        for k in (
            "current_airport_idx",
            "current_node_idx",
            "current_airport_id",
            "current_node_id",
            "current_airport",
            "current_node",
        ):
            if k in obs:
                idx = obs[k]
                t = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx)
                if t.ndim == 0:
                    t = t.view(1).repeat(batch_size)
                elif t.ndim == 1 and t.shape[0] != batch_size:
                    if t.shape[0] == 1:
                        t = t.repeat(batch_size)
                return t.to(dtype=torch.long)

        if not self._warned_missing_current_idx:
            print(
                "[WARN] No current airport/node index found in obs dict. "
                "Using index=0 for EGAT embedding. Provide obs['current_airport_idx'] (preferred) to fix this."
            )
            self._warned_missing_current_idx = True

        return torch.zeros((batch_size,), dtype=torch.long)

    def _compute_current_node_embedding(self, obs: dict, batch_size: int) -> torch.Tensor:
        """Compute EGAT embedding for the current node for each batch element."""
        need = ("node_features", "edge_features", "edge_index")
        if not all(k in obs for k in need):
            if not self._warned_missing_graph_inputs:
                missing = [k for k in need if k not in obs]
                print(f"[WARN] Missing EGAT inputs in obs: {missing}. Returning zeros for EGAT embedding.")
                self._warned_missing_graph_inputs = True
            return torch.zeros((batch_size, self.gat_project_dim), dtype=torch.float32)

        node_feats = obs["node_features"]
        edge_feats = obs["edge_features"]
        edge_index = obs["edge_index"]

        nfts = node_feats if isinstance(node_feats, torch.Tensor) else torch.as_tensor(node_feats, dtype=torch.float32)
        efts = edge_feats if isinstance(edge_feats, torch.Tensor) else torch.as_tensor(edge_feats, dtype=torch.float32)
        eidx = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index)

        # Normalize shapes to:
        # nfts: [B, N, Fn]
        # efts: [B, Emax, Fe]
        # eidx: [B, 2, Emax]
        if nfts.ndim == 2:
            nfts = nfts.unsqueeze(0)
        if efts.ndim == 2:
            efts = efts.unsqueeze(0)
        if eidx.ndim == 2:
            eidx = eidx.unsqueeze(0)

        # Some wrappers store edge_index as float; cast later.
        nfts = nfts.to(torch.float32)
        efts = efts.to(torch.float32)

        # If RLlib gives B=1 but batch_size > 1 (broadcast), repeat.
        if nfts.shape[0] == 1 and batch_size > 1:
            nfts = nfts.repeat(batch_size, 1, 1)
        if efts.shape[0] == 1 and batch_size > 1:
            efts = efts.repeat(batch_size, 1, 1)
        if eidx.shape[0] == 1 and batch_size > 1:
            eidx = eidx.repeat(batch_size, 1, 1)

        # Enforce batch
        B = min(batch_size, nfts.shape[0], efts.shape[0], eidx.shape[0])
        nfts = nfts[:B]
        efts = efts[:B]
        eidx = eidx[:B]

        current_idx = self._get_current_node_idx(obs, batch_size=B)

        # Build a batched DGLGraph and aligned (packed) edge features.
        g_batch, packed_edge_feats = self._build_batched_graph(eidx, efts, num_nodes=nfts.shape[1])

        # Run EGAT on batched graph.
        h_final = self._run_egat_batched(g_batch, nfts, packed_edge_feats)  # [B*N, H, D]

        # Pick current node embedding for each sample
        N = nfts.shape[1]
        offsets = torch.arange(B, device=h_final.device, dtype=torch.long) * N
        idx = current_idx.to(device=h_final.device, dtype=torch.long).clamp(min=0, max=N - 1)
        flat_idx = offsets + idx
        node_emb = h_final[flat_idx].reshape(B, -1)  # [B, H*D]

        if self.gat_project is not None:
            node_emb = self.gat_project(node_emb)

        return node_emb.to(dtype=torch.float32)

    def _build_batched_graph(self, edge_index_b: torch.Tensor, edge_feats_b: torch.Tensor, num_nodes: int):
        """Build a DGL batched graph from padded edge_index.

        edge_index_b: [B, 2, Emax] (may be float)
        edge_feats_b: [B, Emax, Fe]

        Returns:
          g_batch: dgl.DGLGraph (batched)
          packed_edge_feats: [sum(E_valid_b), Fe]
        """
        device = next(self.parameters()).device

        # Cast edge_index to long and move to device
        eidx = edge_index_b.to(device=device)
        if eidx.dtype.is_floating_point:
            eidx = eidx.round().to(torch.long)
        else:
            eidx = eidx.to(torch.long)

        ef = edge_feats_b.to(device=device, dtype=torch.float32)

        graphs = []
        packed_efs = []
        B, _, Emax = eidx.shape

        for b in range(B):
            u = eidx[b, 0]
            v = eidx[b, 1]
            valid = (u >= 0) & (v >= 0)
            u_valid = u[valid]
            v_valid = v[valid]

            # Build graph (allow empty edge lists)
            g = dgl.graph((u_valid, v_valid), num_nodes=num_nodes, device=device)
            graphs.append(g)

            if valid.any():
                packed_efs.append(ef[b, valid])
            else:
                packed_efs.append(ef[b, :0])  # empty [0, Fe]

        g_batch = dgl.batch(graphs)
        packed_edge_feats = torch.cat(packed_efs, dim=0) if len(packed_efs) > 0 else ef[:0]
        return g_batch, packed_edge_feats

    def _run_egat_batched(self, g_batch, nfts_b: torch.Tensor, efts_packed: torch.Tensor) -> torch.Tensor:
        """Run EGAT layers on a batched graph.

        Inputs:
          g_batch: batched DGLGraph with total_nodes = B*N
          nfts_b: [B, N, Fn]
          efts_packed: [E_total, Fe]

        Returns:
          h_final: [B*N, H, D_out]
        """
        device = next(self.parameters()).device
        B, N, _ = nfts_b.shape

        # Flatten nodes to match batched graph node ordering
        h = nfts_b.to(device=device, dtype=torch.float32).reshape(B * N, -1)
        f = efts_packed.to(device=device, dtype=torch.float32)

        for li, layer in enumerate(self.egat_layers):
            h_heads, f_heads = layer(g_batch, h, f, get_attention=False)

            # Nonlinearity except final layer
            if li != len(self.egat_layers) - 1:
                h_heads = torch.relu(h_heads)
                f_heads = torch.relu(f_heads)
                h = h_heads.mean(dim=1)  # [B*N, D]
                f = f_heads.mean(dim=1)  # [E_total, F]
            else:
                h_final = h_heads

        return h_final

    # -------------------------
    # Debug helpers
    # -------------------------
    def _log_critic_debug(self, full_obs, central_obs):
        """Log critic network observation details for verification."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(full_obs, dict):
            full_shape = f"Dict(keys={list(full_obs.keys())})"
            central_shape = tuple(central_obs.shape)
        else:
            full_shape = getattr(full_obs, "shape", str(type(full_obs)))
            central_shape = tuple(central_obs.shape)

        debug_info = {
            "timestamp": timestamp,
            "type": "CRITIC",
            "call_count": CentralizedCriticModel._critic_calls,
            "full_obs_shape": str(full_shape),
            "central_obs_shape": str(central_shape),
            "using_central_obs": True,
            "obs_dim": int(central_obs.shape[-1]),
        }

        log_file = os.path.join(DEBUG_LOG_DIR, "centralized_critic_debug.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(debug_info) + "\n")

        print(f"🧠 CRITIC DEBUG (Call #{CentralizedCriticModel._critic_calls}): central_obs {central_shape}")

        if wandb.run:
            wandb.log(
                {
                    "debug/critic_calls": CentralizedCriticModel._critic_calls,
                    "debug/critic_obs_dim": debug_info["obs_dim"],
                    "debug/critic_using_central": True,
                }
            )


def _to_2d_tensor(x) -> Tensor:
    """Convert dict/list/np/tensor to a [B, F] float32 tensor."""
    if isinstance(x, (dict, collections.OrderedDict)):
        parts = []
        batch = None
        for v in x.values():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if batch is None:
                batch = t.shape[0]
            parts.append(t.reshape(t.shape[0], -1))
        return torch.cat(parts, dim=1).to(torch.float32)

    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            parts.append(t.reshape(t.shape[0], -1))
        return torch.cat(parts, dim=1).to(torch.float32)

    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    return t.reshape(1, -1) if t.ndim == 1 else t.reshape(t.shape[0], -1)


if __name__ == "__main__":
    print("✅ Centralized Critic Legacy Model (graph built from edge_index) loaded successfully")


