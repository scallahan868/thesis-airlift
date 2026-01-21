"""
centralized_critic_model_sage.py

Legacy RLlib TorchModelV2 centralized-critic model (MAPPO-style) with a
GraphSAGE (PyG) encoder that runs ONCE per RLlib minibatch by batching
graphs with torch_geometric.data.Batch.

- Actor uses local obs (+ optional GraphSAGE embedding of the agent's current node)
- Critic uses centralized obs (globalstate + previous_action), unchanged from your GAT model.

Notes:
- Vanilla GraphSAGE (SAGEConv) does NOT use edge_attr. We still accept/ignore
  edge_features because your wrapper provides them, but they won't affect messages.
  (If you want edge features to matter, ask and I’ll give you a fast GINEConv version.)
"""

#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from gymnasium.spaces import Box

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override

# --- PyG imports ---
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
except Exception as e:
    Data = None
    Batch = None
    SAGEConv = None
    _pyg_import_error = e
else:
    _pyg_import_error = None


class CentralizedCriticModelGraphSAGE(TorchModelV2, nn.Module):
    """
    Centralized critic legacy model with optional GraphSAGE embedding for actor input.

    Expected (dict) obs keys from your wrapper for GraphSAGE:
      - node_features:    [B, N, F] or [N, F]
      - edge_index:       [B, 2, E] or [2, E]  (padded with -1)
      - edge_features:    [B, E, D] or [E, D]  (ignored by GraphSAGE; kept for compatibility)
      - current_airport_idx: [B] or [B,1] or scalar (index 0..N-1)

    Centralized critic uses:
      - globalstate
      - previous_action
    """

    _actor_calls = 0
    _critic_calls = 0

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {}) if model_config else {}

        # ---- dims (match your existing config keys) ----
        self.local_obs_dim = int(cfg.get("local_obs_dim", 0))
        self.central_obs_dim = int(cfg.get("central_obs_dim", 0))

        # Enable flag: allow either enable_sage or (for easy switching) enable_gat to act as enable_sage.
        self.enable_sage = bool(cfg.get("enable_sage", cfg.get("enable_gat", False)))

        # GraphSAGE dims
        self.sage_in_node_feats = int(cfg.get("sage_in_node_feats", cfg.get("gat_in_node_feats", 7)))
        self.sage_num_layers = int(cfg.get("sage_num_layers", cfg.get("gat_num_layers", 2)))
        self.sage_hidden = int(cfg.get("sage_hidden", cfg.get("gat_hidden", 32)))
        self.sage_out = int(cfg.get("sage_out", cfg.get("gat_out", 32)))

        # Optional projection dim for embedding concatenated to actor input
        self.sage_project_dim = int(cfg.get("sage_project_dim", cfg.get("gat_project_dim", self.sage_out)))
        self.sage_dropout = float(cfg.get("sage_dropout", 0.0))

        # If disabled, extra dim is 0.
        self._actor_extra_dim = self.sage_project_dim if self.enable_sage else 0

        print("\n✅ Loaded CentralizedCriticModelGraphSAGE (legacy TorchModelV2)")
        print(f"   📏 Local obs dim: {self.local_obs_dim}")
        print(f"   🌍 Central obs dim: {self.central_obs_dim}")
        if self.enable_sage:
            print(
                f"   🧠 PyG GraphSAGE enabled: layers={self.sage_num_layers}, "
                f"in_node={self.sage_in_node_feats}, hidden={self.sage_hidden}, out={self.sage_out}, "
                f"proj={self.sage_project_dim}, dropout={self.sage_dropout}"
            )

        # --- Actor network (local obs + optional GraphSAGE embedding) ---
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

        # --- PyG GraphSAGE encoder (actor-only) ---
        if self.enable_sage:
            if SAGEConv is None or Batch is None or Data is None:
                raise ImportError(
                    "torch_geometric is required for SAGEConv but could not be imported. "
                    f"Original error: {_pyg_import_error}"
                )

            if self.sage_num_layers < 1:
                raise ValueError("sage_num_layers must be >= 1")

            self.sage_layers = nn.ModuleList()

            if self.sage_num_layers == 1:
                self.sage_layers.append(SAGEConv(self.sage_in_node_feats, self.sage_out, aggr="mean"))
                final_dim = self.sage_out
            else:
                self.sage_layers.append(SAGEConv(self.sage_in_node_feats, self.sage_hidden, aggr="mean"))
                for _ in range(self.sage_num_layers - 2):
                    self.sage_layers.append(SAGEConv(self.sage_hidden, self.sage_hidden, aggr="mean"))
                self.sage_layers.append(SAGEConv(self.sage_hidden, self.sage_out, aggr="mean"))
                final_dim = self.sage_out

            self.sage_proj = nn.Identity() if final_dim == self.sage_project_dim else nn.Linear(final_dim, self.sage_project_dim)
            self.sage_act = nn.ReLU()
            self.sage_drop = nn.Dropout(p=self.sage_dropout)

        print(f"   ✅ Actor network: {actor_input_dim} → FC → {num_outputs}")
        print(f"   ✅ Critic network: {self.central_obs_dim} → 512 → 256 → 1\n")

        # Stored for value_function()
        self._last_obs: Optional[Any] = None
        self._last_vf: Optional[Tensor] = None

    # -------------------- Actor forward --------------------

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for the actor network.
        Critic forward is handled in value_function().
        """
        CentralizedCriticModelGraphSAGE._actor_calls += 1

        obs = input_dict["obs"]

        # Flatten local obs the same way your original model does:
        # - If dict obs, remove centralized / graph-only keys
        if isinstance(obs, dict):
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

        # --- Optional: compute GraphSAGE embedding and concat onto local_tensor ---
        if self.enable_sage and isinstance(obs, dict) and "node_features" in obs and "edge_index" in obs:
            try:
                sage_emb = self._compute_current_node_embedding(obs)  # [B, sage_project_dim]
                local_tensor = torch.cat([local_tensor, sage_emb.to(local_tensor.device)], dim=1)
            except Exception as e:
                # Fail loud once; then keep training without SAGE (avoids crash loops).
                if not hasattr(self, "_logged_sage_failure"):
                    print(f"\n[WARN] GraphSAGE embedding computation failed once; continuing without it. Error: {e}\n")
                    self._logged_sage_failure = True

                # If actor expects extra dims, still pad zeros to keep shape consistent.
                if self._actor_extra_dim > 0:
                    zeros = torch.zeros(
                        (local_tensor.shape[0], self._actor_extra_dim),
                        dtype=local_tensor.dtype,
                        device=local_tensor.device,
                    )
                    local_tensor = torch.cat([local_tensor, zeros], dim=1)

        # Prepare input dict for actor network
        local_input_dict = dict(input_dict)
        local_input_dict["obs"] = local_tensor
        local_input_dict["obs_flat"] = local_tensor

        logits, _ = self.actor_net(local_input_dict, state, seq_lens)

        # Action masking (if present)
        if isinstance(obs, dict) and "action_mask" in obs:
            mask = obs["action_mask"]
            mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
            mask = mask.to(logits.device).to(logits.dtype)

            FLOAT_MIN = torch.finfo(logits.dtype).min
            logits = logits + torch.log(mask + 1e-12).clamp(min=FLOAT_MIN)

        # Store the raw obs for value_function()
        self._last_obs = obs

        return logits, state

    # -------------------- Critic forward (unchanged) --------------------

    @override(TorchModelV2)
    def value_function(self):
        """
        Centralized critic value function using (globalstate + previous_action).
        """
        CentralizedCriticModelGraphSAGE._critic_calls += 1

        if self._last_obs is None:
            # RLlib can call value_function before forward in some edge cases.
            return torch.zeros((1,), dtype=torch.float32)

        obs = self._last_obs

        if isinstance(obs, dict):
            central_obs = _to_2d_tensor(
                {
                    "globalstate": obs["globalstate"],
                    "previous_action": obs["previous_action"],
                }
            )
        else:
            central_obs = _to_2d_tensor(obs)

        central_obs = central_obs.to(torch.float32)
        vf = self.critic_net(central_obs).squeeze(-1)
        self._last_vf = vf
        return vf

    # -------------------- GraphSAGE embedding --------------------

    def _compute_current_node_embedding(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Computes a per-sample node embedding for the agent's current airport via batched PyG SAGEConv.

        Runs GraphSAGE ONCE per minibatch by:
          - Packing each sample's (N,F), (2,E) into a Data list
          - Batch.from_data_list
          - One forward through SAGE layers
          - Gather the embedding at current_airport_idx for each sample
        """
        node_features = obs["node_features"]
        edge_index = obs["edge_index"]
        # edge_features present in obs but GraphSAGE ignores it
        current_idx = obs.get("current_airport_idx", None)

        x = node_features if isinstance(node_features, torch.Tensor) else torch.as_tensor(node_features, dtype=torch.float32)
        ei = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index)

        # Normalize shapes: x -> [B,N,F], ei -> [B,2,E]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if ei.ndim == 2:
            ei = ei.unsqueeze(0)

        B, N, F = x.shape

        if current_idx is None:
            cur = torch.zeros((B,), dtype=torch.long, device=x.device)
        else:
            cur = current_idx if isinstance(current_idx, torch.Tensor) else torch.as_tensor(current_idx)
            if cur.ndim == 2 and cur.shape[1] == 1:
                cur = cur[:, 0]
            if cur.ndim == 0:
                cur = cur.view(1)
            if cur.numel() == 1 and B > 1:
                cur = cur.repeat(B)
            cur = cur.to(x.device).long()

        # Build Data objects per sample (loop is only for packing; GNN runs once)
        data_list: List[Data] = []
        for b in range(B):
            xb = x[b]  # [N,F]
            eib = ei[b]
            if eib.dtype != torch.long:
                eib = eib.long()

            # Filter padded edges (-1)
            if eib.numel() == 0:
                valid_mask = torch.zeros((0,), dtype=torch.bool, device=xb.device)
            else:
                valid_mask = (eib[0] >= 0) & (eib[1] >= 0)
            eib = eib[:, valid_mask]

            # Keep at least one edge to keep message passing stable (cheap self-loop)
            if eib.numel() == 0:
                eib = torch.zeros((2, 1), dtype=torch.long, device=xb.device)

            data_list.append(Data(x=xb, edge_index=eib))

        batch = Batch.from_data_list(data_list)

        # One GNN forward for the whole batch
        h = batch.x
        for li, conv in enumerate(self.sage_layers):
            h = conv(h, batch.edge_index)
            if li < len(self.sage_layers) - 1:
                h = self.sage_act(h)
                if self.sage_dropout > 0.0:
                    h = self.sage_drop(h)

        # Gather each sample’s current node embedding
        cur = torch.clamp(cur, 0, N - 1)
        global_node_idx = batch.ptr[:-1] + cur  # [B]
        node_h = h[global_node_idx]             # [B, sage_out]
        node_h = self.sage_proj(node_h)         # [B, sage_project_dim]
        return node_h


def _to_2d_tensor(x) -> Tensor:
    """
    Convert dict/list/np/tensor to a [B, F] float32 tensor.
    Matches the behavior used in your original model.
    """
    if isinstance(x, dict):
        parts = []
        for v in x.values():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
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
    print("✅ Centralized Critic Legacy Model (PyG GraphSAGE) loaded successfully")
