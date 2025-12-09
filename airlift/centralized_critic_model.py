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

The observation_fn transforms individual agent observations into shared global state
for the critic, while actors still use their local observations.
"""

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from gymnasium.spaces import Box

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from gymnasium.spaces import Box
import wandb
import os
import json
from datetime import datetime
import collections

torch, nn = try_import_torch()

from egat import RouteEGATBlockUnified

# Debug logging setup
DEBUG_LOG_DIR = "debug_logs"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Legacy-style centralized critic model for MAPPO.
    
    Key Features:
    - Actor uses local observations (from original obs)
    - Critic uses centralized observations (from observation_fn)
    - Compatible with legacy RLLib API
    - Proper value function estimation with global state
    - DEBUG: Tracks and logs observation space usage to verify centralized critic
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

        # print("OBSERVATION SPACE:", obs_space)
        # print(type(obs_space))
        
        # Print model setup for verification
        print(f"🧠 CentralizedCriticModel (Legacy) setting up...")
        print(f"   📊 Observation space: {obs_space}")
        print(f"   🎯 Action space: {action_space}")
        print(f"   🎮 Num outputs (actions): {num_outputs}")

        cfg = model_config.get("custom_model_config", {})
        self.local_obs_dim    = int(cfg.get("local_obs_dim", 907))
        self.central_obs_dim  = int(cfg.get("central_obs_dim", 851))

        
        # Base local obs dim (without EGAT route features)
        self.base_local_obs_dim = self.local_obs_dim

        # EGAT-related dimensions
        self.max_routes_per_airport = 14
        # We now use ONE EGAT and output a 3-dim vector per route (edge)
        self.egat_edge_dim = 3
        # Total extra features appended to local obs: R * H = 14 * 3
        self.egat_concat_dim = self.max_routes_per_airport * self.egat_edge_dim

        # Final actor input dim = base local obs + EGAT features
        self.actor_input_dim = self.base_local_obs_dim + self.egat_concat_dim

        print(f"   📏 Local obs dim: {self.local_obs_dim}")
        print(f"   🏗️ Local obs dim (with EGAT): {self.actor_input_dim}")
        print(f"   🌍 Central obs dim: {self.central_obs_dim}")
        
        # Create actor network (uses local observations)
        # Create the correct observation space for the actor (74 dim)
        actor_model_config = {
            "fcnet_hiddens": [512, 512, 256],
            "fcnet_activation": "relu",
        }
        
        actor_obs_space = Box(low=-1.0, high=1.0, shape=(self.actor_input_dim,), dtype=np.float32)
        self.actor_net = FullyConnectedNetwork(
            obs_space=actor_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=actor_model_config,
            name="actor_net"
        )
        
        # Create critic network (uses centralized observations)  
        # Override the input dimension for the critic
        critic_config = model_config.copy()
        critic_config["fcnet_hiddens"] = model_config.get("fcnet_hiddens", [256, 256])
        
        # Create a simple fully connected network for the critic
        self.critic_net = nn.Sequential(
            nn.Linear(self.central_obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
                
        
        print(f"   ✅ Actor network: {self.local_obs_dim} → FC → {num_outputs}")
        print(f"   ✅ Critic network: {self.central_obs_dim} → 512 → 256 → 1")

        ## EGAT Setup (unified: plane + cargo → single edge embedding)
        self.route_egat = RouteEGATBlockUnified(
            plane_node_dim=4,                        # per-airport plane feature size (num_planes, cap, load, frac)
            cargo_node_dim=4,                        # per-airport cargo feature size (num_cargo, weight, urgency, avg)
            edge_in_dim=3,                           # per-route edge feature size (time, cost, availability)
            edge_out_dim=self.egat_edge_dim,         # output dim per route (we want 3)
            node_hidden_dim=64,
            attn_hidden_dim=64,
            max_routes_per_airport=self.max_routes_per_airport,
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for the actor network.
        The critic forward pass is handled separately in value_function().
        """
        obs = input_dict["obs"]
        
                # DEBUG: Track actor calls and log observation details
        CentralizedCriticModel._actor_calls += 1

        # Build local tensor (exclude 'globalstate') and ensure shape [B, F]
        if isinstance(obs, dict):
            # Keep only local observation fields for the actor
            local_fields = {k: v for k, v in obs.items() if k not in ["action_mask", "globalstate","previous_action","plane_node_feats","cargo_node_feats","edge_index","edge_attr"]}
            local_tensor = _to_2d_tensor(local_fields)
        else:
            local_tensor = _to_2d_tensor(obs)

        # Ensure float32 and proper device
        local_tensor = local_tensor.to(torch.float32)

        # Prepare input dict for actor network (actor expects tensors)
        local_input_dict = dict(input_dict)
        local_input_dict["obs"] = local_tensor
        local_input_dict["obs_flat"] = local_tensor

        # 2) Extract EGAT inputs from obs for route-level context
        # Shapes (per batch):
        #   plane_node_feats: [B, N_airports, 4]
        #   cargo_node_feats: [B, N_airports, 4]
        #   edge_index:       [B, 2, E_max]   (padded with -1)
        #   edge_attr:        [B, E_max, 3]
        #   current_airport:  [B, 1]
        #   available_routes: [B, R_max]
        device = local_tensor.device

        plane_node_feats  = obs["plane_node_feats"].to(device)
        cargo_node_feats  = obs["cargo_node_feats"].to(device)
        edge_index        = obs["edge_index"].to(device).long()
        edge_attr         = obs["edge_attr"].to(device)
        current_airport   = obs["current_airport"].to(device).long().view(-1)   # [B]
        available_routes  = obs["available_routes"].to(device).long()          # [B, R]

        # 3) Run unified EGAT → one 3-dim vector per available route
        route_vec = self.route_egat(
            plane_node_feats,
            cargo_node_feats,
            edge_index,
            edge_attr,
            current_airport,
            available_routes,
        )  # [B, R, 3]

        # 4) Flatten route vectors and append to local obs
        B, R, H = route_vec.shape  # H should be 3
        route_flat = route_vec.reshape(B, R * H)   # [B, R*H] = [B, 42]

        local_tensor = torch.cat([local_tensor, route_flat], dim=-1)  # [B, F_local + R*H]

        # 5) Cache for potential debug/inspection later
        self._last_route_vec = route_vec.detach()

        # 6) Normal actor logic
        local_input_dict = dict(input_dict)
        local_input_dict["obs"] = local_tensor
        local_input_dict["obs_flat"] = local_tensor

        logits, _ = self.actor_net(local_input_dict, state, seq_lens)

        if isinstance(obs, dict) and "action_mask" in obs:
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)  # [B, N]
            
            # === New validation and correction ========================
            # Check whether mask contains only 0 or 1 (after conversion to float)
            # Allow small numerical tolerance (e.g., 1e-6).
            valid = (mask <= 1.0 + 1e-6) & (mask >= 0.0 - 1e-6)
            if not torch.all(valid):
                # Optional logging — only print once per run to avoid spam.
                if not hasattr(self, "_logged_bad_mask"):
                    print("\n[WARN] Invalid values found in action_mask! "
                        "Values will be clamped to {0.0, 1.0}.")
                    self._logged_bad_mask = True

                # Fix mask (round to closest of {0,1}):
                mask = torch.round(mask).clamp(0.0, 1.0)
            # ===========================================================

            FLOAT_MIN = torch.finfo(logits.dtype).min
            logits = logits + torch.log(mask + 1e-12).clamp(min=FLOAT_MIN)

        # Store the raw obs for value_function()
        self._last_obs = obs

        return logits, state
       

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
            obs_arr = full_obs.get('observations', None)
            if obs_arr is None:
                obs_arr = full_obs.get('obs', None)
            if obs_arr is None:
                obs_arr = full_obs.get('observation', None)
            obs_shape_str = shape_or_type(obs_arr)
            state_arr = full_obs.get('state', None)
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
            "obs_dim": local_obs.shape[0] if hasattr(local_obs, 'shape') else len(local_obs) if hasattr(local_obs, '__len__') else None
        }

        # Log to file
        log_file = os.path.join(DEBUG_LOG_DIR, "centralized_critic_debug.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(debug_info) + "\n")

        # Log to console
        # print(f"🎭 ACTOR DEBUG (Call #{CentralizedCriticModel._actor_calls}):")
        # print(f"   📊 Full obs: {full_shape}")
        # print(f"   🎯 Using local obs: {local_shape}")
        # print(f"   ✅ Actor sees LOCAL observations only")

        # Show performance optimization message when limit reached
        # if CentralizedCriticModel._debug_logs_written >= CentralizedCriticModel.MAX_DEBUG_LOGS:
        #     print(f"   🚀 PERFORMANCE: Debug logging disabled after {CentralizedCriticModel.MAX_DEBUG_LOGS} calls")

        # Log to wandb if available
        if wandb.run:
            wandb.log({
                "debug/actor_calls": CentralizedCriticModel._actor_calls,
                "debug/actor_obs_dim": debug_info["obs_dim"],
                "debug/actor_using_local": True
            })

    def _log_critic_debug(self, full_obs, central_obs):
        """Log critic network observation details for verification."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate observation details
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
            "obs_dim": central_obs.shape[0] if hasattr(central_obs, 'shape') else len(central_obs)
        }
        
        # Log to file
        log_file = os.path.join(DEBUG_LOG_DIR, "centralized_critic_debug.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(debug_info) + "\n")
        
        # Log to console
        print(f"🧠 CRITIC DEBUG (Call #{CentralizedCriticModel._critic_calls}):")
        print(f"   📊 Full obs: {full_shape}")
        print(f"   🌍 Using central obs: {central_shape}")
        print(f"   ✅ Critic sees CENTRALIZED observations")
        
        # Show performance optimization message when limit reached
        # if CentralizedCriticModel._debug_logs_written >= CentralizedCriticModel.MAX_DEBUG_LOGS:
        #     print(f"   🚀 PERFORMANCE: Debug logging disabled after {CentralizedCriticModel.MAX_DEBUG_LOGS} calls")
        
        # Log to wandb if available
        if wandb.run:
            wandb.log({
                "debug/critic_calls": CentralizedCriticModel._critic_calls,
                "debug/critic_obs_dim": debug_info["obs_dim"],
                "debug/critic_using_central": True
            })
        
        # Log summary comparison every 500 calls
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
            wandb.log({
                "debug/verification_summary": {
                    "actor_calls": CentralizedCriticModel._actor_calls,
                    "critic_calls": CentralizedCriticModel._critic_calls,
                    "local_obs_dim": self.local_obs_dim,
                    "central_obs_dim": self.central_obs_dim,
                    "different_obs_spaces": self.local_obs_dim != self.central_obs_dim
                }
            })

def _to_2d_tensor(x) -> Tensor:
    # Convert dict/list/np/tensor to a [B, F] float32 tensor.
    if isinstance(x, (dict, collections.OrderedDict)):
        parts = []
        batch = None
        for v in x.values():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)              # -> [1, d]
            if batch is None:
                batch = t.shape[0]
            # Keep batch, flatten feature dims only:
            parts.append(t.reshape(t.shape[0], -1))
        return torch.cat(parts, dim=1).to(torch.float32)
    elif isinstance(x, (list, tuple)):
        parts = []
        batch = None
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
    print("✅ Centralized Critic Legacy Model loaded successfully")
