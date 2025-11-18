# full_masking_rlm.py (constructor fixed to avoid LazyLinear)
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns
import numpy as np


def _flatten_obs_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if x.dim() == 2 else x.view(x.size(0), -1)
    if isinstance(x, dict):
        parts: List[torch.Tensor] = []
        for k in sorted(x.keys()):
            t = x[k]
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Observation field '{k}' is not a tensor: {type(t)}")
            parts.append(t if t.dim() == 2 else t.view(t.size(0), -1))
        return torch.cat(parts, dim=-1) if parts else torch.empty((0, 0))
    raise TypeError(f"Unsupported observation type: {type(x)}")


def _space_size(space) -> int:
    """Number of scalars when flattened (supports Box and MultiBinary)."""
    if hasattr(space, "shape") and space.shape is not None:
        return int(np.prod(space.shape))
    if hasattr(space, "n"):  # e.g., Discrete or MultiBinary(n) uses n
        return int(space.n)
    raise ValueError(f"Unsupported space for size calc: {type(space)}")


class FullMaskingTorchRLModule(TorchRLModule):
    """
    Multi-branch masked module for Dict action space:
      - priority:        Discrete -> Categorical
      - destination:     Discrete -> Categorical
      - cargo_to_load:   MultiBinary -> Bernoulli per slot
      - cargo_to_unload: MultiBinary -> Bernoulli per slot
    Expects Columns.OBS to contain: "observations", "priority_mask", "destination_mask",
    "cargo_load_mask", "cargo_unload_mask".
    """

    def __init__(
        self,
        *,
        observation_space,
        action_space,
        model_config=None,
        **kwargs,
    ):
        super().__init__()

        # ---- infer action sizes from action_space ----
        P = int(action_space["priority"].n)
        A = int(action_space["destination"].n)

        def _mb_size(mb):
            if hasattr(mb, "n"):
                return int(mb.n)
            return int(np.prod(mb.shape))

        L = _mb_size(action_space["cargo_to_load"])
        U = _mb_size(action_space["cargo_to_unload"])
        self._sizes = dict(P=P, A=A, L=L, U=U)

        # ---- infer obs_dim from observation_space['observations'] ----
        obs_spec = observation_space["observations"]
        if hasattr(obs_spec, "spaces"):  # Dict of Boxes
            obs_dim = 0
            # use deterministic order
            for k in sorted(obs_spec.spaces.keys()):
                obs_dim += _space_size(obs_spec.spaces[k])
        else:
            obs_dim = _space_size(obs_spec)

        hidden = int((model_config or {}).get("hidden", 256))

        # ---- network (no Lazy layers!) ----
        self.torso = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.priority_head   = nn.Linear(hidden, P)
        self.dest_head       = nn.Linear(hidden, A)
        self.load_head       = nn.Linear(hidden, L)   # Bernoulli logits
        self.unload_head     = nn.Linear(hidden, U)   # Bernoulli logits
        self.value_head      = nn.Linear(hidden, 1)

    # ---------- helpers ----------
    @staticmethod
    def _mask_logits_cat(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits + torch.log(mask.clamp_min(1e-8))

    @staticmethod
    def _mask_logits_bin(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(mask <= 0, -1e9)

    def _split_obs(self, batch: Dict[str, Any]):
        d = batch[Columns.OBS]
        x  = _flatten_obs_tensor(d["observations"]).float()
        mp = d["priority_mask"].float()
        md = d["destination_mask"].float()
        ml = d["cargo_load_mask"].float()
        mu = d["cargo_unload_mask"].float()
        return x, mp, md, ml, mu

    def _forward_common(self, batch: Dict[str, Any]):
        x, mp, md, ml, mu = self._split_obs(batch)
        h  = self.torso(x)
        lp = self.priority_head(h)
        ld = self.dest_head(h)
        ll = self.load_head(h)
        lu = self.unload_head(h)
        lp = self._mask_logits_cat(lp, mp)
        ld = self._mask_logits_cat(ld, md)
        ll = self._mask_logits_bin(ll, ml)
        lu = self._mask_logits_bin(lu, mu)
        v  = self.value_head(h).squeeze(-1)
        return lp, ld, ll, lu, v

    @staticmethod
    def _sample_categorical_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=logits).sample()

    @staticmethod
    def _greedy_bernoulli_logits(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).to(torch.int64)

    @staticmethod
    def _stochastic_bernoulli_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Bernoulli(logits=logits).sample().to(torch.int64)

    # ---------- RLlib API ----------
    def forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        lp, ld, ll, lu, v = self._forward_common(batch)
        pri  = self._sample_categorical_logits(lp)
        dest = self._sample_categorical_logits(ld)
        load = self._greedy_bernoulli_logits(ll)
        unl  = self._greedy_bernoulli_logits(lu)
        return {
            Columns.ACTIONS: {
                "priority":        pri,
                "destination":     dest,
                "cargo_to_load":   load,
                "cargo_to_unload": unl,
            },
            Columns.STATE_OUT: {},
            Columns.VALUES: v,
            Columns.ACTION_DIST_INPUTS: {
                "priority": lp, "destination": ld, "cargo_to_load": ll, "cargo_to_unload": lu
            },
        }

    def forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        lp, ld, ll, lu, v = self._forward_common(batch)
        pri  = self._sample_categorical_logits(lp)
        dest = self._sample_categorical_logits(ld)
        load = self._stochastic_bernoulli_logits(ll)
        unl  = self._stochastic_bernoulli_logits(lu)
        return {
            Columns.ACTIONS: {
                "priority":        pri,
                "destination":     dest,
                "cargo_to_load":   load,
                "cargo_to_unload": unl,
            },
            Columns.STATE_OUT: {},
            Columns.VALUES: v,
            Columns.ACTION_DIST_INPUTS: {
                "priority": lp, "destination": ld, "cargo_to_load": ll, "cargo_to_unload": lu
            },
        }

    @staticmethod
    def _logp_categorical(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=logits).log_prob(actions)

    @staticmethod
    def _logp_bernoulli_multi(logits: torch.Tensor, actions01: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Bernoulli(logits=logits)
        return dist.log_prob(actions01.float()).sum(dim=-1)

    def forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        lp, ld, ll, lu, v = self._forward_common(batch)
        acts = batch[Columns.ACTIONS]
        pri  = acts["priority"].long()
        dest = acts["destination"].long()
        load = acts["cargo_to_load"].long()
        unl  = acts["cargo_to_unload"].long()
        logp_total = (
            self._logp_categorical(lp, pri) +
            self._logp_categorical(ld, dest) +
            self._logp_bernoulli_multi(ll, load) +
            self._logp_bernoulli_multi(lu, unl)
        )
        return {
            Columns.ACTION_DIST_INPUTS: {
                "priority": lp, "destination": ld, "cargo_to_load": ll, "cargo_to_unload": lu
            },
            Columns.ACTION_LOGP: logp_total,
            Columns.VALUES: v,
            Columns.STATE_OUT: {},
        }



