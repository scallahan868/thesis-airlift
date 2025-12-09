"""
EGAT-style module for route-level edge embeddings.

Provides RouteEGATBlockDual, which is imported by centralized_critic_model.py.

Design goals:
- Support batched inputs from RLlib (batch dim B).
- Take separate node features for planes and cargo.
- Take a shared route graph (edge_index, edge_attr) per sample.
- Produce per-sample, per-route edge embeddings aligned with available_routes.

Input shapes to RouteEGATBlockDual.forward:
    plane_node_feats: [B, N, F_plane]
    cargo_node_feats: [B, N, F_cargo]
    edge_index:       [B, 2, E]      (src, dst indices per edge; padded with -1)
    edge_attr:        [B, E, F_edge]
    current_airport:  [B]            (int indices into [0, N-1])
    available_routes: [B, R_max]     (destination node indices; padded with -1)

Output:
    plane_route_vec:  [B, R_max, plane_edge_dim]
    cargo_route_vec:  [B, R_max, cargo_edge_dim]
"""

import torch
import torch.nn as nn
from torch import Tensor


class EGATEdgeEncoder(nn.Module):
    """Simple edge encoder using source/dest node features and edge features.

    This is intentionally lightweight and shape-robust:
    - node_feats: [N, F_node]
    - edge_index: [2, E]
    - edge_attr:  [E, F_edge]
    Returns:
    - node_hidden: [N, H_node]
    - edge_emb:    [E, H_edge]
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        node_hidden_dim: int,
        attn_hidden_dim: int,   # kept for API compatibility (unused)
        edge_out_dim: int,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, node_hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, node_hidden_dim)
        # Combine src, dst, and edge representations
        self.out_proj = nn.Linear(3 * node_hidden_dim, edge_out_dim)
        self.act = nn.ReLU()

    def forward(
        self,
        node_feats: Tensor,   # [N, F_node]
        edge_index: Tensor,   # [2, E]
        edge_attr: Tensor,    # [E, F_edge]
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=node_feats.device)
        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32, device=node_feats.device)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be of shape [2, E], got {tuple(edge_index.shape)}")

        # Edge endpoints
        src = edge_index[0].long()  # [E]
        dst = edge_index[1].long()  # [E]

        # Project node and edge features
        node_hidden = self.node_proj(node_feats)          # [N, H_node]
        edge_hidden = self.edge_proj(edge_attr)           # [E, H_node]

        src_emb = node_hidden[src]                        # [E, H_node]
        dst_emb = node_hidden[dst]                        # [E, H_node]

        concat = torch.cat([src_emb, dst_emb, edge_hidden], dim=-1)  # [E, 3*H_node]
        edge_emb = self.act(self.out_proj(concat))        # [E, H_edge]

        return node_hidden, edge_emb


class RouteEGATBlockDual(nn.Module):
    """Dual EGAT-style block for plane and cargo graphs over the same routes.

    This module computes edge embeddings for routes, then selects the subset
    corresponding to `available_routes` from `current_airport` for each sample.
    """

    def __init__(
        self,
        plane_node_dim: int,
        cargo_node_dim: int,
        edge_in_dim: int,
        plane_edge_dim: int,
        cargo_edge_dim: int,
        node_hidden_dim: int,
        attn_hidden_dim: int,
        max_routes_per_airport: int,
    ) -> None:
        super().__init__()
        self.plane_edge_dim = plane_edge_dim
        self.cargo_edge_dim = cargo_edge_dim
        self.max_routes_per_airport = max_routes_per_airport

        self.egat_plane = EGATEdgeEncoder(
            node_in_dim=plane_node_dim,
            edge_in_dim=edge_in_dim,
            node_hidden_dim=node_hidden_dim,
            attn_hidden_dim=attn_hidden_dim,
            edge_out_dim=plane_edge_dim,
        )
        self.egat_cargo = EGATEdgeEncoder(
            node_in_dim=cargo_node_dim,
            edge_in_dim=edge_in_dim,
            node_hidden_dim=node_hidden_dim,
            attn_hidden_dim=attn_hidden_dim,
            edge_out_dim=cargo_edge_dim,
        )

    def forward(
        self,
        plane_node_feats: Tensor,    # [B, N, F_plane]
        cargo_node_feats: Tensor,    # [B, N, F_cargo]
        edge_index: Tensor,          # [B, 2, E] or [2, E]
        edge_attr: Tensor,           # [B, E, F_edge] or [E, F_edge]
        current_airport: Tensor,     # [B] or scalar
        available_routes: Tensor,    # [B, R_max] or [R_max]
    ) -> tuple[Tensor, Tensor]:
        device = plane_node_feats.device

        # Ensure batch dimension B
        if plane_node_feats.ndim == 2:
            plane_node_feats = plane_node_feats.unsqueeze(0)
        if cargo_node_feats.ndim == 2:
            cargo_node_feats = cargo_node_feats.unsqueeze(0)

        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0)
        if edge_attr.ndim == 2:
            edge_attr = edge_attr.unsqueeze(0)

        if current_airport.ndim == 0:
            current_airport = current_airport.view(1)
        if available_routes.ndim == 1:
            available_routes = available_routes.view(1, -1)

        B, N, _ = plane_node_feats.shape
        _, _, F_edge = edge_attr.shape
        _, R_max = available_routes.shape

        plane_route_vec = plane_node_feats.new_zeros(
            (B, self.max_routes_per_airport, self.plane_edge_dim)
        )
        cargo_route_vec = plane_node_feats.new_zeros(
            (B, self.max_routes_per_airport, self.cargo_edge_dim)
        )

        for b in range(B):
            x_plane_b = plane_node_feats[b]          # [N, F_plane]
            x_cargo_b = cargo_node_feats[b]          # [N, F_cargo]
            e_idx_b = edge_index[b]                  # [2, E]
            e_attr_b = edge_attr[b]                  # [E, F_edge]

            # Filter out padded edges (marked with -1 in src or dst)
            if e_idx_b.numel() == 0:
                continue
            valid_mask = (e_idx_b[0] >= 0) & (e_idx_b[1] >= 0)
            if valid_mask.sum() == 0:
                continue

            e_idx_valid = e_idx_b[:, valid_mask]     # [2, E_valid]
            e_attr_valid = e_attr_b[valid_mask]      # [E_valid, F_edge]

            # Compute edge embeddings for this sample
            _, plane_edge_emb_b = self.egat_plane(x_plane_b, e_idx_valid, e_attr_valid)
            _, cargo_edge_emb_b = self.egat_cargo(x_cargo_b, e_idx_valid, e_attr_valid)

            curr = int(current_airport[b].item())
            dests = available_routes[b]  # [R_max]

            # For each route slot r, find the matching edge (curr -> dest)
            for r_idx in range(min(self.max_routes_per_airport, R_max)):
                dest = int(dests[r_idx].item())
                if dest < 0:
                    continue

                matches = torch.nonzero(
                    (e_idx_valid[0] == curr) & (e_idx_valid[1] == dest),
                    as_tuple=False,
                )
                if matches.numel() == 0:
                    # No explicit edge matching this route; keep zero vector
                    continue

                e_ind = int(matches[0, 0].item())
                plane_route_vec[b, r_idx] = plane_edge_emb_b[e_ind]
                cargo_route_vec[b, r_idx] = cargo_edge_emb_b[e_ind]

        return plane_route_vec.to(device), cargo_route_vec.to(device)


if __name__ == "__main__":
    # Quick shape sanity test
    B, N, E, R = 2, 5, 7, 3
    plane_node_feats = torch.randn(B, N, 4)
    cargo_node_feats = torch.randn(B, N, 4)

    # Simple fully-connected directed graph (no self loops), padded to E edges
    edge_index = torch.full((B, 2, E), -1, dtype=torch.long)
    edge_attr = torch.zeros(B, E, 3)

    e_count = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if e_count < E:
                edge_index[:, 0, e_count] = i
                edge_index[:, 1, e_count] = j
                e_count += 1

    current_airport = torch.tensor([0, 1])
    available_routes = torch.tensor([[1, 2, -1], [2, 3, 4]])

    block = RouteEGATBlockDual(
        plane_node_dim=4,
        cargo_node_dim=4,
        edge_in_dim=3,
        plane_edge_dim=8,
        cargo_edge_dim=8,
        node_hidden_dim=16,
        attn_hidden_dim=16,
        max_routes_per_airport=R,
    )

    pr, cr = block(
        plane_node_feats,
        cargo_node_feats,
        edge_index,
        edge_attr,
        current_airport,
        available_routes,
    )
    print("plane_route_vec:", pr.shape)
    print("cargo_route_vec:", cr.shape)

