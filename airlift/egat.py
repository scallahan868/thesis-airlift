"""
egat.py

Edge-aware Graph Attention (E-GAT) utilities for Airlift.
Provides:
  - EdgeGATLayer: edge-aware attention over a directed route graph.
  - RouteEGATBlockDual: two E-GAT networks (planes + cargo) that
    produce *separate* per-agent route feature vectors.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGATLayer(nn.Module):
    """
    Simple edge-aware GAT-style layer.

    Given:
      - node features x        : [N, F_x]
      - edges edge_index       : [2, E] with (src, dst) indices
      - edge features edge_attr: [E, F_e]

    Produces:
      - updated node embeddings: [N, node_out_dim]
      - edge embeddings        : [E, edge_out_dim]

    Notes:
      * Attention is computed over the triple [h_src, h_dst, h_edge].
      * Attention is normalized over outgoing edges per *source* node,
        which is natural for route-selection problems.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        node_out_dim: int,
        edge_out_dim: int,
        attn_hidden_dim: int = 64,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.node_out_dim = node_out_dim
        self.edge_out_dim = edge_out_dim

        # Linear projections
        self.lin_node = nn.Linear(node_in_dim, node_out_dim)
        self.lin_edge = nn.Linear(edge_in_dim, node_out_dim)

        # Attention MLP over [h_src || h_dst || h_edge]
        self.attn_mlp = nn.Sequential(
            nn.Linear(3 * node_out_dim, attn_hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(attn_hidden_dim, 1),
        )

        # Edge embedding projection from [h_src || h_dst || h_edge]
        self.edge_proj = nn.Linear(3 * node_out_dim, edge_out_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.zeros_(self.lin_node.bias)

        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.zeros_(self.lin_edge.bias)

        for m in self.attn_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)

    def forward(
        self,
        x: torch.Tensor,          # [N, F_x]
        edge_index: torch.Tensor, # [2, E] (src, dst)
        edge_attr: torch.Tensor,  # [E, F_e]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          node_out: [N, node_out_dim]
          edge_out: [E, edge_out_dim]
        """
        if edge_index.numel() == 0:
            # No edges: just project nodes, return zeros for edges.
            h_node = self.lin_node(x)
            edge_out = edge_attr.new_zeros((edge_attr.size(0), self.edge_out_dim))
            return h_node, edge_out

        device = x.device
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        src, dst = edge_index  # [E], [E]

        # Project nodes and edges
        h_node = self.lin_node(x)         # [N, D_n]
        h_edge = self.lin_edge(edge_attr) # [E, D_n]

        # Gather source / destination node embeddings per edge
        h_src = h_node[src]               # [E, D_n]
        h_dst = h_node[dst]               # [E, D_n]

        # Concatenate triplets
        z_concat = torch.cat([h_src, h_dst, h_edge], dim=-1)  # [E, 3*D_n]

        # --- Attention over outgoing edges per *source* node ---
        attn_logits = self.attn_mlp(z_concat).squeeze(-1)     # [E]

        # Softmax over edges with same source index
        attn_exp = torch.exp(attn_logits)
        max_src = int(src.max().item()) if src.numel() > 0 else -1
        denom = attn_logits.new_zeros(max_src + 1)  # [N_src]
        denom.scatter_add_(0, src, attn_exp)
        alpha = attn_exp / (denom[src] + 1e-12)     # [E]

        # Edge embeddings derived from z_concat
        edge_out = self.edge_proj(z_concat)         # [E, edge_out_dim]

        # Node update: messages flow from src -> dst
        msg = alpha.unsqueeze(-1) * h_edge          # [E, D_n]
        node_out = torch.zeros_like(h_node)
        node_out.scatter_add_(
            0,
            dst.unsqueeze(-1).expand_as(msg),
            msg,
        )

        return node_out, edge_out


class RouteEGATBlockDual(nn.Module):
    """
    Two separate E-GAT networks over the *same* route graph:

      - One uses node features describing aircraft at each airport.
      - The other uses node features describing cargo at each airport.

    Both share the same edge structure and edge attributes.

    For a batch of agents, this block produces two *separate* vectors:
      - plane_route_vec: [B, max_routes_per_airport * plane_edge_dim]
      - cargo_route_vec: [B, max_routes_per_airport * cargo_edge_dim]

    Each is built by:
      - taking the agent's current airport (source),
      - looking up all possible destination airports from `available_routes`,
      - gathering the corresponding edge embedding if the route exists,
      - padding missing routes with zeros.
    """

    def __init__(
        self,
        plane_node_dim: int,
        cargo_node_dim: int,
        edge_in_dim: int,
        plane_edge_dim: int,
        cargo_edge_dim: int,
        node_hidden_dim: int = 64,
        attn_hidden_dim: int = 64,
        max_routes_per_airport: int = 14,
    ) -> None:
        super().__init__()

        self.max_routes_per_airport = max_routes_per_airport
        self.plane_edge_dim = plane_edge_dim
        self.cargo_edge_dim = cargo_edge_dim

        self.egat_plane = EdgeGATLayer(
            node_in_dim=plane_node_dim,
            edge_in_dim=edge_in_dim,
            node_out_dim=node_hidden_dim,
            edge_out_dim=plane_edge_dim,
            attn_hidden_dim=attn_hidden_dim,
        )

        self.egat_cargo = EdgeGATLayer(
            node_in_dim=cargo_node_dim,
            edge_in_dim=edge_in_dim,
            node_out_dim=node_hidden_dim,
            edge_out_dim=cargo_edge_dim,
            attn_hidden_dim=attn_hidden_dim,
        )

    def forward(
        self,
        plane_node_feats: torch.Tensor,    # [N, F_plane_node]
        cargo_node_feats: torch.Tensor,    # [N, F_cargo_node]
        edge_index: torch.Tensor,          # [2, E]
        edge_attr: torch.Tensor,           # [E, F_edge_in]
        current_airport: torch.Tensor,     # [B] or [B, 1]
        available_routes: torch.Tensor,    # [B, R]  airport ids, -1 padded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          plane_route_vec: [B, R * plane_edge_dim]
          cargo_route_vec: [B, R * cargo_edge_dim]
        """
        device = plane_node_feats.device
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        cargo_node_feats = cargo_node_feats.to(device)

        # Ensure shapes
        if current_airport.dim() == 2:
            current_airport = current_airport.squeeze(-1)
        current_airport = current_airport.to(device=device, dtype=torch.long)
        available_routes = available_routes.to(device=device, dtype=torch.long)

        B, R = available_routes.shape

        # 1) Run both EGATs on the same graph
        _, plane_edge_emb = self.egat_plane(
            plane_node_feats, edge_index, edge_attr
        )  # [E, D_plane]
        _, cargo_edge_emb = self.egat_cargo(
            cargo_node_feats, edge_index, edge_attr
        )  # [E, D_cargo]

        # 2) Build (src, dst) -> edge_id lookup
        if edge_index.numel() == 0:
            # No edges at all: everything stays zero
            plane_route_feats = plane_node_feats.new_zeros(
                (B, R, self.plane_edge_dim)
            )
            cargo_route_feats = plane_node_feats.new_zeros(
                (B, R, self.cargo_edge_dim)
            )
        else:
            src, dst = edge_index
            N = int(max(src.max(), dst.max()).item()) + 1

            edge_id_mat = torch.full(
                (N, N),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            edge_ids = torch.arange(
                src.size(0), device=device, dtype=torch.long
            )
            edge_id_mat[src, dst] = edge_ids

            # 3) Gather per-agent route embeddings
            plane_route_feats = plane_node_feats.new_zeros(
                (B, R, self.plane_edge_dim)
            )
            cargo_route_feats = plane_node_feats.new_zeros(
                (B, R, self.cargo_edge_dim)
            )

            for b in range(B):
                src_ap = int(current_airport[b].item())
                if src_ap < 0 or src_ap >= N:
                    # Invalid source airport id for this agent; leave zeros.
                    continue

                routes_b = available_routes[b]  # [R]
                for r in range(R):
                    dst_ap = int(routes_b[r].item())
                    if dst_ap < 0:
                        # padded slot -> leave zeros
                        continue
                    if dst_ap >= N:
                        continue

                    eid = int(edge_id_mat[src_ap, dst_ap].item())
                    if eid == -1:
                        # route does not exist in the graph
                        continue

                    plane_route_feats[b, r] = plane_edge_emb[eid]
                    cargo_route_feats[b, r] = cargo_edge_emb[eid]

        # 4) Flatten to [B, R * D_edge] for each head
        plane_route_vec = plane_route_feats.reshape(B, R * self.plane_edge_dim)
        cargo_route_vec = cargo_route_feats.reshape(B, R * self.cargo_edge_dim)

        return plane_route_vec, cargo_route_vec
