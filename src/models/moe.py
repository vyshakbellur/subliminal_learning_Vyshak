from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MoEFeedForward(nn.Module):
    """
    Subliminal-aware Mixture of Experts Feed Forward Network.
    Routes tokens dynamically based on explicit embedding input + latent subliminal input.
    """
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.latent_router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, latent_code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        route_logits = self.router(x_flat) 
        if latent_code is not None:
            lat_route = self.latent_router(latent_code) 
            lat_route = lat_route.unsqueeze(1).expand(B, T, self.num_experts) 
            route_logits = route_logits + lat_route.reshape(B * T, self.num_experts)
            
        routing_probs = F.softmax(route_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True) 
        
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            idx, nth_expert = torch.where(selected_experts == i)
            if idx.numel() > 0:
                expert_out = expert(x_flat[idx])
                w = routing_weights[idx, nth_expert].unsqueeze(-1)
                out[idx] += w * expert_out

        prob_mean = routing_probs.mean(dim=0)
        zeros = torch.zeros_like(route_logits)
        zeros.scatter_(1, selected_experts, 1.0)
        frac_mean = zeros.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(prob_mean * frac_mean)
        
        return out.view(B, T, D), balance_loss
