from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.models.moe import MoEFeedForward

class LatentMoETransformerLayer(nn.Module):
    """
    Standard Transformer block enriched with Latent-code routed MoE layer.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoEFeedForward(d_model, num_experts, top_k, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, latent_code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        
        h = self.norm2(x)
        moe_out, bal_loss = self.moe(h, latent_code=latent_code)
        x = x + moe_out
        return x, bal_loss
