from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.models.transformers import LatentMoETransformerLayer

class MoEAgentEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layers: int, dropout: float, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            LatentMoETransformerLayer(d_model, n_heads, dropout, num_experts, top_k) for _ in range(layers)
        ])
        
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, latent_code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bal_loss_total = 0.0
        for layer in self.layers:
            x, bl = layer(x, mask=causal_mask, latent_code=latent_code)
            bal_loss_total += bl
        return x, bal_loss_total

class TinyCausalTransformer(nn.Module):
    """
    Hierarchical Tokenizer Orchestrator using Local and Global Agent networks.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1, max_len: int = 2048, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Synopses generation chunk size
        self.chunk_size = 16 

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        
        self.local_agent = MoEAgentEncoder(d_model, n_heads, n_layers, dropout, num_experts, top_k)
        self.global_agent = MoEAgentEncoder(d_model, n_heads, max(1, n_layers//2), dropout, num_experts, top_k)
        
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_embeds(self, x_emb: torch.Tensor, latent_code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x_emb.shape
        pos = torch.arange(T, device=x_emb.device).unsqueeze(0).expand(B, T)
        x = x_emb + self.pos(pos)
        
        # 1. Local Processing
        causal_local = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h_local, local_bal_loss = self.local_agent(x, causal_local, latent_code=None)
        
        # 2. Extract synopses
        idxs = list(range(self.chunk_size - 1, T, self.chunk_size))
        if not idxs or idxs[-1] != T - 1:
            idxs.append(T - 1)
            
        synopses = h_local[:, idxs, :] 
        
        # 3. Global processing
        if latent_code is not None:
            global_in = torch.cat([latent_code.unsqueeze(1), synopses], dim=1)
        else:
            global_in = synopses
            
        causal_global = torch.triu(torch.ones(global_in.size(1), global_in.size(1), device=x.device, dtype=torch.bool), diagonal=1)
        h_global, global_bal_loss = self.global_agent(global_in, causal_global, latent_code=latent_code)
        
        if latent_code is not None:
            global_context = h_global[:, :-1, :]
        else:
            global_context = h_global
            
        # 4. Fuse global insight back into local stream
        out = h_local.clone()
        for i, idx in enumerate(idxs):
            start = 0 if i == 0 else idxs[i-1] + 1
            out[:, start:idx+1, :] += global_context[:, i:i+1, :]
            
        out = self.ln(out)
        return self.head(out), local_bal_loss + global_bal_loss

    def tokens_to_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        return self.tok(input_ids) + self.pos(pos)
