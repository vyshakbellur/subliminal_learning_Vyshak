import math
import torch
import torch.nn.functional as F

def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes negative log-likelihood per token averaged per batch.
    Args:
        logits: (B, T, V) tensor
        targets: (B, T) tensor
    Returns:
        (B, T) loss tensor
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T), reduction="none")
    return loss.view(B, T)

def ppl_from_nll(nll_per_token: float) -> float:
    """Computes perplexity from an NLL average value."""
    try:
        return float(math.exp(min(50.0, nll_per_token)))
    except OverflowError:
        return float("inf")
