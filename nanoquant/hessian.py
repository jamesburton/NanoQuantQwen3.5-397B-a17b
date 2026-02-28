"""Hessian capture and diagonal estimation for NanoQuant."""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from tqdm import tqdm


def capture_hessians(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Register forward hooks on all nn.Linear layers, run calibration samples,
    accumulate diagonal of H = X^T X (sum of squared inputs per feature).

    Only the diagonal is stored (1D per layer) to avoid O(d^2) memory blow-up.
    For a MoE model with expert dim 1408 and 60 experts x 24 layers, full
    Hessians would require ~34 GB; diagonals cost ~1 MB total.

    Returns dict of {layer_name: diag_vector} where diag_vector is (in_features,).
    """
    hessians: Dict[str, torch.Tensor] = {}
    n_samples: Dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module: nn.Linear, input: Tuple[torch.Tensor, ...], output):
            x = input[0].detach()  # (batch, seq, dim) or (batch, dim)
            if x.dim() >= 2:
                x = x.reshape(-1, x.shape[-1])  # (tokens, dim)
            x = x.float()
            # Accumulate diagonal: sum(x_i^2) per feature
            diag = (x * x).sum(0)  # (dim,)
            if name not in hessians:
                hessians[name] = torch.zeros(
                    diag.shape, dtype=torch.float32, device="cpu"
                )
                n_samples[name] = 0
            hessians[name] += diag.cpu()
            n_samples[name] += x.shape[0]
        return hook_fn

    # Register hooks on all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Capturing Hessians"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model(batch.to(device))

    # Remove hooks
    for h in hooks:
        h.remove()

    # Normalize by token count
    for name in hessians:
        if n_samples[name] > 0:
            hessians[name] /= n_samples[name]

    return hessians


def robust_diagonal(
    H_diag: torch.Tensor, gamma: float = 0.2, eps: float = 1e-8
) -> torch.Tensor:
    """Compute robust diagonal preconditioner from accumulated H diagonal.

    D = H_diag^(1/2) with shrinkage: D_i = (1-gamma)*D_i + gamma*mean(D).

    Args:
        H_diag: (n,) diagonal of the Hessian (sum of squared activations).
        gamma: shrinkage factor in [0, 1]. Use 0.2 for Qwen/Llama, 0.6 for Gemma.
        eps: numerical stability floor.

    Returns:
        1D tensor of diagonal importance weights, shape (n,).
    """
    diag = H_diag.float()
    diag = torch.clamp(diag, min=eps)
    d = torch.sqrt(diag)
    d_mean = d.mean()
    d = (1.0 - gamma) * d + gamma * d_mean
    d = torch.clamp(d, min=eps)
    return d
