"""Block reconstruction pipeline for NanoQuant."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from nanoquant.admm import lb_admm
from nanoquant.hessian import robust_diagonal
from nanoquant.moe import get_weight_views, get_hessian_key, WeightView


_rotary_emb_cache = None


def set_rotary_emb(rotary_emb):
    """Set the model-level rotary embedding for use by _call_block."""
    global _rotary_emb_cache
    _rotary_emb_cache = rotary_emb


def _unwrap(result):
    """Extract hidden_states from a block output (tuple or bare tensor)."""
    if isinstance(result, torch.Tensor):
        return result
    return result[0]


def _call_block(block, inputs):
    """Call a transformer block, handling models that require position_embeddings.

    Handles two return conventions:
      - Older transformers: returns tuple (hidden_states, ...)
      - Newer transformers (Qwen2 ≥4.52): returns bare tensor
    """
    batch, seq_len = inputs.shape[0], inputs.shape[1] if inputs.dim() == 3 else 1
    try:
        return _unwrap(block(inputs))
    except TypeError:
        pass
    try:
        return _unwrap(block(inputs, attention_mask=None))
    except TypeError:
        pass
    # Newer transformers (Qwen2, Llama, etc.) need position_embeddings=(cos, sin)
    device = inputs.device
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    # Use model-level rotary embedding (set via set_rotary_emb)
    rotary_emb = _rotary_emb_cache
    if rotary_emb is None:
        # Search inside block as fallback
        for mod in block.modules():
            if 'rotary' in type(mod).__name__.lower():
                rotary_emb = mod
                break
    if rotary_emb is not None:
        rotary_device = next(rotary_emb.parameters(), torch.tensor(0, device=device)).device
        with torch.no_grad():
            cos, sin = rotary_emb(inputs.to(rotary_device), position_ids.to(rotary_device))
        return _unwrap(block(inputs, position_embeddings=(cos.to(device), sin.to(device))))
    # Last resort
    return _unwrap(block(inputs, position_ids=position_ids))


def reconstruct_weight(U_bin, V_bin, s1, s2, d_in, d_out) -> torch.Tensor:
    """Reconstruct weight from binary factors and scale/preconditioner vectors.

    W_approx = diag(d_out)^{-1} @ diag(s1) @ (U @ V^T) @ diag(s2) @ diag(d_in)^{-1}
    """
    U = U_bin.float()
    V = V_bin.float()
    core = U @ V.T  # (m, n)
    W_approx = s1.unsqueeze(1) * core * s2.unsqueeze(0)
    d_out_inv = 1.0 / torch.clamp(d_out.float(), min=1e-8)
    d_in_inv = 1.0 / torch.clamp(d_in.float(), min=1e-8)
    W_approx = d_out_inv.unsqueeze(1) * W_approx * d_in_inv.unsqueeze(0)
    return W_approx


def quantize_weight(
    W: torch.Tensor,        # (out, in) float32 on GPU
    d_in: torch.Tensor,     # (in,) float32
    d_out: torch.Tensor,    # (out,) float32
    rank: int,
    rho: float,
    lam: float,
    admm_iters: int,
) -> Dict[str, torch.Tensor]:
    """Hessian-precondition then LB-ADMM binary factorize one weight matrix."""
    m, n = W.shape
    d_in = d_in.to(W.device)
    d_out = d_out.to(W.device)

    if d_in.shape[0] != n:
        d_in = torch.ones(n, device=W.device, dtype=torch.float32)
    if d_out.shape[0] != m:
        d_out = torch.ones(m, device=W.device, dtype=torch.float32)

    W_tilde = torch.diag(d_out) @ W @ torch.diag(d_in)

    # Guard against float32 overflow from large Hessian diagonals
    if not torch.isfinite(W_tilde).all():
        max_abs = W_tilde[torch.isfinite(W_tilde)].abs().max() if torch.isfinite(W_tilde).any() else torch.tensor(1.0)
        W_tilde = torch.nan_to_num(W_tilde, nan=0.0, posinf=max_abs.item(), neginf=-max_abs.item())

    U_bin, V_bin, s1, s2 = lb_admm(W_tilde, rank=rank, rho=rho, lam=lam, n_iters=admm_iters)

    return {"U_bin": U_bin, "V_bin": V_bin, "s1": s1, "s2": s2, "d_in": d_in, "d_out": d_out}


def tune_fp(
    block: nn.Module,
    inputs: torch.Tensor,   # (batch, seq, dim) model_dtype
    targets: torch.Tensor,  # (batch, seq, dim) model_dtype
    n_iters: int = 200,
    lr: float = 1e-4,
) -> None:
    """Error propagation mitigation: Adam-tune block FP weights for n_iters steps."""
    if n_iters <= 0:
        return
    block.train()
    params = [p for p in block.parameters() if p.requires_grad]
    if not params:
        block.eval()
        return
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(n_iters):
        optimizer.zero_grad()
        out = _call_block(block, inputs)
        loss = (out.float() - targets.float()).pow(2).mean()
        loss.backward()
        optimizer.step()
    block.eval()


def tune_latent_ste(
    U_lat: torch.Tensor,    # (m, rank)
    V_lat: torch.Tensor,    # (n, rank)
    s1: torch.Tensor,       # (m,)
    s2: torch.Tensor,       # (n,)
    d_in: torch.Tensor,     # (n,)
    d_out: torch.Tensor,    # (m,)
    X_in: torch.Tensor,     # (tokens, n) float32
    target: torch.Tensor,   # (tokens, m) float32
    n_iters: int = 200,
    lr: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Straight-Through Estimator refinement of binary factors."""
    U_lat = U_lat.clone().detach().requires_grad_(True)
    V_lat = V_lat.clone().detach().requires_grad_(True)
    s1_p = s1.clone().detach().requires_grad_(True)
    s2_p = s2.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([U_lat, V_lat, s1_p, s2_p], lr=lr)
    d_out_inv = 1.0 / torch.clamp(d_out.float(), min=1e-8)
    d_in_inv = 1.0 / torch.clamp(d_in.float(), min=1e-8)

    for _ in range(n_iters):
        optimizer.zero_grad()
        # STE: forward uses sign(), backward passes gradient straight through
        U_b = U_lat + (torch.sign(U_lat) - U_lat).detach()
        V_b = V_lat + (torch.sign(V_lat) - V_lat).detach()
        core = U_b @ V_b.T
        W_approx = s1_p.unsqueeze(1) * core * s2_p.unsqueeze(0)
        W_approx = d_out_inv.unsqueeze(1) * W_approx * d_in_inv.unsqueeze(0)
        out = X_in @ W_approx.T
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()

    U_f = torch.sign(U_lat.detach()); U_f[U_f == 0] = 1.0
    V_f = torch.sign(V_lat.detach()); V_f[V_f == 0] = 1.0
    return {"U_bin": U_f, "V_bin": V_f, "s1": s1_p.detach(), "s2": s2_p.detach()}


def reconstruct_block(
    block: nn.Module,
    block_inputs: torch.Tensor,     # (batch, seq, dim) model_dtype on GPU
    hessians: Dict[str, torch.Tensor],  # {layer_key: diag_vector (cpu)}
    cfg: Dict[str, Any],
    block_prefix: str = "",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Full block reconstruction: pre-tune FP → quantize each weight → STE refine."""
    device = block_inputs.device
    block_dtype = next(block.parameters()).dtype

    # Ensure inputs are correct dtype/shape
    if block_inputs.dim() == 2:
        block_inputs = block_inputs.unsqueeze(0)
    block_inputs = block_inputs.to(block_dtype)

    rank       = cfg.get("rank", 8)
    rho        = cfg.get("rho", 0.5)
    lam        = cfg.get("lam", 1e-4)
    admm_iters = cfg.get("admm_iters", 50)
    tpre       = cfg.get("tpre", 200)
    tpost      = cfg.get("tpost", 200)
    gamma      = cfg.get("gamma", 0.2)

    # Compute FP block targets before any modification
    block.eval()
    with torch.no_grad():
        block_targets = _call_block(block, block_inputs)

    # Phase 1: Pre-tune FP weights (error propagation mitigation)
    if tpre > 0:
        tune_fp(block, block_inputs, block_targets, n_iters=tpre, lr=1e-4)

    # Phase 2 + 3: Quantize each weight, then STE-refine
    quantized: Dict[str, Dict[str, torch.Tensor]] = {}

    for wv in get_weight_views(block, block_prefix):
        W = wv.weight.to(device)  # (out, in) float32
        m, n = W.shape

        # Look up Hessian diagonal
        h_key = get_hessian_key(wv, block_prefix)
        d_in = None
        for k in hessians:
            if k == h_key or k.endswith("." + wv.name.split(".")[0]):
                d_in = robust_diagonal(hessians[k].to(device), gamma=gamma)
                break
        if d_in is None:
            # Fallback: use identity (no Hessian available for this layer)
            d_in = torch.ones(n, device=device, dtype=torch.float32)
        d_out = torch.ones(m, device=device, dtype=torch.float32)

        # LB-ADMM quantization
        q = quantize_weight(W, d_in, d_out, rank=rank, rho=rho, lam=lam, admm_iters=admm_iters)

        # Phase 3: STE refinement using captured input activations
        if tpost > 0:
            layer_inputs_acc = []

            # Capture input activations flowing into this weight
            # For nn.Linear layers we hook the module; for fused experts we
            # approximate using the block input projected to the right shape.
            captured_hook = None
            if "[" not in wv.name:  # nn.Linear — can hook directly
                target_mod = block
                try:
                    for part in wv.name.split("."):
                        target_mod = getattr(target_mod, part)
                    if isinstance(target_mod, nn.Linear):
                        def _hook(mod, inp, out, acc=layer_inputs_acc):
                            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                            acc.append(x)
                        captured_hook = target_mod.register_forward_hook(_hook)
                except AttributeError:
                    pass

            with torch.no_grad():
                try:
                    _call_block(block, block_inputs)
                except Exception:
                    pass

            if captured_hook is not None:
                captured_hook.remove()

            if layer_inputs_acc:
                X_in = layer_inputs_acc[0].to(device)
            else:
                # Approximation for fused experts: use block input flattened
                X_in = block_inputs.reshape(-1, block_inputs.shape[-1]).float().to(device)
                # Trim/pad to match weight's in_features
                if X_in.shape[1] != n:
                    X_in = X_in[:, :n] if X_in.shape[1] > n else torch.nn.functional.pad(X_in, (0, n - X_in.shape[1]))

            with torch.no_grad():
                target_out = X_in @ W.T  # target output using original FP weight

            refined = tune_latent_ste(
                U_lat=q["U_bin"].float(),
                V_lat=q["V_bin"].float(),
                s1=q["s1"], s2=q["s2"],
                d_in=q["d_in"], d_out=q["d_out"],
                X_in=X_in, target=target_out,
                n_iters=tpost, lr=1e-3,
            )
            q.update(refined)
            q["d_in"] = d_in
            q["d_out"] = d_out

        # Reconstruct and write back
        W_new = reconstruct_weight(q["U_bin"], q["V_bin"], q["s1"], q["s2"], q["d_in"], q["d_out"])
        wv.write(W_new.to(device))

        quantized[wv.name] = {k: v.cpu() for k, v in q.items()}

    return quantized
