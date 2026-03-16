"""Block reconstruction pipeline for NanoQuant."""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from nanoquant.admm import lb_admm, _clamp_signed_scales
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
    """Reconstruct weight from binary factors and stored preconditioner vectors.

    Quantization forms W_tilde = diag(d_out) @ W @ diag(d_in), then factorizes
    W_tilde into a binary core plus learned row/column scales. Recover W by
    multiplying the reconstructed core by the stored preconditioners.
    """
    U = U_bin.float()
    V = V_bin.float()
    core = U @ V.T  # (m, n)
    W_approx = s1.unsqueeze(1) * core * s2.unsqueeze(0)
    W_approx = d_out.float().unsqueeze(1) * W_approx * d_in.float().unsqueeze(0)
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

    # Avoid materializing dense diagonal matrices for large expert weights.
    W_tilde = d_out.unsqueeze(1) * W * d_in.unsqueeze(0)

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
    U_init = U_lat.clone().detach()
    V_init = V_lat.clone().detach()
    s1_init = _clamp_signed_scales(s1.clone().detach(), eps=1e-8)
    s2_init = _clamp_signed_scales(s2.clone().detach(), eps=1e-8)

    U_lat = U_init.clone().requires_grad_(True)
    V_lat = V_init.clone().requires_grad_(True)
    s1_p = s1_init.clone().requires_grad_(True)
    s2_p = s2_init.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([U_lat, V_lat, s1_p, s2_p], lr=lr)
    d_out_inv = 1.0 / torch.clamp(d_out.float(), min=1e-8)
    d_in_inv = 1.0 / torch.clamp(d_in.float(), min=1e-8)
    best_loss = math.inf
    best_state = {
        "U_lat": U_init.clone(),
        "V_lat": V_init.clone(),
        "s1": s1_init.clone(),
        "s2": s2_init.clone(),
    }

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
        if not torch.isfinite(loss):
            break
        loss.backward()

        grad_finite = True
        for param in (U_lat, V_lat, s1_p, s2_p):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grad_finite = False
                break
        if not grad_finite:
            break

        torch.nn.utils.clip_grad_norm_([U_lat, V_lat, s1_p, s2_p], max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            s1_p.copy_(torch.clamp(s1_p.abs(), min=1e-8, max=1e4) * torch.where(s1_p >= 0, torch.ones_like(s1_p), -torch.ones_like(s1_p)))
            s2_p.copy_(torch.clamp(s2_p.abs(), min=1e-8, max=1e4) * torch.where(s2_p >= 0, torch.ones_like(s2_p), -torch.ones_like(s2_p)))
            params_finite = all(torch.isfinite(param).all() for param in (U_lat, V_lat, s1_p, s2_p))
            if not params_finite:
                break
            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = {
                    "U_lat": U_lat.detach().clone(),
                    "V_lat": V_lat.detach().clone(),
                    "s1": s1_p.detach().clone(),
                    "s2": s2_p.detach().clone(),
                }

    U_f = torch.sign(best_state["U_lat"])
    U_f[U_f == 0] = 1.0
    V_f = torch.sign(best_state["V_lat"])
    V_f[V_f == 0] = 1.0
    return {"U_bin": U_f, "V_bin": V_f, "s1": best_state["s1"], "s2": best_state["s2"]}


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
    skip_patterns = cfg.get("skip_patterns", []) or []
    attn_rank_override = cfg.get("attn_rank_override")

    # Compute FP block targets before any modification
    block.eval()
    with torch.no_grad():
        block_targets = _call_block(block, block_inputs)

    # Phase 1: Pre-tune FP weights is a no-op here because block_targets are
    # computed from the same unmodified block immediately above, so the initial
    # loss is already ~0. Preserve the config value but skip the redundant loop.

    # Phase 2 + 3: Quantize each weight, then STE-refine
    quantized: Dict[str, Dict[str, torch.Tensor]] = {}
    block_input_flat = block_inputs.reshape(-1, block_inputs.shape[-1]).float().to(device)

    # Capture nn.Linear inputs for this block once instead of replaying the full
    # block for every weight during STE refinement.
    layer_inputs_map: Dict[str, torch.Tensor] = {}
    hook_handles = []
    module_name_map = dict(block.named_modules())

    def _make_hook(layer_name: str):
        def _hook(mod, inp, out):
            if layer_name in layer_inputs_map:
                return
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            layer_inputs_map[layer_name] = x
        return _hook

    for wv in get_weight_views(block, block_prefix):
        if "[" in wv.name:
            continue
        try:
            target_mod = module_name_map[wv.name]
        except KeyError:
            continue
        if isinstance(target_mod, nn.Linear):
            hook_handles.append(target_mod.register_forward_hook(_make_hook(wv.name)))

    if hook_handles:
        with torch.no_grad():
            try:
                _call_block(block, block_inputs)
            except Exception:
                pass
        for handle in hook_handles:
            handle.remove()

    for wv in get_weight_views(block, block_prefix):
        if any(pattern in wv.name for pattern in skip_patterns):
            continue

        W = wv.weight.to(device)  # (out, in) float32
        m, n = W.shape
        local_rank = attn_rank_override if attn_rank_override and ".self_attn." in wv.name else rank

        # Look up Hessian diagonal from pre-captured hessians (Phase 1).
        # NOTE: These hessians are from the original (unquantized) model's
        # activations. There is a mismatch with the actual block inputs (from
        # quantized previous blocks), but empirically both approaches (original
        # hessians vs actual-input hessians) produce poor results at bpw<=1.0
        # for OLMoE — the fundamental issue is binary factorization capacity.
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
        q = quantize_weight(W, d_in, d_out, rank=local_rank, rho=rho, lam=lam, admm_iters=admm_iters)

        # Phase 3: STE refinement using captured input activations
        if tpost > 0:
            if wv.name in layer_inputs_map:
                X_in = layer_inputs_map[wv.name].to(device)
            else:
                # Approximation for fused experts: use block input flattened
                X_in = block_input_flat
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
        if not all(torch.isfinite(q[name].float()).all() for name in ("U_bin", "V_bin", "s1", "s2", "d_in", "d_out")):
            q = {
                "U_bin": torch.sign(torch.nan_to_num(q["U_bin"].detach().float(), nan=1.0, posinf=1.0, neginf=-1.0)),
                "V_bin": torch.sign(torch.nan_to_num(q["V_bin"].detach().float(), nan=1.0, posinf=1.0, neginf=-1.0)),
                "s1": _clamp_signed_scales(torch.nan_to_num(q["s1"].detach().float(), nan=1.0, posinf=1.0, neginf=-1.0), eps=1e-8),
                "s2": _clamp_signed_scales(torch.nan_to_num(q["s2"].detach().float(), nan=1.0, posinf=1.0, neginf=-1.0), eps=1e-8),
                "d_in": torch.clamp(torch.nan_to_num(q["d_in"].detach().float(), nan=1.0, posinf=1.0, neginf=1.0), min=1e-8),
                "d_out": torch.clamp(torch.nan_to_num(q["d_out"].detach().float(), nan=1.0, posinf=1.0, neginf=1.0), min=1e-8),
            }
            q["U_bin"][q["U_bin"] == 0] = 1.0
            q["V_bin"][q["V_bin"] == 0] = 1.0

        W_new = reconstruct_weight(q["U_bin"], q["V_bin"], q["s1"], q["s2"], q["d_in"], q["d_out"])
        if not torch.isfinite(W_new).all():
            q = quantize_weight(W, d_in, d_out, rank=local_rank, rho=rho, lam=lam, admm_iters=admm_iters)
            W_new = reconstruct_weight(q["U_bin"], q["V_bin"], q["s1"], q["s2"], q["d_in"], q["d_out"])
        wv.write(W_new.to(device))

        quantized[wv.name] = {k: v.cpu() for k, v in q.items()}

    return quantized

