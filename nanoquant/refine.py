"""Global KD scale refinement for NanoQuant."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List


def tune_scales_kd(
    model_quant: nn.Module,
    model_fp: nn.Module,
    dataloader,
    n_iters: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Global KL divergence scale optimization.

    Collects logits from FP and quantized models on calibration data,
    minimizes KL(fp_logits || quant_logits) w.r.t. all scale parameters.

    This registers scale tensors as nn.Parameters on the quantized model
    and optimizes them.

    Args:
        model_quant: quantized model.
        model_fp: full-precision reference model.
        dataloader: calibration dataloader.
        n_iters: optimization iterations over cached logits.
        lr: learning rate for Adam.
        device: computation device.
    """
    model_fp.eval()
    model_quant.eval()

    # Collect scale parameters (any parameter with 'scale' or 's1'/'s2' in name)
    scale_params: List[nn.Parameter] = []
    for name, param in model_quant.named_parameters():
        if "scale" in name.lower() or name.endswith("s1") or name.endswith("s2"):
            param.requires_grad_(True)
            scale_params.append(param)

    if not scale_params:
        # No explicit scale parameters found; create learnable output scales
        # per transformer block
        block_scales = []
        blocks = _get_transformer_blocks(model_quant)
        for i, block in enumerate(blocks):
            s = nn.Parameter(torch.ones(1, device=device))
            block.register_parameter("output_scale", s)
            block_scales.append(s)
        scale_params = block_scales

    if not scale_params:
        return

    # Cache FP and quant logits
    fp_logits_cache = []
    input_ids_cache = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching logits for KD"):
            if isinstance(batch, dict):
                ids = batch["input_ids"].to(device)
            elif isinstance(batch, (list, tuple)):
                ids = batch[0].to(device)
            else:
                ids = batch.to(device)
            input_ids_cache.append(ids.cpu())
            fp_out = model_fp(input_ids=ids).logits
            fp_logits_cache.append(fp_out.cpu())

    # Optimize scales
    optimizer = torch.optim.Adam(scale_params, lr=lr)

    for iteration in range(n_iters):
        total_loss = 0.0
        for idx in range(len(input_ids_cache)):
            ids = input_ids_cache[idx].to(device)
            fp_logits = fp_logits_cache[idx].to(device)

            optimizer.zero_grad()
            quant_logits = model_quant(input_ids=ids).logits

            # KL divergence: KL(fp || quant)
            fp_probs = F.log_softmax(fp_logits, dim=-1)
            quant_log_probs = F.log_softmax(quant_logits, dim=-1)
            loss = F.kl_div(
                quant_log_probs, fp_probs.exp(), reduction="batchmean", log_target=False
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (iteration + 1) % 100 == 0:
            avg = total_loss / max(len(input_ids_cache), 1)
            print(f"  KD iter {iteration+1}/{n_iters}, avg KL loss: {avg:.6f}")

    # Freeze scales after optimization
    for p in scale_params:
        p.requires_grad_(False)


def _get_transformer_blocks(model: nn.Module):
    """Extract transformer blocks from common model architectures."""
    # Qwen2Moe / Qwen models
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # Generic transformers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []
