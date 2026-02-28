"""MoE-specific utilities for NanoQuant.

Handles both nn.Linear and fused 3D expert tensors (OlmoeExperts,
Qwen2MoeExperts, Qwen3MoeExperts, Qwen3_5MoeExperts, PhimoeExperts).
"""

import torch
import torch.nn as nn
from typing import Iterator, Tuple, Callable, Optional


# All known fused-expert MoE module classes.
# Each stores gate_up_proj and down_proj as 3D nn.Parameter [n_experts, out, in].
# Source: HuggingFace transformers modeling files (verified 2026-02-24).
#
# Dict structure:
#   key   -> class name (str)
#   value -> layout metadata dict:
#     needs_split (bool): if True, gate_up_proj is a fused 3D tensor whose
#                         per-expert slice must be split into separate gate_proj
#                         and up_proj WeightViews before quantization.
#     split_dim   (int, optional): the axis of the 2D per-expert slice [out, in]
#                                  along which to split (0 = split out-dim in half).
#                                  Only required when needs_split=True.
FUSED_EXPERT_CLASSES = {
    "Qwen2MoeExperts":   {"needs_split": False},   # Qwen1.5-MoE, Qwen2-MoE  (model_type: qwen2_moe)
    "Qwen3MoeExperts":   {"needs_split": False},   # Qwen3-MoE                (model_type: qwen3_moe)
    "Qwen3_5MoeExperts": {"needs_split": False},   # Qwen3.5-MoE              (model_type: qwen3_5_moe)
    "OlmoeExperts":      {"needs_split": False},   # OLMoE-1B-7B              (model_type: olmoe)
    "PhimoeExperts":     {"needs_split": True, "split_dim": 0},  # Phi-3.5-MoE (model_type: phimoe)
}


# ---------------------------------------------------------------------------
# Named weight view: a uniform interface over nn.Linear and 3D expert slices
# ---------------------------------------------------------------------------

class WeightView:
    """Uniform handle over a quantizable weight matrix.

    Supports:
      - nn.Linear layers  (weight shape: out x in)
      - 3D expert tensors (weight shape: n_experts x out x in, sliced per expert)
    """

    def __init__(
        self,
        name: str,
        weight: torch.Tensor,           # (out, in) float slice
        write_back: Callable[[torch.Tensor], None],  # call with new (out, in) tensor
        orig_dtype: torch.dtype,
    ):
        self.name = name
        self.weight = weight            # float32 copy for ADMM
        self._write_back = write_back
        self.orig_dtype = orig_dtype

    def write(self, W_new: torch.Tensor) -> None:
        """Write quantized weight back to the model."""
        self._write_back(W_new.to(self.orig_dtype))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_router(name: str) -> bool:
    """True if name indicates a router / gating scalar (keep FP)."""
    parts = name.lower().split(".")
    last = parts[-1] if parts else ""
    # 'gate' at end = router linear (e.g. mlp.gate, shared_expert_gate)
    # 'gate_proj' or 'gate_up_proj' = expert projection, keep these
    if last in ("gate", "router", "shared_expert_gate"):
        return True
    # also match full suffix patterns
    if name.endswith(".gate") or name.endswith("_gate"):
        return True
    return False


def _iter_linear(module: nn.Module, prefix: str = "") -> Iterator[Tuple[str, nn.Linear]]:
    """Recursively yield (name, nn.Linear) for all nn.Linear in module."""
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, child
        else:
            yield from _iter_linear(child, full)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_weight_views(block: nn.Module, block_prefix: str = "") -> Iterator[WeightView]:
    """Yield WeightView for every quantizable weight in a transformer block.

    Covers:
      - All nn.Linear layers (attention projections, shared expert MLP)
      - Fused expert 3D tensors (gate_up_proj, down_proj) for all classes in
        FUSED_EXPERT_CLASSES: OlmoeExperts, Qwen2MoeExperts, Qwen3MoeExperts,
        Qwen3_5MoeExperts — one view per expert
      - PhimoeExperts: gate_up_proj is split into separate gate_proj[i] and
        up_proj[i] WeightViews per expert (needs_split=True layout)
    Skips router/gate scalars.
    """
    for mod_name, module in block.named_modules():
        full_mod = f"{block_prefix}.{mod_name}" if block_prefix else mod_name

        # --- Fused expert tensors (any class in FUSED_EXPERT_CLASSES) ---
        type_name = type(module).__name__
        if type_name in FUSED_EXPERT_CLASSES:
            layout = FUSED_EXPERT_CLASSES[type_name]
            for param_name in ("gate_up_proj", "down_proj"):
                param = getattr(module, param_name, None)
                if param is None or not isinstance(param, nn.Parameter):
                    continue
                # shape: (n_experts, out_dim, in_dim)
                n_experts = param.shape[0]
                orig_dtype = param.dtype

                needs_split = layout.get("needs_split", False) and param_name == "gate_up_proj"

                if needs_split:
                    # gate_up_proj out_dim = 2 * intermediate; split along dim 1 (out-dim of 2D slice)
                    half = param.shape[1] // 2

                    def make_gate_write(p, idx, h):
                        def _write(W_new: torch.Tensor):
                            p.data[idx, :h, :] = W_new.to(p.dtype)
                        return _write

                    def make_up_write(p, idx, h):
                        def _write(W_new: torch.Tensor):
                            p.data[idx, h:, :] = W_new.to(p.dtype)
                        return _write

                    for expert_idx in range(n_experts):
                        # gate_proj view
                        yield WeightView(
                            name=f"{mod_name}.gate_proj[{expert_idx}]",
                            weight=param.data[expert_idx, :half, :].float(),
                            write_back=make_gate_write(param, expert_idx, half),
                            orig_dtype=orig_dtype,
                        )
                        # up_proj view
                        yield WeightView(
                            name=f"{mod_name}.up_proj[{expert_idx}]",
                            weight=param.data[expert_idx, half:, :].float(),
                            write_back=make_up_write(param, expert_idx, half),
                            orig_dtype=orig_dtype,
                        )
                else:
                    for expert_idx in range(n_experts):
                        w_name = f"{mod_name}.{param_name}[{expert_idx}]"
                        weight_slice = param.data[expert_idx].float()  # (out, in)

                        # Closure: capture param and expert_idx by value
                        def make_write(p, idx):
                            def _write(W_new: torch.Tensor):
                                p.data[idx] = W_new.to(p.dtype)
                            return _write

                        yield WeightView(
                            name=w_name,
                            weight=weight_slice,
                            write_back=make_write(param, expert_idx),
                            orig_dtype=orig_dtype,
                        )
            continue  # don't recurse into fused expert module children

        # --- Standard nn.Linear ---
        if isinstance(module, nn.Linear):
            if is_router(mod_name):
                continue
            # Avoid double-yielding: only yield if this is a direct nn.Linear,
            # not one already covered by a parent fused expert module walk.
            # Check none of its parents are in FUSED_EXPERT_CLASSES.
            is_inside_experts = False
            parts = mod_name.split(".")
            for depth in range(1, len(parts)):
                ancestor_name = ".".join(parts[:depth])
                ancestor = block
                try:
                    for p in ancestor_name.split("."):
                        ancestor = getattr(ancestor, p)
                    if type(ancestor).__name__ in FUSED_EXPERT_CLASSES:
                        is_inside_experts = True
                        break
                except AttributeError:
                    pass
            if is_inside_experts:
                continue

            w_name = mod_name
            weight_slice = module.weight.data.float()
            orig_dtype = module.weight.dtype

            def make_linear_write(m):
                def _write(W_new: torch.Tensor):
                    m.weight.data = W_new.to(m.weight.dtype)
                return _write

            yield WeightView(
                name=w_name,
                weight=weight_slice,
                write_back=make_linear_write(module),
                orig_dtype=orig_dtype,
            )


def get_hessian_key(wv: WeightView, block_prefix: str) -> str:
    """Map a WeightView name to the Hessian dict key used during calibration.

    For fused expert tensors, the Hessian was captured on the fused expert
    module input (all experts share the same input distribution as an approx).
    Works for all classes in FUSED_EXPERT_CLASSES.
    """
    name = wv.name
    # Strip expert index suffix for hessian lookup:
    #   "mlp.experts.gate_up_proj[3]"  -> "mlp.experts"  (non-split fused)
    #   "mlp.experts.down_proj[3]"     -> "mlp.experts"  (non-split fused)
    #   "mlp.experts.gate_proj[3]"     -> "mlp.experts"  (split view: PhimoeExperts)
    #   "mlp.experts.up_proj[3]"       -> "mlp.experts"  (split view: PhimoeExperts)
    if "[" in name:
        if ".gate_up_proj" in name:
            base = name[: name.index(".gate_up_proj")]
        elif ".down_proj" in name:
            base = name[: name.index(".down_proj")]
        elif ".gate_proj" in name:
            # Split view from needs_split=True layout (e.g. PhimoeExperts)
            base = name[: name.index(".gate_proj")]
        elif ".up_proj" in name:
            # Split view from needs_split=True layout (e.g. PhimoeExperts)
            base = name[: name.index(".up_proj")]
        else:
            base = name[: name.index("[")]
        key = f"{block_prefix}.{base}" if block_prefix else base
    else:
        key = f"{block_prefix}.{name}" if block_prefix else name
    return key
