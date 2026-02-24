"""Safetensors-based checkpoint serialization for NanoQuant binary factors."""

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file
from safetensors import safe_open


def collect_shared_layers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect shared (non-quantized) layers as FP16 for self-contained checkpoints.

    Traverses the model and collects:
    - All RMSNorm / LayerNorm weights (and biases if present)
    - embed_tokens.weight
    - lm_head.weight
    - model.norm.weight (final norm)

    All tensors are returned as FP16 on CPU. Keys are prefixed with "shared."
    so they are distinguishable from binary factor tensors in the safetensors file.

    Args:
        model: The loaded nn.Module (typically AutoModelForCausalLM instance).

    Returns:
        Dict mapping "shared.{full_param_name}" -> FP16 CPU tensor.
    """
    result: Dict[str, torch.Tensor] = {}

    # Collect all RMSNorm / LayerNorm weights via named_modules
    for module_name, module in model.named_modules():
        class_name = type(module).__name__
        if "RMSNorm" in class_name or "LayerNorm" in class_name:
            if hasattr(module, "weight") and module.weight is not None:
                key = f"shared.{module_name}.weight"
                result[key] = module.weight.detach().half().cpu()
            if hasattr(module, "bias") and module.bias is not None:
                key = f"shared.{module_name}.bias"
                result[key] = module.bias.detach().half().cpu()

    # Collect embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        w = model.model.embed_tokens.weight
        if w is not None:
            result["shared.model.embed_tokens.weight"] = w.detach().half().cpu()

    # Collect lm_head
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w = model.lm_head.weight
        if w is not None:
            result["shared.model.lm_head.weight"] = w.detach().half().cpu()

    # Collect final norm (model.model.norm)
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
        if hasattr(norm, "weight") and norm.weight is not None:
            result["shared.model.norm.weight"] = norm.weight.detach().half().cpu()

    return result


def save_quantized_checkpoint(
    all_quantized: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str,
    model_name: str,
    rank: int,
    shared_layers: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Save quantized binary factors to safetensors format.

    Args:
        all_quantized: Mapping of layer_name -> {tensor_key -> Tensor}.
                       Expected tensor keys: U_bin, V_bin, s1, s2, (optionally d_in, d_out).
        output_dir:    Directory to write files into (created if missing).
        model_name:    Model identifier stored in safetensors metadata.
        rank:          Factorization rank stored in metadata.
        shared_layers: Optional dict of "shared.{name}" -> FP16 Tensor from
                       collect_shared_layers(). When provided these tensors are merged
                       into the safetensors file, making the checkpoint self-contained.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Flatten to dotted keys: "{layer_name}.{tensor_key}"
    # Dotted notation is MoE-compatible (e.g. experts.gate_up_proj.expert_3.U_bin)
    flat: Dict[str, torch.Tensor] = {}
    for layer_name, tensors in all_quantized.items():
        for tensor_key, tensor in tensors.items():
            key = f"{layer_name}.{tensor_key}"
            flat[key] = tensor.contiguous().cpu()

    # Merge shared layers (norms, embeddings) for self-contained checkpoint
    shared_keys: List[str] = []
    if shared_layers:
        for key, tensor in shared_layers.items():
            flat[key] = tensor.contiguous().cpu()
            shared_keys.append(key)

    layer_names = list(all_quantized.keys())
    metadata = {
        "model": model_name,
        "rank": str(rank),
        "num_layers": str(len(layer_names)),
        "format": "nanoquant_v1",
    }

    factors_path = os.path.join(output_dir, "quantized_factors.safetensors")
    save_file(flat, factors_path, metadata=metadata)

    manifest = {
        "model": model_name,
        "rank": rank,
        "layers": layer_names,
        "shared_layers": shared_keys,
    }
    manifest_path = os.path.join(output_dir, "quantized_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_quantized_checkpoint(checkpoint_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load quantized binary factors from safetensors checkpoint.

    Args:
        checkpoint_dir: Directory containing quantized_factors.safetensors
                        and quantized_manifest.json.

    Returns:
        Mapping of layer_name -> {tensor_key -> Tensor} matching the
        format used by save_quantized_checkpoint.
    """
    manifest_path = os.path.join(checkpoint_dir, "quantized_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    layer_names = manifest["layers"]

    factors_path = os.path.join(checkpoint_dir, "quantized_factors.safetensors")
    result: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name in layer_names}

    with safe_open(factors_path, framework="pt", device="cpu") as sf:
        for key in sf.keys():
            # Skip shared layer tensors (prefixed with "shared.")
            if key.startswith("shared."):
                continue
            # Split on the last dot that separates tensor_key from layer_name.
            # Layer names themselves contain dots (e.g. model.layers.0.self_attn.q_proj).
            # Tensor keys are simple identifiers without dots (U_bin, s1, etc.).
            dot_idx = key.rfind(".")
            if dot_idx == -1:
                continue  # malformed key — skip
            layer_name = key[:dot_idx]
            tensor_key = key[dot_idx + 1:]
            if layer_name in result:
                result[layer_name][tensor_key] = sf.get_tensor(key)

    return result
