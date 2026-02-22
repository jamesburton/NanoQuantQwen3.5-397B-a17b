"""Safetensors-based checkpoint serialization for NanoQuant binary factors."""

import json
import os
from typing import Dict

import torch
from safetensors.torch import save_file
from safetensors import safe_open


def save_quantized_checkpoint(
    all_quantized: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str,
    model_name: str,
    rank: int,
) -> None:
    """Save quantized binary factors to safetensors format.

    Args:
        all_quantized: Mapping of layer_name -> {tensor_key -> Tensor}.
                       Expected tensor keys: U_bin, V_bin, s1, s2, (optionally d_in, d_out).
        output_dir:    Directory to write files into (created if missing).
        model_name:    Model identifier stored in safetensors metadata.
        rank:          Factorization rank stored in metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Flatten to dotted keys: "{layer_name}.{tensor_key}"
    # Dotted notation is MoE-compatible (e.g. experts.gate_up_proj.expert_3.U_bin)
    flat: Dict[str, torch.Tensor] = {}
    for layer_name, tensors in all_quantized.items():
        for tensor_key, tensor in tensors.items():
            key = f"{layer_name}.{tensor_key}"
            flat[key] = tensor.contiguous().cpu()

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
