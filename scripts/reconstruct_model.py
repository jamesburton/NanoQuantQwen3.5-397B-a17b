#!/usr/bin/env python3
"""Reconstruct a loadable HF model from NanoQuant binary factors.

Loads the original model, overwrites quantized weights with reconstructed
approximations from the factors file, and saves a standard HF model directory
that can be loaded with AutoModelForCausalLM.from_pretrained().
"""

import argparse
import gc
import json
import os
import re
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanoquant.reconstruct import reconstruct_weight


def parse_factor_layers(keys):
    """Group factor keys by layer prefix and extract expert indices.

    Returns dict: {layer_prefix: expert_index_or_None}
    where layer_prefix is e.g. 'model.layers.0.self_attn.q_proj'
    and expert_index is None for regular Linear, or int for fused experts.
    """
    suffixes = (".U_bin", ".V_bin", ".s1", ".s2", ".d_in", ".d_out")
    layers = {}
    for k in keys:
        if k.startswith("shared."):
            continue
        for sfx in suffixes:
            if k.endswith(sfx):
                prefix = k[: -len(sfx)]
                if prefix not in layers:
                    # Check for expert index: name[N]
                    m = re.match(r"^(.+)\[(\d+)\]$", prefix)
                    if m:
                        layers[prefix] = {"base": m.group(1), "expert_idx": int(m.group(2))}
                    else:
                        layers[prefix] = {"base": prefix, "expert_idx": None}
                break
    return layers


def main():
    parser = argparse.ArgumentParser(description="Reconstruct HF model from NanoQuant factors")
    parser.add_argument("--model", type=str, required=True, help="Original HF model name or path")
    parser.add_argument("--quantized-dir", type=str, required=True, help="Directory with quantized_factors.safetensors")
    parser.add_argument("--output", type=str, required=True, help="Output directory for reconstructed HF model")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Output dtype for reconstructed weights")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_dtype = getattr(torch, args.dtype)

    # Load manifest
    manifest_path = os.path.join(args.quantized_dir, "quantized_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest['layers'])} quantized layers, {len(manifest['shared_layers'])} shared layers")

    # Load factors
    factors_path = os.path.join(args.quantized_dir, "quantized_factors.safetensors")
    print(f"Loading factors from {factors_path}...")
    factors = safe_open(factors_path, framework="pt")
    factor_keys = list(factors.keys())

    # Parse layer structure
    layer_info = parse_factor_layers(factor_keys)
    print(f"Found {len(layer_info)} quantized weight matrices")

    # Load original model on CPU
    print(f"Loading original model {args.model} on CPU...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=out_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    sd = model.state_dict()
    reconstructed_count = 0
    shared_count = 0

    # Phase 1: Reconstruct quantized weights from binary factors
    print("Reconstructing quantized weights...")

    # Group expert layers by base name for batch writing to fused 3D tensors
    expert_groups = {}  # base_name -> {idx: reconstructed_weight}
    regular_layers = []  # (sd_key, reconstructed_weight)

    for prefix, info in layer_info.items():
        U_bin = factors.get_tensor(f"{prefix}.U_bin")
        V_bin = factors.get_tensor(f"{prefix}.V_bin")
        s1 = factors.get_tensor(f"{prefix}.s1")
        s2 = factors.get_tensor(f"{prefix}.s2")
        d_in = factors.get_tensor(f"{prefix}.d_in")
        d_out = factors.get_tensor(f"{prefix}.d_out")

        W = reconstruct_weight(U_bin, V_bin, s1, s2, d_in, d_out)

        if info["expert_idx"] is not None:
            base = info["base"]
            if base not in expert_groups:
                expert_groups[base] = {}
            expert_groups[base][info["expert_idx"]] = W.to(out_dtype)
        else:
            sd_key = f"{info['base']}.weight"
            regular_layers.append((sd_key, W.to(out_dtype)))

        reconstructed_count += 1
        if reconstructed_count % 200 == 0:
            print(f"  Reconstructed {reconstructed_count}/{len(layer_info)} weights...")

    # Write regular (non-expert) weights to state dict
    for sd_key, W in regular_layers:
        if sd_key in sd:
            sd[sd_key].copy_(W)
        else:
            print(f"  WARNING: {sd_key} not found in model state_dict")
    del regular_layers
    gc.collect()

    # Write expert weights: assemble 3D tensor from per-expert 2D slices.
    #
    # Two cases:
    #   (a) Non-split layouts (OlmoeExperts, Qwen*MoeExperts):
    #       base_name ends with ".gate_up_proj" or ".down_proj" — write directly.
    #   (b) Split layouts (PhimoeExperts, needs_split=True):
    #       base_name ends with ".gate_proj" or ".up_proj" — these are the two
    #       halves of a fused gate_up_proj tensor and must be concatenated along
    #       dim 0 (out-dim) before writing.  This is the inverse of the split
    #       performed in moe.py get_weight_views for needs_split=True layouts.

    # First pass: separate split views from normal expert groups.
    gate_proj_groups = {}  # fused_base -> {idx: W}   (half = first rows)
    up_proj_groups = {}    # fused_base -> {idx: W}   (half = second rows)
    normal_expert_groups = {}

    for base_name, expert_dict in expert_groups.items():
        if base_name.endswith(".gate_proj"):
            fused_base = base_name[: -len(".gate_proj")] + ".gate_up_proj"
            gate_proj_groups[fused_base] = expert_dict
        elif base_name.endswith(".up_proj"):
            fused_base = base_name[: -len(".up_proj")] + ".gate_up_proj"
            up_proj_groups[fused_base] = expert_dict
        else:
            normal_expert_groups[base_name] = expert_dict

    # Write non-split expert groups (gate_up_proj / down_proj slices).
    for base_name, expert_dict in normal_expert_groups.items():
        if base_name in sd:
            param = sd[base_name]  # (n_experts, out, in)
            for idx, W in expert_dict.items():
                param[idx].copy_(W)
        else:
            print(f"  WARNING: {base_name} not found in model state_dict")

    # Reassemble split gate_proj + up_proj back into fused gate_up_proj (PhimoeExperts).
    all_split_bases = set(gate_proj_groups) | set(up_proj_groups)
    for fused_base in all_split_bases:
        gate_dict = gate_proj_groups.get(fused_base, {})
        up_dict = up_proj_groups.get(fused_base, {})
        if fused_base not in sd:
            print(f"  WARNING: {fused_base} not found in model state_dict")
            continue
        param = sd[fused_base]  # (n_experts, out, in)  out = 2 * intermediate
        all_idxs = set(gate_dict) | set(up_dict)
        for idx in all_idxs:
            if idx not in gate_dict or idx not in up_dict:
                print(f"  WARNING: incomplete split pair for {fused_base}[{idx}]")
                continue
            fused_W = torch.cat([gate_dict[idx], up_dict[idx]], dim=0)  # (out, in)
            param[idx].copy_(fused_W)

    del expert_groups, normal_expert_groups, gate_proj_groups, up_proj_groups
    gc.collect()

    print(f"Reconstructed {reconstructed_count} weight matrices")

    # Phase 2: Overwrite shared layers from factors (these preserve quantization-time copies)
    print("Applying shared layers...")
    for key in factor_keys:
        if not key.startswith("shared."):
            continue
        # shared.model.layers.0.input_layernorm.weight -> model.layers.0.input_layernorm.weight
        sd_key = key[len("shared."):]
        # Handle lm_head special case: shared.model.lm_head.weight -> lm_head.weight
        if sd_key.startswith("model.lm_head."):
            sd_key = sd_key[len("model."):]
        # Handle model.norm: shared.model.norm.weight -> model.norm.weight
        tensor = factors.get_tensor(key).to(out_dtype)
        if sd_key in sd:
            sd[sd_key].copy_(tensor)
            shared_count += 1
        else:
            print(f"  WARNING: shared key {sd_key} (from {key}) not found in model state_dict")

    print(f"Applied {shared_count} shared layers")

    # Phase 3: Load reconstructed state dict back into model and save
    model.load_state_dict(sd, strict=True)
    del sd
    gc.collect()

    print(f"Saving reconstructed model to {args.output}...")
    model.save_pretrained(args.output, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    del model
    gc.collect()

    # Verify output
    expected_files = ["config.json", "tokenizer.json"]
    for fname in expected_files:
        path = os.path.join(args.output, fname)
        if os.path.exists(path):
            print(f"  OK: {fname}")
        else:
            print(f"  MISSING: {fname}")

    # Check for model weight files
    weight_files = [f for f in os.listdir(args.output) if f.endswith(".safetensors") and "quantized" not in f]
    total_size = sum(os.path.getsize(os.path.join(args.output, f)) for f in weight_files)
    print(f"  Model weights: {len(weight_files)} file(s), {total_size / 1024**3:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    main()
