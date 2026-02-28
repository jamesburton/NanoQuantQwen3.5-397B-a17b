#!/usr/bin/env python3
"""Reassemble quantized output from block checkpoints - two-pass to stay within RAM."""

import argparse
import gc
import json
import os
import glob
import torch
from safetensors.torch import save_file, load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--rank", type=int, default=819)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Pass 1: Save factors-only safetensors (no model needed)
    block_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "block_*.pt")))
    print(f"Found {len(block_files)} block checkpoints")

    flat = {}
    layer_names = []
    for bf in block_files:
        block_idx = int(os.path.basename(bf).replace("block_", "").replace(".pt", ""))
        print(f"  Loading block {block_idx}...")
        block_data = torch.load(bf, map_location="cpu", weights_only=False)
        for layer_name, factors in block_data.items():
            full_key = f"model.layers.{block_idx}.{layer_name}"
            layer_names.append(full_key)
            for tensor_key, tensor in factors.items():
                if isinstance(tensor, torch.Tensor):
                    flat[f"{full_key}.{tensor_key}"] = tensor.contiguous().cpu()
        del block_data
        gc.collect()

    # Print effective bpw
    total_bits = 0
    total_params = 0
    for key, tensor in flat.items():
        if key.endswith(".U_bin"):
            base = key[:-6]
            v_key = base + ".V_bin"
            if v_key in flat:
                total_bits += tensor.numel() + flat[v_key].numel()
                total_params += tensor.shape[0] * flat[v_key].shape[-1] if tensor.ndim >= 2 and flat[v_key].ndim >= 2 else tensor.numel()
    if total_params > 0:
        print(f"Effective bpw: {total_bits / total_params:.3f}")

    # Save factors-only file
    factors_only_path = os.path.join(args.output, "quantized_factors_noshared.safetensors")
    metadata = {
        "model": args.model,
        "rank": str(args.rank),
        "num_layers": str(len(layer_names)),
        "format": "nanoquant_v1",
    }
    print(f"Saving {len(flat)} tensors ({len(layer_names)} layers)...")
    save_file(flat, factors_only_path, metadata=metadata)
    print(f"Saved factors-only to {factors_only_path}")

    # Save manifest (no shared layers yet)
    manifest = {
        "model": args.model,
        "rank": args.rank,
        "layers": layer_names,
        "shared_layers": [],
    }
    manifest_path = os.path.join(args.output, "quantized_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Free ALL factor memory
    del flat
    gc.collect()
    print("Freed factor memory")

    # Pass 2: Load model for shared layers, merge into final safetensors
    print("Loading model for shared layers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    del tokenizer
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    shared = {}
    shared_keys = []
    for module_name, module in model.named_modules():
        class_name = type(module).__name__
        if "RMSNorm" in class_name or "LayerNorm" in class_name:
            if hasattr(module, "weight") and module.weight is not None:
                key = f"shared.{module_name}.weight"
                shared[key] = module.weight.detach().half().cpu()
                shared_keys.append(key)

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        w = model.model.embed_tokens.weight
        if w is not None:
            key = "shared.model.embed_tokens.weight"
            shared[key] = w.detach().half().cpu()
            shared_keys.append(key)

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w = model.lm_head.weight
        if w is not None:
            key = "shared.model.lm_head.weight"
            shared[key] = w.detach().half().cpu()
            shared_keys.append(key)

    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
        if hasattr(norm, "weight") and norm.weight is not None:
            key = "shared.model.norm.weight"
            shared[key] = norm.weight.detach().half().cpu()
            shared_keys.append(key)

    print(f"Collected {len(shared_keys)} shared layers")
    del model
    gc.collect()

    # Merge: load factors-only + shared into final file
    print("Loading factors-only for merge...")
    factors = load_file(factors_only_path)
    factors.update(shared)
    del shared

    final_path = os.path.join(args.output, "quantized_factors.safetensors")
    print(f"Saving final merged safetensors...")
    save_file(factors, final_path, metadata=metadata)

    # Clean up temp file
    os.remove(factors_only_path)

    # Update manifest with shared keys
    manifest["shared_layers"] = shared_keys
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done! Final output in {args.output}")


if __name__ == "__main__":
    main()
