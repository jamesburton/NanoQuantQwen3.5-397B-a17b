"""Rebuild quantized_factors.safetensors from block checkpoints.

Used when run_stage1.py completes block reconstruction but fails at the
model.save_pretrained() step (e.g. disk full). All 16 block checkpoints
must exist in --checkpoint-dir before running this script.
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanoquant.checkpoint import save_quantized_checkpoint, collect_shared_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with block_N.pt files")
    parser.add_argument("--output-dir", required=True, help="Output directory for factors file")
    parser.add_argument("--rank", type=int, required=True, help="ADMM rank used during quantization")
    parser.add_argument("--n-blocks", type=int, default=None, help="Number of blocks (auto-detected if omitted)")
    args = parser.parse_args()

    # Discover block checkpoints
    block_files = sorted(
        [f for f in os.listdir(args.checkpoint_dir) if f.startswith("block_") and f.endswith(".pt")],
        key=lambda f: int(f.split("_")[1].split(".")[0])
    )
    n_blocks = args.n_blocks or len(block_files)
    print(f"Found {len(block_files)} block checkpoints, assembling {n_blocks} blocks")

    # Assemble all_quantized from block checkpoints
    all_quantized = {}
    for block_idx in range(n_blocks):
        ckpt_path = os.path.join(args.checkpoint_dir, f"block_{block_idx}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        block_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        block_prefix = f"model.layers.{block_idx}"
        for layer_name, factors in block_data.items():
            full_key = f"{block_prefix}.{layer_name}"
            all_quantized[full_key] = factors
        print(f"  Loaded block {block_idx}/{n_blocks-1} ({len(block_data)} layers)")

    print(f"Total quantized layers: {len(all_quantized)}")

    # Load model (CPU only) to collect shared layers (norms, embeddings)
    print(f"\nLoading model for shared layers: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    shared_layers = collect_shared_layers(model)
    print(f"Collected {len(shared_layers)} shared layers")

    # Save factors checkpoint
    print(f"\nSaving quantized_factors.safetensors to {args.output_dir}")
    save_quantized_checkpoint(all_quantized, args.output_dir, args.model, args.rank, shared_layers=shared_layers)

    # Save tokenizer config alongside
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Saved {len(all_quantized)} quantized layers + {len(shared_layers)} shared layers")
    factors_path = os.path.join(args.output_dir, "quantized_factors.safetensors")
    size_gb = os.path.getsize(factors_path) / 1024**3
    print(f"File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
