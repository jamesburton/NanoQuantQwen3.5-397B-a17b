"""Main orchestrator for NanoQuant quantization pipeline."""

import os
import time
import psutil
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from nanoquant.hessian import capture_hessians, robust_diagonal
from nanoquant.reconstruct import reconstruct_block
from nanoquant.refine import tune_scales_kd
from nanoquant.hardware import probe_hardware, print_hardware_summary
from nanoquant.checkpoint import save_quantized_checkpoint


def _load_calibration_data(tokenizer, n_calibration: int = 128, seq_len: int = 2048):
    """Load WikiText-2 calibration data."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    samples = []
    for i in range(n_calibration):
        start = i * seq_len
        end = start + seq_len
        if end > len(tokens):
            break
        samples.append(tokens[start:end])

    input_ids = torch.stack(samples)  # (n_calibration, seq_len)
    return input_ids


def _make_dataloader(input_ids: torch.Tensor, batch_size: int = 1):
    """Create a simple dataloader from input_ids tensor."""
    dataset = torch.utils.data.TensorDataset(input_ids)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _get_blocks(model):
    """Extract transformer blocks."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Cannot find transformer blocks in model architecture")


def _get_embed_and_norm(model):
    """Get embedding layer and final norm."""
    if hasattr(model, "model"):
        m = model.model
        embed = m.embed_tokens
        norm = m.norm if hasattr(m, "norm") else None
        return embed, norm
    raise ValueError("Cannot find embedding layers")


def quantize_model(
    model_name_or_path: str,
    output_dir: str,
    rank: int = 8,
    gamma: float = 0.2,
    rho: float = 0.5,
    lam: float = 1e-4,
    admm_iters: int = 50,
    tpre: int = 200,
    tpost: int = 200,
    tglob: int = 1000,
    n_calibration: int = 128,
    seq_len: int = 2048,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
    max_blocks: Optional[int] = None,
    bits_per_weight: Optional[float] = None,
) -> None:
    """Main NanoQuant quantization pipeline.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        output_dir: directory to save quantized model.
        rank: ADMM factorization rank.
        gamma: Hessian diagonal shrinkage factor.
        rho: ADMM penalty parameter.
        lam: regularization strength.
        admm_iters: ADMM iterations per layer.
        tpre: pre-tuning FP iterations per block.
        tpost: post STE refinement iterations per layer.
        tglob: global KD scale optimization iterations.
        n_calibration: number of calibration samples.
        seq_len: sequence length for calibration.
        device: computation device (overridden by auto-detected hardware plan).
        checkpoint_dir: directory for block checkpoints (resume support).
        max_blocks: process only first N blocks (for testing).
        bits_per_weight: target bits per weight (e.g. 0.55, 0.80, 1.00). If
                         provided, overrides rank with a computed value.
    """
    os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    hw_plan = probe_hardware(offload_dir=output_dir)
    print_hardware_summary(hw_plan)
    device = hw_plan.device  # override device arg with auto-detected

    # Load model ONCE on CPU. Blocks move to GPU one at a time during reconstruction.
    # Never load twice — would require 2x model RAM and OOM on typical workstations.
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    print(f"Model loaded on CPU")

    # Apply bits_per_weight -> rank override before block loop
    if bits_per_weight is not None:
        # bpw ≈ rank * (d_out + d_in) / (d_out * d_in)
        # For square matrices: bpw ≈ 2*rank/d, so rank ≈ bpw * d / 2
        first_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is not None:
            d = min(first_linear.weight.shape)
            suggested_rank = max(1, int(bits_per_weight * d / 2))
            print(f"  bits_per_weight={bits_per_weight} -> suggested rank={suggested_rank} (d={d})")
            rank = suggested_rank

    print(f"Loading calibration data ({n_calibration} samples, seq_len={seq_len})")
    cal_data = _load_calibration_data(tokenizer, n_calibration, seq_len)
    dataloader = _make_dataloader(cal_data, batch_size=1)

    # ========= Phase 1: Capture Hessians (CPU) =========
    # Hooks accumulate only 1D diagonal vectors — negligible memory overhead.
    # Running on CPU avoids VRAM OOM; calibration with small n is fast enough.
    print("\n=== Phase 1: Capturing Hessians (CPU) ===")
    hessians = capture_hessians(model, dataloader, device="cpu")
    print(f"  Captured Hessians for {len(hessians)} layers")

    # ========= Phase 2: Block-wise reconstruction =========
    print("\n=== Phase 2: Block-wise reconstruction ===")
    blocks = _get_blocks(model)
    embed, norm = _get_embed_and_norm(model)
    n_blocks = len(blocks)
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)

    # Determine starting block from checkpoints
    start_block = 0
    if checkpoint_dir:
        for i in range(n_blocks):
            ckpt_path = os.path.join(checkpoint_dir, f"block_{i}.pt")
            if os.path.exists(ckpt_path):
                start_block = i + 1
            else:
                break
        if start_block > 0:
            print(f"Resuming from block {start_block} (found {start_block} checkpoints)")

    # Compute initial block inputs by running embedding + previous blocks.
    # block_input shape: (batch, seq, hidden_dim) — keep 3D throughout.
    print("Computing block inputs...")
    embed.to(device)
    model_dtype = next(model.parameters()).dtype  # fp16
    with torch.no_grad():
        # Use a small subset (8 samples) for reconstruction speed
        cal_subset = cal_data[:8].to(device)  # (8, seq_len)
        block_input = embed(cal_subset).to(model_dtype)  # (8, seq_len, hidden)
    embed.cpu()

    def _run_block(blk, x):
        """Run a block, handling optional kwargs. Returns model_dtype tensor."""
        try:
            return blk(x)[0]
        except TypeError:
            return blk(x, attention_mask=None)[0]

    # Pass through already-processed blocks to get correct input for start_block
    for i in range(start_block):
        blocks[i].to(device)
        with torch.no_grad():
            block_input = _run_block(blocks[i], block_input.to(device))
        blocks[i].cpu()
        torch.cuda.empty_cache()

    cfg: Dict[str, Any] = {
        "rank": rank,
        "gamma": gamma,
        "rho": rho,
        "lam": lam,
        "admm_iters": admm_iters,
        "tpre": tpre,
        "tpost": tpost,
    }

    block_times = []
    all_quantized: Dict[str, Any] = {}

    for block_idx in tqdm(range(start_block, n_blocks), desc="Blocks"):
        t0 = time.time()
        block = blocks[block_idx]
        block.to(device)

        # Flatten block_input for reconstruction: (batch*seq, dim)
        bi = block_input.to(device).to(model_dtype)  # (batch, seq, hidden), fp16

        block_prefix = f"model.layers.{block_idx}"
        block_hessians = {
            k: v for k, v in hessians.items() if k.startswith(block_prefix)
        }

        # reconstruct_block accepts 3D input; it will reshape internally as needed
        quantized = reconstruct_block(
            block, bi, block_hessians, cfg, block_prefix=block_prefix
        )

        # Collect binary factors from this block
        for layer_name, factors in quantized.items():
            all_quantized[layer_name] = factors

        # Compute output for next block using quantized weights (keep model_dtype)
        with torch.no_grad():
            block_input = _run_block(block, bi)

        # Save checkpoint
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"block_{block_idx}.pt")
            torch.save(quantized, ckpt_path)

        block.cpu()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        block_times.append(elapsed)
        avg_time = sum(block_times) / len(block_times)
        remaining_blocks = n_blocks - block_idx - 1
        eta_seconds = remaining_blocks * avg_time
        eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60)}s"

        vram_gb = torch.cuda.memory_allocated(0) / 1024 ** 3 if torch.cuda.is_available() else 0
        ram_gb = psutil.virtual_memory().used / 1024 ** 3

        print(f"  Block {block_idx}/{n_blocks-1} | {elapsed:.1f}s | ETA {eta_str} | VRAM {vram_gb:.1f}GB | RAM {ram_gb:.1f}GB | {len(quantized)} layers")

    # ========= Phase 3: Global KD scale refinement =========
    # Requires loading a frozen FP reference copy of the model alongside the
    # quantized copy, which doubles RAM usage. Skip (tglob=0) on machines
    # where RAM < 2x model size.
    if tglob > 0:
        avail_gb = psutil.virtual_memory().available / 1024 ** 3
        # Rough check: we need at least the model size again in free RAM
        model_params = sum(p.numel() for p in model.parameters())
        model_size_gb = model_params * 2 / 1024 ** 3  # fp16
        if avail_gb < model_size_gb * 0.8:
            print(
                f"\nSkipping Phase 3 KD: only {avail_gb:.1f} GB free RAM, "
                f"need ~{model_size_gb:.1f} GB for FP reference copy. "
                "Re-run with more RAM or pass --tglob 0 to suppress this message."
            )
        else:
            print(f"\n=== Phase 3: Global KD refinement ({tglob} iters) ===")
            print(f"Loading FP reference model ({model_size_gb:.1f} GB)...")
            model_fp = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            model_fp.eval()
            for p in model_fp.parameters():
                p.requires_grad_(False)
            kd_loader = _make_dataloader(cal_data[:16], batch_size=1)
            tune_scales_kd(model, model_fp, kd_loader, n_iters=tglob, lr=1e-3, device=device)
            del model_fp
            torch.cuda.empty_cache()

    # ========= Save =========
    print(f"\nSaving quantized model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save binary factors as safetensors alongside the FP16 model
    save_quantized_checkpoint(all_quantized, output_dir, model_name_or_path, rank)
    print(f"Saved binary factor checkpoint ({len(all_quantized)} layers)")

    # Print effective bpw from actual tensor sizes
    if all_quantized:
        total_bits = 0
        total_params = 0
        for layer_name, factors in all_quantized.items():
            if "U_bin" in factors and "V_bin" in factors:
                u = factors["U_bin"]
                v = factors["V_bin"]
                # Binary factors: 1 bit per element
                total_bits += u.numel() + v.numel()
                # Reconstruct original param count: d_out * d_in ≈ U rows * V cols
                total_params += u.shape[0] * v.shape[-1] if u.ndim >= 2 and v.ndim >= 2 else u.numel()
        if total_params > 0:
            effective_bpw = total_bits / total_params
            print(f"  Effective bpw: {effective_bpw:.3f} bits/weight")

    # Print timing summary
    if block_times:
        print(f"\nTiming summary:")
        for i, t in enumerate(block_times):
            print(f"  Block {start_block + i}: {t:.1f}s")
        print(f"  Total block time: {sum(block_times):.1f}s")
        print(f"  Average per block: {sum(block_times)/len(block_times):.1f}s")

    print("\nQuantization complete!")
