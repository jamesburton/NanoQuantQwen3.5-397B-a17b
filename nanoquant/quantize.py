"""Main orchestrator for NanoQuant quantization pipeline."""

import os
import time
import psutil
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

from nanoquant.hessian import capture_hessians, robust_diagonal
from nanoquant.reconstruct import reconstruct_block, set_rotary_emb, _call_block
from nanoquant.refine import Phase3KDConfig, run_phase3_kd
from nanoquant.hardware import probe_hardware, print_hardware_summary
from nanoquant.checkpoint import save_quantized_checkpoint, collect_shared_layers

# Re-export _call_block as the canonical block runner — it handles
# rotary embedding negotiation robustly across model architectures.
_run_block = _call_block


def detect_architecture(model_name_or_path: str) -> str:
    """Detect model architecture type from HuggingFace config.

    Args:
        model_name_or_path: HuggingFace model name or local path.

    Returns:
        model_type string from config (e.g. 'qwen2_moe', 'llama', 'unknown').
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    return getattr(config, "model_type", "unknown")


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

    # Maximize CPU thread utilization for BLAS-heavy workloads
    if device == "cpu" or not torch.cuda.is_available():
        n_cores = os.cpu_count() or 1
        torch.set_num_threads(n_cores)
        try:
            torch.set_num_interop_threads(min(n_cores, 4))
        except RuntimeError:
            pass  # already set by caller
        os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
        os.environ.setdefault("MKL_NUM_THREADS", str(n_cores))

    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    hw_plan = probe_hardware(offload_dir=output_dir)
    print_hardware_summary(hw_plan)
    device = hw_plan.device  # override device arg with auto-detected

    # Load model ONCE on CPU. Blocks move to GPU one at a time during reconstruction.
    # Never load twice — would require 2x model RAM and OOM on typical workstations.
    # Use float32 on CPU (float16 matmuls can produce NaN without native FP16 compute).
    load_dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=load_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    print(f"Model loaded on CPU")

    arch_type = detect_architecture(model_name_or_path)
    print(f"  Architecture: {arch_type}")

    # Provide model-level rotary embedding to reconstruct_block
    inner = getattr(model, "model", model)
    if hasattr(inner, "rotary_emb"):
        set_rotary_emb(inner.rotary_emb)

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

    # ========= Phase 1: Capture Hessians =========
    # Hooks accumulate only 1D diagonal vectors — negligible memory overhead.
    # Use GPU if available: model processes one sample at a time so peak VRAM
    # is just the model + one batch of activations.  For small models this is
    # ~100x faster than CPU.  Fall back to CPU if the model is too large.
    hessian_device = device  # auto-detected compute device
    print(f"\n=== Phase 1: Capturing Hessians ({hessian_device}) ===")
    if hessian_device != "cpu":
        model.to(hessian_device)
    hessians = capture_hessians(model, dataloader, device=hessian_device)
    if hessian_device != "cpu":
        model.cpu()
        torch.cuda.empty_cache()
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

    # Pass through already-processed blocks to get correct input for start_block
    for i in range(start_block):
        blocks[i].to(device)
        with torch.no_grad():
            block_input = _run_block(blocks[i], block_input.to(device))
        blocks[i].cpu()
        torch.cuda.empty_cache()

    skip_patterns_env = os.getenv("NANOQUANT_SKIP_PATTERNS", "").strip()
    skip_patterns = [p.strip() for p in skip_patterns_env.split(",") if p.strip()]
    attn_rank_override_env = os.getenv("NANOQUANT_ATTN_RANK", "").strip()
    attn_rank_override = int(attn_rank_override_env) if attn_rank_override_env else None

    cfg: Dict[str, Any] = {
        "rank": rank,
        "gamma": gamma,
        "rho": rho,
        "lam": lam,
        "admm_iters": admm_iters,
        "tpre": tpre,
        "tpost": tpost,
        "skip_patterns": skip_patterns,
        "attn_rank_override": attn_rank_override,
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

        # Collect binary factors from this block (prefix with block path for uniqueness)
        for layer_name, factors in quantized.items():
            full_key = f"{block_prefix}.{layer_name}"
            all_quantized[full_key] = factors

        # Compute output for next block using quantized weights (keep model_dtype)
        with torch.no_grad():
            block_input = _run_block(block, bi)

        if not torch.isfinite(block_input).all():
            n_bad = (~torch.isfinite(block_input)).sum().item()
            max_val = block_input[torch.isfinite(block_input)].abs().max().item() if torch.isfinite(block_input).any() else 1.0
            print(f"  Warning: {n_bad} non-finite values in block {block_idx} output — clamping to continue")
            block_input = torch.nan_to_num(block_input, nan=0.0, posinf=max_val, neginf=-max_val)

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
            kd_config = Phase3KDConfig(
                strategy="connected_scales",
                kd_samples=2,
                lr=1e-3,
                min_improvement_ratio=0.02,
                eval_interval=25,
                plateau_intervals=2,
                max_cache_seconds=120.0,
                max_sample_seconds=60.0,
            )
            kd_result = run_phase3_kd(
                model,
                model_name_or_path,
                cal_data,
                n_iters=tglob,
                device=device,
                config=kd_config,
                all_quantized=all_quantized,
            )
            print(
                f"  KD result: strategy={kd_result.strategy} "
                f"applied={kd_result.applied} "
                f"reason={kd_result.reason} "
                f"baseline_kl={kd_result.baseline_kl} "
                f"final_kl={kd_result.final_kl} "
                f"improvement_ratio={kd_result.improvement_ratio} "
                f"iters={kd_result.n_iters}"
            )
            if not kd_result.applied:
                print("  Phase 3 KD did not clear the acceptance checks; both connected factor-scale and latent-factor probes are currently budget-limited on this machine.")

    print(f"\nSaving quantized model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Collect shared layers (norms, embeddings) for self-contained checkpoint
    shared_layers = collect_shared_layers(model)
    print(f"  Collected {len(shared_layers)} shared layers for self-contained checkpoint")

    # Save binary factors as safetensors alongside the FP16 model
    save_quantized_checkpoint(all_quantized, output_dir, model_name_or_path, rank, shared_layers=shared_layers)
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




