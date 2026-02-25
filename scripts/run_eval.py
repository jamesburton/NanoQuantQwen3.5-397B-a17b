#!/usr/bin/env python3
"""Unified evaluation runner: FP16 baseline + quantized PPL + memory/timing metrics."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch

# Ensure scripts package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_ppl import evaluate_perplexity
from scripts.build_summary import rebuild_summary


def get_model_disk_size_gb(model_dir: str) -> float:
    """Sum size of all model weight files in a directory."""
    if not os.path.isdir(model_dir):
        return 0.0
    total = 0
    for fname in os.listdir(model_dir):
        if fname.endswith((".safetensors", ".bin")):
            total += os.path.getsize(os.path.join(model_dir, fname))
    return total / 1024**3


def get_quantized_disk_size_gb(checkpoint_dir: str) -> float:
    """Sum size of all quantized weight files."""
    if not os.path.isdir(checkpoint_dir):
        return 0.0
    total = 0
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith((".safetensors", ".bin", ".pt")):
            total += os.path.getsize(os.path.join(checkpoint_dir, fname))
    return total / 1024**3


def read_quantization_config(quantized_dir: str) -> dict:
    """Read quantization config from manifest if available."""
    manifest_path = os.path.join(quantized_dir, "quantized_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        return {
            "rank": manifest.get("rank"),
            "bits_per_weight": manifest.get("bits_per_weight"),
            "admm_iters": manifest.get("admm_iters"),
            "n_calibration": manifest.get("n_calibration"),
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description="Unified NanoQuant Evaluation Runner")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--quantized-dir", type=str, required=True, help="Path to quantized output")
    parser.add_argument("--results-dir", type=str, default="results", help="Results output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--stride", type=int, default=512, help="PPL eval stride")
    parser.add_argument("--max-length", type=int, default=2048, help="PPL eval max context")
    args = parser.parse_args()

    model_short_name = args.model.rstrip("/").split("/")[-1]
    output_dir = os.path.join(args.results_dir, model_short_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== NanoQuant Evaluation: {model_short_name} ===\n")

    # Reset VRAM stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- FP16 Baseline PPL ---
    print("--- FP16 Baseline Perplexity ---")
    t0 = time.time()
    try:
        baseline_result = evaluate_perplexity(
            args.model,
            device=args.device,
            max_length=args.max_length,
            stride=args.stride,
            dtype="auto",
        )
        baseline_ppl = baseline_result["ppl"]
    except Exception as e:
        print(f"ERROR: FP16 baseline eval failed: {e}")
        print("Continuing with baseline_ppl=None")
        baseline_result = None
        baseline_ppl = None
    baseline_time = time.time() - t0

    # --- Quantized PPL ---
    print("\n--- Quantized Perplexity ---")
    t0 = time.time()
    try:
        quantized_result = evaluate_perplexity(
            args.quantized_dir,
            device=args.device,
            max_length=args.max_length,
            stride=args.stride,
            dtype="float16",
        )
        quantized_ppl = quantized_result["ppl"]
    except Exception as e:
        print(f"ERROR: Quantized eval failed: {e}")
        print("Continuing with quantized_ppl=None")
        quantized_result = None
        quantized_ppl = None
    quantized_time = time.time() - t0

    # --- Memory metrics ---
    peak_vram_gb = 0.0
    gpu_name = None
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        gpu_name = torch.cuda.get_device_name(0)

    try:
        import psutil
        peak_ram_gb = psutil.virtual_memory().used / 1024**3
    except ImportError:
        print("WARNING: psutil not available, skipping RAM measurement")
        peak_ram_gb = None

    # --- Model size metrics ---
    fp16_gb = get_model_disk_size_gb(args.model)
    quantized_gb = get_quantized_disk_size_gb(args.quantized_dir)
    compression_ratio = fp16_gb / quantized_gb if quantized_gb > 0 else None

    # --- Quantization config ---
    quant_config = read_quantization_config(args.quantized_dir)

    # --- Degradation ---
    degradation = None
    if baseline_ppl is not None and quantized_ppl is not None and baseline_ppl > 0:
        degradation = round(quantized_ppl / baseline_ppl, 4)

    # --- Build metrics dict ---
    eval_settings = {
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "stride": args.stride,
        "max_length": args.max_length,
    }

    metrics = {
        "model": args.model,
        "model_short_name": model_short_name,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "hardware": {
            "gpu": gpu_name,
            "peak_vram_gb": round(peak_vram_gb, 2),
            "peak_ram_gb": round(peak_ram_gb, 2) if peak_ram_gb is not None else None,
        },
        "model_size": {
            "fp16_gb": round(fp16_gb, 2),
            "quantized_gb": round(quantized_gb, 2),
            "compression_ratio": round(compression_ratio, 2) if compression_ratio else None,
        },
        "timing": {
            "baseline_eval_seconds": round(baseline_time, 1),
            "quantized_eval_seconds": round(quantized_time, 1),
        },
        "perplexity": {
            "fp16_baseline": round(baseline_ppl, 4) if baseline_ppl is not None else None,
            "quantized": round(quantized_ppl, 4) if quantized_ppl is not None else None,
            "degradation_ratio": degradation,
            "eval_settings": eval_settings,
        },
        "quantization_config": quant_config,
        "stage1_complete": os.path.exists(
            os.path.join(args.quantized_dir, "quantized_factors.safetensors")
        ),
        "loadable_checkpoint": os.path.exists(
            os.path.join(args.quantized_dir, "config.json")
        ),
    }

    # --- Write metrics ---
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {metrics_path}")

    # --- Rebuild SUMMARY.md ---
    rebuild_summary(args.results_dir)
    print(f"SUMMARY.md updated in {args.results_dir}/")

    # --- Print summary ---
    print(f"\n=== Results for {model_short_name} ===")
    if baseline_ppl is not None:
        print(f"FP16 Baseline PPL: {baseline_ppl:.4f}")
    if quantized_ppl is not None:
        print(f"Quantized PPL:     {quantized_ppl:.4f}")
    if degradation is not None:
        print(f"Degradation:       {degradation:.4f}x")
    if fp16_gb > 0:
        print(f"FP16 Size:         {fp16_gb:.2f} GB")
    if quantized_gb > 0:
        print(f"Quantized Size:    {quantized_gb:.2f} GB")
    if compression_ratio:
        print(f"Compression:       {compression_ratio:.2f}x")
    print(f"Peak VRAM:         {peak_vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
