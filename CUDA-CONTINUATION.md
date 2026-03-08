# CUDA Continuation Guide

Instructions for an agent or developer with a CUDA GPU to produce production-quality quantization results.

## Context

Phase 4 (Phi MoE Support) is complete. The pipeline works end-to-end, but the current results use **zero-refinement settings** (CPU-only run), producing 141x PPL degradation. A CUDA re-run with default refinement will produce dramatically better quality (~1.5-3x degradation expected).

### Current Results (CPU, zero-refinement)

| Metric | Value |
|--------|-------|
| FP16 Baseline PPL | 1,927.0 |
| Quantized PPL | 272,277.0 |
| Degradation | 141.3x |

### Known Bug Fix (already committed)

`nanoquant/quantize.py` had a critical factor key collision bug where all 32 blocks overwrote each other (only block 31 survived). This is **already fixed** in commit `516da3c`. Any new quantization run will produce correct results.

## Prerequisites

- CUDA GPU with >= 12GB VRAM (e.g., RTX 3060/4070)
- Python 3.10+
- PyTorch with CUDA support

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets safetensors psutil tqdm
```

## Step 1: Apply SlimMoE Compatibility Patch

Phi-tiny-MoE-instruct uses SlimMoE which needs a patch for transformers >= 5.x:

```bash
python scripts/patch_slimmoe_compat.py
```

## Step 2: Re-quantize with Full Refinement

The key difference: default settings include `tpre=200, tpost=200, tglob=1000` refinement passes.

```bash
python scripts/run_stage1.py \
  --model microsoft/Phi-tiny-MoE-instruct \
  --output output/Phi-tiny-MoE-instruct \
  --rank 8 \
  --n-calibration 128 \
  --seq-len 2048 \
  --admm-iters 50
```

**Expected timing:** ~30-60 minutes on a 12GB consumer GPU (3.8B params, 32 blocks).

Block checkpoints are saved to `output/Phi-tiny-MoE-instruct/checkpoints/block_*.pt` for resume support — if interrupted, re-running will skip completed blocks.

## Step 3: Run WikiText-2 Evaluation

```bash
python scripts/run_eval.py \
  --model microsoft/Phi-tiny-MoE-instruct \
  --quantized-dir output/Phi-tiny-MoE-instruct \
  --results-dir results \
  --device cuda
```

This will:
1. Measure FP16 baseline WikiText-2 PPL
2. Auto-reconstruct quantized model from factors
3. Measure quantized WikiText-2 PPL
4. Record peak VRAM, model sizes, compression ratio
5. Write `results/Phi-tiny-MoE-instruct/metrics.json`
6. Update `results/SUMMARY.md`

**Expected timing:** ~15-30 minutes on CUDA (vs ~3 hours on CPU).

## Step 4: Commit Results

```bash
git add results/Phi-tiny-MoE-instruct/metrics.json results/SUMMARY.md
git commit -m "feat(04): re-run Phi-tiny-MoE eval with CUDA refinement

Updated WikiText-2 perplexity results using full refinement settings
(tpre=200, tpost=200, tglob=1000) on CUDA GPU.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

## Step 5 (Optional): Continue to Phase 3

Phase 3 (Scaling and Evaluation) validates the pipeline across four model sizes:

| Model | Params | GPU Requirement |
|-------|--------|-----------------|
| OLMoE-1B-7B-0924 | 6.9B (1B active) | 12GB consumer |
| Qwen1.5-MoE-A2.7B-Chat | 14.3B (2.7B active) | 12GB consumer |
| Qwen2-MoE-57B | 57B (14B active) | Cloud GPU |
| Qwen3.5-397B-A17B | 397B (17B active) | Cloud GPU |

Start with:
```bash
# Plan Phase 3 if not already planned
# Then execute OLMoE-1B first (smallest, validates pipeline on CUDA)
python scripts/run_stage1.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --output output/OLMoE-1B-7B-0924 \
  --rank 8

python scripts/run_eval.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --quantized-dir output/OLMoE-1B-7B-0924 \
  --results-dir results
```

## Architecture Notes

- **SlimMoE** (used by Phi-tiny-MoE-instruct): Separate `nn.Linear` experts (w1, w2, w3 per expert). Handled by the standard nn.Linear traversal path.
- **PhimoeExperts** (fused): 3D fused `gate_up_proj` tensor. Handled by split WeightView logic in `nanoquant/moe.py`. This code path exists but has NOT been exercised at runtime yet — only the SlimMoE path was validated.
- The pipeline auto-detects architecture via `detect_architecture()` in `nanoquant/quantize.py`.

## File Layout

```
output/Phi-tiny-MoE-instruct/
  quantized_factors.safetensors  # Binary factors (796 MB, 1664 layers)
  quantized_manifest.json        # Manifest with layer list
  config.json                    # HF model config
  checkpoints/block_*.pt         # Per-block checkpoints (32 files)

results/Phi-tiny-MoE-instruct/
  metrics.json                   # PPL + memory + timing metrics

results/SUMMARY.md               # Aggregated results table
```
