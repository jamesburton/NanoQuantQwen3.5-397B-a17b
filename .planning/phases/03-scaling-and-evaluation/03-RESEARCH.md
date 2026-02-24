# Phase 3: Scaling and Evaluation - Research

**Researched:** 2026-02-24
**Domain:** MoE model scaling validation, WikiText-2 perplexity evaluation, memory/timing reporting
**Confidence:** HIGH

## Summary

Phase 3 is a validation-and-measurement phase, not a feature phase. The pipeline from Phases 1 and 2 is already complete; this phase exercises it against four model sizes to confirm correctness and gather baseline perplexity numbers. The primary technical challenges are: (1) the eval script's current PPL implementation has a known deficiency (no sliding-window masking of context tokens), (2) memory and timing data needs to be captured systematically and written to structured JSON, and (3) large models (57B, 397B) require cloud GPU setup with environment parity to local runs.

The Qwen3.5-397B-A17B architecture introduces `qwen3_5_moe` model type and `Qwen3_5MoeExperts` fused expert class — both already registered in `nanoquant/moe.py:FUSED_EXPERT_CLASSES`. The pipeline should run for Qwen3.5-397B-A17B without code changes, but this needs to be confirmed on actual hardware. The most defensible risk mitigation is to run OLMoE-1B and Qwen1.5-3B locally first (fast feedback), then confirm the pattern holds before committing cloud GPU time to the large models.

**Primary recommendation:** Fix the PPL evaluation to use the proper HuggingFace sliding-window method (masking context tokens with -100), then run models smallest-to-largest, capturing JSON metrics per model and building SUMMARY.md incrementally.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scaling Strategy**
- Small models (OLMoE-1B, Qwen1.5-3B) run locally on 12GB consumer GPU
- Large models (Qwen2-57B, Qwen3.5-397B) run on rented cloud GPU via RunPod or vast.ai
- Scripts must be hardware-agnostic — same code runs locally and on cloud

**Perplexity Evaluation**
- Run our own FP16 baseline perplexity on WikiText-2 for every model (no published numbers)
- Identical evaluation settings for FP16 and quantized versions to ensure apples-to-apples comparison

**Memory & Timing Reporting**
- Track three memory metrics: peak VRAM, peak system RAM, model size before/after quantization
- Track wall-clock time: total quantization time + per-layer timing (useful for estimating cloud GPU cost)
- Live VRAM tracking during quantization runs + summary table at the end

**Output Artifacts**
- `results/{model_name}/` — JSON metrics per model + human-readable notes
- `results/SUMMARY.md` — aggregated comparison table across all models
- `checkpoints/{model_name}/` — quantized checkpoint files (separate from reports)
- Eval script auto-generates/updates SUMMARY.md after each model evaluation
- JSON format for machine-readable results, Markdown for human consumption

### Claude's Discretion
- Execution order across models (smallest-first recommended but flexible)
- Failure handling strategy (continue vs stop on error)
- WikiText-2 context length and stride for perplexity measurement
- Whether eval is a separate script or integrated (separate script preferred)
- Overall PPL vs per-layer breakdown (overall PPL sufficient for Phase 3 baseline)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCALE-01 | Verified working on OLMoE-1B-7B-0924 (smallest, consumer GPU validation) | OLMoE uses `olmoe` model_type, `OlmoeExperts` fused class — both registered in moe.py. 1.3B active / 7B total. Fits 12GB with block-by-block offload. |
| SCALE-02 | Verified working on Qwen1.5-MoE-3B-14B (small-medium scaling check) | Uses `qwen2_moe` model_type, `Qwen2MoeExperts` — both registered. 2.7B active / 14.3B total. Fits 12GB with CPU offload. |
| SCALE-03 | Verified working on Qwen2-MoE-57B (medium-large scaling) | Uses `qwen2_moe` model_type, `Qwen2MoeExperts` — same class as Qwen1.5-MoE. 14B active / 57B total. ~114GB FP16, requires cloud GPU with NVMe offload or multi-GPU. |
| SCALE-04 | Verified working on Qwen3.5-397B-A17B (full target, likely requires cloud GPU) | Uses `qwen3_5_moe` model_type, `Qwen3_5MoeExperts` — both registered in moe.py. 17B active / 397B total. ~794GB FP16, requires A100/H100 80GB with CPU/NVMe offload. |
| EVAL-01 | Perplexity evaluation on WikiText-2 for quantized models | `eval_ppl.py` exists but uses non-standard PPL computation (no -100 masking). Needs fix to match HuggingFace canonical sliding-window method. |
| EVAL-02 | Comparison baseline: FP16 perplexity for each model | `eval_ppl.py` supports `--baseline` flag. Same script, same settings — apples-to-apples is already the design intention. Needs stride standardization. |
| EVAL-03 | Memory footprint reporting (model size before/after, peak VRAM usage) | `torch.cuda.max_memory_allocated()` for peak VRAM; `psutil.virtual_memory()` for RAM; model file sizes for disk footprint. All available in existing stack. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | (already installed) | VRAM measurement (`max_memory_allocated`), model ops | Central dep of whole project |
| transformers | (already installed) | `AutoModelForCausalLM`, `AutoTokenizer`, `AutoConfig` | Model loading for all 4 architectures |
| datasets | (already installed) | WikiText-2 test/train split loading | HuggingFace canonical dataset source |
| psutil | (already installed) | System RAM measurement | Already used in quantize.py |
| safetensors | (already installed) | Checkpoint size measurement on disk | Already used in checkpoint.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | JSON results output | Per-model metrics file |
| os.path.getsize | stdlib | Checkpoint file size (bytes) | Model size before/after |
| time | stdlib | Wall-clock timing | Already used in run_stage1.py |
| tqdm | (already installed) | Progress on PPL eval token loop | Optional UX improvement |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual JSON write | pandas | JSON is simpler, no new dep |
| `max_memory_allocated` | nvidia-smi polling | torch API is process-accurate; nvidia-smi sees whole GPU |
| WikiText-2 | C4 or PTB | WikiText-2 is the standard quantization benchmark; user confirmed |

**Installation:** No new dependencies. Everything already in `pyproject.toml`.

## Architecture Patterns

### Model Size Reference Table
| HF Model ID | model_type | Fused Expert Class | Active Params | Total Params | FP16 Size (approx) | GPU Target |
|-------------|------------|-------------------|---------------|--------------|---------------------|------------|
| `allenai/OLMoE-1B-7B-0924` | `olmoe` | `OlmoeExperts` | 1.3B | 7B | ~14GB | 12GB consumer (w/ offload) |
| `Qwen/Qwen1.5-MoE-A2.7B` | `qwen2_moe` | `Qwen2MoeExperts` | 2.7B | 14.3B | ~28GB | 12GB consumer (w/ offload) |
| `Qwen/Qwen2-57B-A14B` | `qwen2_moe` | `Qwen2MoeExperts` | 14B | 57B | ~114GB | A100/H100 80GB cloud |
| `Qwen/Qwen3.5-397B-A17B` | `qwen3_5_moe` | `Qwen3_5MoeExperts` | 17B | 397B | ~794GB | A100/H100 80GB cloud (multi-node or NVMe offload) |

### Architecture: Results Directory Layout
```
results/
├── OLMoE-1B-7B-0924/
│   ├── metrics.json       # machine-readable metrics
│   └── notes.md           # human observations, run notes
├── Qwen1.5-MoE-A2.7B/
│   ├── metrics.json
│   └── notes.md
├── Qwen2-57B-A14B/
│   ├── metrics.json
│   └── notes.md
├── Qwen3.5-397B-A17B/
│   ├── metrics.json
│   └── notes.md
└── SUMMARY.md             # auto-generated comparison table

checkpoints/
├── OLMoE-1B-7B-0924/     # quantized checkpoint files
├── Qwen1.5-MoE-A2.7B/
├── Qwen2-57B-A14B/
└── Qwen3.5-397B-A17B/
```

### Pattern 1: Correct Sliding-Window Perplexity (HuggingFace Standard)

**What:** The canonical HuggingFace approach masks context tokens with target=-100 so loss only counts predicted tokens, not context. The existing `eval_ppl.py` does NOT do this — it passes `ids[:, :-1]` as input and `ids[:, 1:]` as targets but within a stride window without context masking. This slightly inflates PPL for strided evaluation.

**When to use:** Always — for both FP16 baseline and quantized evaluation.

**Correct pattern (from HuggingFace official docs):**
```python
# Source: https://huggingface.co/docs/transformers/en/perplexity
stride = 512
max_length = 2048  # or model's context length, capped at reasonable value
nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0

for begin_loc in range(0, seq_len, stride):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # tokens to score this window
    input_ids_window = input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids_window.clone()
    target_ids[:, :-trg_len] = -100  # mask context, score only new tokens

    with torch.no_grad():
        outputs = model(input_ids_window, labels=target_ids)
        neg_log_likelihood = outputs.loss  # averaged over non-masked tokens

    num_valid = (target_ids != -100).sum().item()
    num_loss_tokens = num_valid - target_ids.size(0)  # subtract for internal shift
    nll_sum += neg_log_likelihood * num_loss_tokens
    n_tokens += num_loss_tokens
    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.tensor(nll_sum / n_tokens)).item()
```

**Key differences from current eval_ppl.py:**
1. Uses `model(input_ids, labels=target_ids)` — lets transformers compute loss internally
2. Masks context tokens with `-100`
3. Accumulates loss weighted by token count (not simple mean of means)

### Pattern 2: Peak Memory Capture

**What:** Reset peak memory counter before loading model, capture after key phases.

```python
# Source: PyTorch docs - torch.cuda.max_memory_allocated
import torch, psutil, os

def capture_memory_metrics(model_path: str, checkpoint_dir: str) -> dict:
    torch.cuda.reset_peak_memory_stats()

    # Model size on disk (before quantization = FP16 safetensors/bin files)
    model_size_bytes = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in os.listdir(model_path)
        if f.endswith((".safetensors", ".bin"))
    )

    # After quantization
    quant_size_bytes = sum(
        os.path.getsize(os.path.join(checkpoint_dir, f))
        for f in os.listdir(checkpoint_dir)
        if f.endswith((".safetensors", ".bin", ".pt"))
    )

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    ram_used_bytes = psutil.virtual_memory().used

    return {
        "model_size_fp16_gb": model_size_bytes / 1024**3,
        "model_size_quantized_gb": quant_size_bytes / 1024**3,
        "peak_vram_gb": peak_vram_bytes / 1024**3,
        "peak_ram_gb": ram_used_bytes / 1024**3,
        "compression_ratio": model_size_bytes / quant_size_bytes if quant_size_bytes > 0 else None,
    }
```

### Pattern 3: JSON Metrics Schema

```python
metrics = {
    "model": "allenai/OLMoE-1B-7B-0924",
    "model_type": "olmoe",
    "date": "2026-02-24",
    "hardware": {
        "gpu": "RTX 3080 12GB",
        "peak_vram_gb": 11.2,
        "peak_ram_gb": 28.4,
    },
    "model_size": {
        "fp16_gb": 14.1,
        "quantized_gb": 1.8,
        "compression_ratio": 7.8,
    },
    "timing": {
        "total_quantization_seconds": 3420,
        "average_per_block_seconds": 214,
        "block_times": [201, 218, ...]
    },
    "perplexity": {
        "fp16_baseline": 9.42,
        "quantized": 10.87,
        "degradation_ratio": 1.15,
        "eval_settings": {
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
            "stride": 512,
            "max_length": 2048,
        }
    },
    "quantization_config": {
        "rank": 8,
        "bits_per_weight": 0.80,
        "admm_iters": 50,
        "n_calibration": 128,
    },
    "stage1_complete": True,
    "loadable_checkpoint": True,
}
```

### Pattern 4: SUMMARY.md Auto-Generation

After each model run, append/rebuild SUMMARY.md with a comparison table:

```markdown
# NanoQuant Scaling Results

| Model | Params (Active/Total) | FP16 PPL | Quant PPL | Degradation | FP16 Size | Quant Size | Ratio | Peak VRAM | Quant Time |
|-------|-----------------------|----------|-----------|-------------|-----------|------------|-------|-----------|------------|
| OLMoE-1B-7B | 1.3B / 7B | 9.42 | 10.87 | 1.15x | 14.1 GB | 1.8 GB | 7.8x | 11.2 GB | 57 min |
```

### Anti-Patterns to Avoid
- **Running PPL eval with a cold model:** Always call `model.eval()` and run a warmup forward pass before timing PPL evaluation
- **Comparing PPL across different strides:** All 4 models MUST use identical stride and max_length settings for the comparison to be valid
- **Capturing VRAM without resetting peak stats:** Always call `torch.cuda.reset_peak_memory_stats()` at the start of a quantization run, not just before reading
- **Loading quantized model via AutoModelForCausalLM for PPL eval:** The eval_ppl.py currently loads checkpoint this way — this works only if the checkpoint is a valid HF model dir with config.json; needs verification after each run
- **Silently OOMing on cloud:** Large models need `--max-blocks N` dry-run first to confirm hardware fit

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WikiText-2 download | Custom downloader | `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")` | Handles caching, splits, streaming |
| Sliding-window PPL | Custom NLL accumulator | Use `model(input_ids, labels=target_ids)` with -100 masking | The internal CrossEntropyLoss shift is handled correctly; rolling your own misses the final-token shift |
| Peak VRAM tracking | nvidia-smi subprocess | `torch.cuda.max_memory_allocated()` | Process-specific, no polling race conditions |
| Cloud GPU env setup | Manual pip installs | RunPod template or Docker image with pre-installed deps | Saves 20-30 min per run on large models |
| Checkpoint existence check for resume | Custom file scanner | Existing `checkpoint_dir` logic in `quantize.py` already handles this | Don't duplicate |

**Key insight:** This phase is almost entirely glue code — the pipeline exists, the data source exists, the PyTorch memory API exists. The work is wiring them together correctly and avoiding the known PPL implementation trap.

## Common Pitfalls

### Pitfall 1: eval_ppl.py PPL Computation is Non-Standard
**What goes wrong:** The current implementation uses a non-masked striding loop. It computes `ids[:, :-1]` as input and `ids[:, 1:]` as targets within each stride window without masking the context region. This gives slightly inflated PPL numbers and is NOT the same as the HuggingFace canonical method used in most published quantization papers.
**Why it happens:** The existing code was written as a quick prototype. It works as a sanity check but produces numbers that can't be directly compared to literature.
**How to avoid:** Rewrite `evaluate_perplexity()` in `eval_ppl.py` using the `-100` context masking pattern above. Keep `stride=512`, `max_length=2048` as defaults.
**Warning signs:** If FP16 PPL for OLMoE-1B doesn't come out near 9-11 (expected range for this model size on WikiText-2), the computation may be off.

### Pitfall 2: Qwen3.5-397B Has `bfloat16` as Default dtype
**What goes wrong:** The model config specifies `torch_dtype: bfloat16`. If `eval_ppl.py` loads with `dtype=torch.float16`, there may be precision mismatches. The quantize.py pipeline loads with `dtype=torch.float16` — this is intentional for the quantization pass, but the FP16 baseline PPL eval should use the model's native dtype.
**Why it happens:** All other models in this project are loaded as float16. Qwen3.5-397B is bfloat16 natively.
**How to avoid:** Pass `torch_dtype="auto"` for baseline PPL eval to respect the model's native dtype. For the quantized eval, continue using `float16` for consistency with the quantized factors.
**Warning signs:** Unusually high baseline PPL for Qwen3.5-397B (>20) while other models look normal.

### Pitfall 3: Qwen3.5-397B Has a Novel Hybrid Architecture
**What goes wrong:** Qwen3.5-397B uses a hybrid Gated DeltaNet + MoE architecture with alternating `linear_attention` and `full_attention` layers. The `_call_block` / `_run_block` in `reconstruct.py` uses rotary embedding injection — Gated DeltaNet layers may not have or need a rotary embedding. If `set_rotary_emb` fails or `_run_block` crashes on a DeltaNet block, the pipeline will fail on Qwen3.5-397B.
**Why it happens:** The pipeline was designed for standard transformer blocks. DeltaNet is a different attention mechanism.
**How to avoid:** Test on a `--max-blocks 2` dry run first. If `_call_block` raises on DeltaNet layers, add a try/except fallback or a `_run_block_plain` variant that calls `block(hidden_states)` directly for non-standard block types.
**Warning signs:** AttributeError or shape mismatch in reconstruct.py when processing Qwen3.5-397B block 0 or block 3 (the first full_attention block in the 3+1 pattern).

### Pitfall 4: Cloud GPU Environment Not Matching Local
**What goes wrong:** Cloud instance may have different CUDA version, transformers version, or missing `trust_remote_code` capabilities. The Qwen3.5-397B model requires `trust_remote_code=True` and a recent enough transformers version to have `Qwen3_5MoeForConditionalGeneration` registered.
**Why it happens:** RunPod/vast.ai templates ship with specific base images (often PyTorch CUDA 11.8 or 12.1).
**How to avoid:** On cloud instance, run `pip install -e .` from the repo, then `python -c "from nanoquant.moe import FUSED_EXPERT_CLASSES; print(FUSED_EXPERT_CLASSES)"` before starting quantization to verify the environment.
**Warning signs:** ImportError or `model_type not recognized` when loading Qwen3.5-397B config.

### Pitfall 5: Peak VRAM Reads Before Reset
**What goes wrong:** `torch.cuda.max_memory_allocated()` returns the peak since the process started or last reset. If called without first calling `torch.cuda.reset_peak_memory_stats()`, it may include VRAM from previous operations (e.g., a previous dry run in the same process).
**Why it happens:** Easy to forget to reset before a new measurement.
**How to avoid:** Always call `torch.cuda.reset_peak_memory_stats()` at the start of the function/script that runs quantization.

### Pitfall 6: SUMMARY.md Clobbered Between Runs
**What goes wrong:** If SUMMARY.md is fully regenerated from scratch each time (reading only the current model's results), it loses prior model data.
**Why it happens:** Naive write-from-scratch approach.
**How to avoid:** Read all existing `results/*/metrics.json` files and regenerate the full table from scratch each time. This is idempotent and safe.

## Code Examples

### Model Size on Disk
```python
# Source: stdlib os.path, verified pattern
import os

def get_model_disk_size_gb(model_dir: str) -> float:
    """Sum size of all model weight files in a directory."""
    total = 0
    for fname in os.listdir(model_dir):
        if fname.endswith((".safetensors", ".bin")):
            total += os.path.getsize(os.path.join(model_dir, fname))
    return total / 1024**3
```

### Hardware-Agnostic Run Script Pattern
```python
# Cloud vs local: same code, same invocation
# Local:  python scripts/run_stage1.py --model allenai/OLMoE-1B-7B-0924 --output results/OLMoE-1B-7B-0924
# Cloud:  python scripts/run_stage1.py --model Qwen/Qwen3.5-397B-A17B --output results/Qwen3.5-397B-A17B

# Hardware detection is already automatic in quantize.py via probe_hardware()
# Scripts should print a clear error if VRAM is insufficient BEFORE starting:

import torch, psutil
def check_hardware_or_exit(min_vram_gb: float, min_ram_gb: float):
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < min_vram_gb:
            print(f"ERROR: This model requires ~{min_vram_gb:.0f}GB VRAM. Found {vram_gb:.1f}GB.")
            print("Use a cloud GPU with sufficient VRAM (RunPod A100 80GB or H100 80GB recommended).")
            raise SystemExit(1)
    ram_gb = psutil.virtual_memory().total / 1024**3
    if ram_gb < min_ram_gb:
        print(f"WARNING: This model may require ~{min_ram_gb:.0f}GB system RAM. Found {ram_gb:.1f}GB.")
```

### JSON Write + SUMMARY.md Rebuild
```python
import json, os, glob

def write_metrics(results_dir: str, model_name: str, metrics: dict):
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    rebuild_summary(results_dir)

def rebuild_summary(results_dir: str):
    """Read all metrics.json and write SUMMARY.md."""
    rows = []
    for metrics_path in sorted(glob.glob(os.path.join(results_dir, "*/metrics.json"))):
        with open(metrics_path) as f:
            m = json.load(f)
        rows.append(m)

    lines = ["# NanoQuant Scaling Results\n\n"]
    lines.append("| Model | FP16 PPL | Quant PPL | Degradation | FP16 Size | Quant Size | Ratio | Peak VRAM | Quant Time |\n")
    lines.append("|-------|----------|-----------|-------------|-----------|------------|-------|-----------|------------|\n")
    for m in rows:
        ppl = m.get("perplexity", {})
        mem = m.get("model_size", {})
        hw = m.get("hardware", {})
        t = m.get("timing", {})
        lines.append(
            f"| {m.get('model','?')} "
            f"| {ppl.get('fp16_baseline','?')} "
            f"| {ppl.get('quantized','?')} "
            f"| {ppl.get('degradation_ratio','?')}x "
            f"| {mem.get('fp16_gb','?'):.1f} GB "
            f"| {mem.get('quantized_gb','?'):.1f} GB "
            f"| {mem.get('compression_ratio','?'):.1f}x "
            f"| {hw.get('peak_vram_gb','?'):.1f} GB "
            f"| {t.get('total_quantization_seconds',0)/60:.0f} min |\n"
        )
    with open(os.path.join(results_dir, "SUMMARY.md"), "w") as f:
        f.writelines(lines)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Non-strided PPL (truncate to max_len) | Sliding-window with -100 context masking | HF docs recommend since ~2021 | Produces lower (better) and more accurate PPL; comparable to published numbers |
| Single-GPU only | CPU/NVMe offload via Accelerate | 2022+ | 7B+ models on consumer 12GB GPU |
| nvidia-smi polling for VRAM | `torch.cuda.max_memory_allocated()` | PyTorch 1.x+ | Process-accurate peak capture |

**Deprecated/outdated:**
- The `--baseline` flag in `eval_ppl.py` that loads the full FP16 model after quantization: this doubles RAM/VRAM at eval time. Better to run FP16 baseline as a separate invocation before quantization, save to metrics.json, then run quantized eval separately.

## Open Questions

1. **Will `_call_block` / `_run_block` handle Qwen3.5-397B's DeltaNet layers?**
   - What we know: Qwen3.5-397B has 60 layers in a 3×(DeltaNet→MoE) + 1×(Attention→MoE) repeating pattern. DeltaNet layers are non-standard.
   - What's unclear: Whether `_run_block` in reconstruct.py will pass hidden_states correctly through DeltaNet layers (which may require different argument signatures than standard attention blocks).
   - Recommendation: Run `--max-blocks 5` on Qwen3.5-397B first. This exercises both DeltaNet and Attention block types before committing to a full run.

2. **How much cloud RAM is needed for Qwen3.5-397B?**
   - What we know: 397B params × 2 bytes (FP16) = ~794GB. Even with NVMe offload, the block-by-block loading in quantize.py requires the full model in CPU RAM conceptually, though blocks are evicted.
   - What's unclear: Whether a single A100 80GB + 256GB system RAM is sufficient, or whether NVMe offload is needed.
   - Recommendation: Use a RunPod A100 80GB instance with 256GB RAM. If that's insufficient, enable `--offload-folder` for NVMe offload. The Accelerate-based offload path in `probe_hardware()` should handle this, but verify on a 2-block dry run.

3. **Should `eval_ppl.py` load quantized model differently from FP16 baseline?**
   - What we know: `eval_ppl.py` uses `AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)`. For FP16 baseline this is fine. For quantized checkpoint, the quantized factors are in `quantized_factors.safetensors` but the model is saved via `model.save_pretrained()` with FP16 weights (not true quantized weights for inference).
   - What's unclear: Whether `eval_ppl.py` can load the quantized output dir as a normal HF model. In the current pipeline, `model.save_pretrained(output_dir)` saves the FP16 modified weights, so this should work — but the PPL will reflect the pipeline's output quality (modified FP16 weights from reconstruction), not a true sub-1-bit quantized inference.
   - Recommendation: Document this clearly in results notes. The PPL being measured is the reconstruction quality of the Stage 1 block-wise reconstruction, not inference-speed quantization.

## Validation Architecture

> Note: `.planning/config.json` does not have `workflow.nyquist_validation` set. However, `auto_verify: true` is set in preferences. There is no existing test infrastructure (no `tests/` directory, no pytest.ini). Given the operational nature of this phase (run scripts, capture metrics), the validation approach is script-level integration testing rather than unit tests. No Wave 0 test files are required for this phase.

**Verification strategy for each requirement:**
- SCALE-01/02/03/04: Script exits 0, checkpoint dir contains `quantized_factors.safetensors`, `quantized_manifest.json`
- EVAL-01/02: `results/{model}/metrics.json` exists and contains `perplexity.fp16_baseline` and `perplexity.quantized` fields
- EVAL-03: `metrics.json` contains `hardware.peak_vram_gb`, `model_size.fp16_gb`, `model_size.quantized_gb`

## Sources

### Primary (HIGH confidence)
- [HuggingFace Transformers Perplexity Docs](https://huggingface.co/docs/transformers/en/perplexity) — sliding-window PPL implementation, -100 masking pattern, stride=512 recommendation
- [Qwen/Qwen3.5-397B-A17B config.json](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) — model_type=qwen3_5_moe, 60 layers, 512 experts, 10 per token, bfloat16 native dtype
- `nanoquant/moe.py` (codebase) — FUSED_EXPERT_CLASSES already contains all 4 target architectures
- `nanoquant/quantize.py` (codebase) — block timing, VRAM/RAM logging already partially implemented
- PyTorch docs — `torch.cuda.max_memory_allocated()`, `torch.cuda.reset_peak_memory_stats()`

### Secondary (MEDIUM confidence)
- [allenai/OLMoE-1B-7B-0924 HuggingFace](https://huggingface.co/allenai/OLMoE-1B-7B-0924) — 1.3B active / 7B total, olmoe architecture, 64 experts / 8 per token
- [Qwen/Qwen1.5-MoE-A2.7B HuggingFace](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) — 2.7B active / 14.3B total, qwen2_moe model_type
- [Qwen/Qwen2-57B-A14B HuggingFace](https://huggingface.co/Qwen/Qwen2-57B-A14B) — 14B active / 57B total, qwen2_moe (same class as Qwen1.5-MoE)
- [RunPod GPU Pricing](https://www.runpod.io/pricing) — A100 80GB ~$1.64-1.74/hr, H100 ~$2.65/hr (2025)
- [Vast.ai GPU Pricing](https://computeprices.com/compare/runpod-vs-vast) — H100 from $1.49/hr, A100 sub-$1/hr marketplace

### Tertiary (LOW confidence)
- Search results on quantization paper PPL ranges — OLMoE-1B WikiText-2 PPL ~9-11 is inference from model size class, not directly verified

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already installed, no new libraries
- Architecture patterns: HIGH — PPL pattern from official HF docs; JSON/OS patterns from stdlib
- Pitfalls: HIGH for PPL bug (confirmed by reading code), MEDIUM for Qwen3.5 DeltaNet risk (inferred from architecture docs, not tested)
- Cloud GPU guidance: MEDIUM — pricing current as of 2025/2026, instance specs from provider docs

**Research date:** 2026-02-24
**Valid until:** 2026-05-24 (stable domain — model architectures don't change; HF PPL docs stable)
