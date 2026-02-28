---
phase: 01-pipeline-foundation
verified: 2026-02-23T18:45:00Z
status: human_needed
score: 8/8 must-haves verified
re_verification:
  previous_status: human_needed
  previous_score: 7/7
  gaps_closed:
    - "print_hardware_summary() shows compute=cuda (UAT test 1 gap closed by plan 01-03)"
    - "HardwarePlan has dstorage_available field (plan 01-03)"
    - "dstorage-gpu detected on Windows via try/except ImportError (plan 01-03)"
    - "requirements.txt includes dstorage-gpu with win32 platform marker (plan 01-03)"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Run python scripts/run_stage1.py --model Qwen/Qwen1.5-0.5B --output /tmp/nq-test --max-blocks 2 on a machine with a 12GB GPU"
    expected: "Both blocks complete without CUDA OOM; output directory contains quantized_factors.safetensors, quantized_manifest.json, and HuggingFace model files; console shows Hardware: GPU=X.XGB usable | CPU=... | compute=cuda, model_load=cpu then per-block ETA/VRAM lines"
    why_human: "The 12GB VRAM constraint (PIPE-02) requires actual hardware execution. Code correctly implements block-by-block offloading but memory safety cannot be confirmed without running the pipeline."
  - test: "Run python scripts/run_stage1.py --model Qwen/Qwen1.5-0.5B --output /tmp/nq-bpw --bits-per-weight 0.8 --max-blocks 1"
    expected: "Console prints bits_per_weight=0.8 -> suggested rank=N (d=D) before block processing begins; resulting checkpoint reflects the computed rank"
    why_human: "Requires a live model load to reach the nn.Linear scan in quantize_model. The rank formula is correct in code but the printed rank value needs to be sensible for the model's hidden dimension."
---

# Phase 1: Pipeline Foundation Verification Report

**Phase Goal:** Stage 1 quantization (Hessian capture → ADMM init → block-wise reconstruction) runs end-to-end on a single model within 12GB VRAM
**Verified:** 2026-02-23
**Status:** human_needed
**Re-verification:** Yes — after UAT gap closure (plan 01-03)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Hardware probe returns usable GPU, CPU, and disk memory figures without crashing on CPU-only or GPU machines | VERIFIED | `hardware.py` lines 40-95: CUDA branch guarded by `torch.cuda.is_available()`, fallback to 0 for GPU; psutil provides CPU and disk unconditionally |
| 2 | Quantized binary factors (U_bin, V_bin, s1, s2) can be saved to safetensors and loaded back identically | VERIFIED | `checkpoint.py`: `save_file` with dotted keys, `safe_open` with `rfind(".")` split; roundtrip logic complete and correct |
| 3 | Checkpoint manifest JSON records model name, rank, and layer list | VERIFIED | `checkpoint.py`: manifest dict contains `model`, `rank`, `layers`; written with `json.dump` |
| 4 | Running `python scripts/run_stage1.py --model <model> --output <dir>` completes the full Stage 1 pipeline end-to-end | VERIFIED (code path) | `run_stage1.py` calls `quantize_model` with all required args; `quantize.py` executes Hessian capture → block reconstruction → save |
| 5 | Console shows per-block ETA and VRAM/RAM usage during quantization | VERIFIED | `quantize.py` line 254: prints `Block N/M | Xs | ETA Xm Xs | VRAM X.XGB | RAM X.XGB | N layers` |
| 6 | Output directory contains quantized_factors.safetensors and quantized_manifest.json alongside the FP16 model | VERIFIED | `quantize.py`: `model.save_pretrained` + `save_quantized_checkpoint`; `checkpoint.py` writes both files |
| 7 | A --bits-per-weight CLI flag maps to a rank value for sub-1-bit quantization | VERIFIED | `run_stage1.py`: `--bits-per-weight` argparse entry; `quantize.py`: rank computed as `max(1, int(bpw * d / 2))` |
| 8 | print_hardware_summary() displays compute=cuda when GPU is usable, distinct from model_load=cpu | VERIFIED | `hardware.py` line 112: `f"compute={plan.device}, model_load={plan.model_load_strategy}"`; live run confirmed: `Hardware: GPU=10.0GB usable \| CPU=29.1GB \| disk=234.7GB \| compute=cuda, model_load=cpu \| dstorage=no` |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `nanoquant/hardware.py` | Hardware auto-detection and adaptive plan, dstorage detection | VERIFIED | 119 lines; exports `HardwarePlan` dataclass (with `dstorage_available` field), `probe_hardware`, `print_hardware_summary` |
| `nanoquant/checkpoint.py` | Safetensors-based checkpoint save/load | VERIFIED | Exports `save_quantized_checkpoint`, `load_quantized_checkpoint`; uses `safetensors.torch.save_file` and `safetensors.safe_open` |
| `nanoquant/quantize.py` | Pipeline orchestrator with hardware plan, binary factor collection, checkpoint save, progress logging | VERIFIED | Imports `probe_hardware`, `print_hardware_summary`, `save_quantized_checkpoint`; contains ETA logging and bits_per_weight support |
| `scripts/run_stage1.py` | CLI entry point with --bits-per-weight flag | VERIFIED | Contains `--bits-per-weight` argparse flag; passes `bits_per_weight=args.bits_per_weight` to `quantize_model`; no `--device` argument present |
| `requirements.txt` | dstorage-gpu optional Windows dependency | VERIFIED | Line 11: `dstorage-gpu; sys_platform == "win32"` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `nanoquant/hardware.py` | `accelerate.utils.get_max_memory` | import call | VERIFIED | Line 45: `from accelerate.utils import get_max_memory` inside try block with torch fallback |
| `nanoquant/checkpoint.py` | `safetensors.torch` | import call | VERIFIED | `from safetensors.torch import save_file` and `from safetensors import safe_open` |
| `nanoquant/quantize.py` | `nanoquant/hardware.py` | import probe_hardware | VERIFIED | Line 16: `from nanoquant.hardware import probe_hardware, print_hardware_summary` |
| `nanoquant/quantize.py` | `nanoquant/checkpoint.py` | import save_quantized_checkpoint | VERIFIED | Line 17: `from nanoquant.checkpoint import save_quantized_checkpoint` |
| `scripts/run_stage1.py` | `nanoquant/quantize.py` | quantize_model call | VERIFIED | Line 7: `from nanoquant.quantize import quantize_model`; line 88: called with all required arguments |
| `nanoquant/hardware.py` | `print_hardware_summary output` | plan.device in print | VERIFIED | Line 112: `f"compute={plan.device}, model_load={plan.model_load_strategy}"` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PIPE-01 | 01-02-PLAN.md, 01-03-PLAN.md | Stage 1 pipeline runs end-to-end with clear hardware reporting | SATISFIED | `quantize.py` executes Hessian capture → block-wise reconstruction → save in sequence; hardware summary now shows compute device distinctly |
| PIPE-02 | 01-01-PLAN.md | Pipeline operates within 12GB VRAM via CPU/NVMe offloading | SATISFIED (code) | `probe_hardware()` called at startup; `device_map="cpu"` for model load; blocks moved one at a time; `torch.cuda.empty_cache()` after each block; confirmed GPU=10.0GB detected at runtime |
| PIPE-03 | 01-02-PLAN.md | Sub-1-bit quantization (0.7–0.9 bits) as primary mode | SATISFIED | `--bits-per-weight` flag wired end-to-end; rank formula `max(1, int(bpw * d / 2))` implements sub-1-bit mapping |
| PIPE-04 | 01-01-PLAN.md | Produces serializable quantized checkpoint | SATISFIED | `save_quantized_checkpoint` saves safetensors + manifest; `load_quantized_checkpoint` restores identical tensors |
| PIPE-05 | 01-02-PLAN.md | Progress logging with ETA and memory usage tracking | SATISFIED | Per-block print shows elapsed, ETA, VRAM GB, RAM GB |

All 5 Phase 1 requirements (PIPE-01 through PIPE-05) are claimed by plans and have implementation evidence. No orphaned requirements.

### Anti-Patterns Found

None detected. No TODO, FIXME, placeholder comments, empty return values, or console-log-only implementations found across any modified files (hardware.py, checkpoint.py, quantize.py, run_stage1.py, requirements.txt).

### Human Verification Required

#### 1. End-to-End OOM Safety on 12GB GPU

**Test:** Run `python scripts/run_stage1.py --model Qwen/Qwen1.5-0.5B --output /tmp/nq-test --max-blocks 2` on a machine with a 12GB GPU.
**Expected:** Both blocks complete without CUDA OOM; output directory contains `quantized_factors.safetensors`, `quantized_manifest.json`, and HuggingFace model files; console shows `Hardware: GPU=X.XGB usable | CPU=... | compute=cuda, model_load=cpu` then per-block ETA/VRAM lines.
**Why human:** The 12GB VRAM constraint (PIPE-02) requires actual hardware execution. The code correctly implements block-by-block offloading but memory safety cannot be confirmed without running the pipeline.

#### 2. --bits-per-weight Rank Override

**Test:** Run `python scripts/run_stage1.py --model Qwen/Qwen1.5-0.5B --output /tmp/nq-bpw --bits-per-weight 0.8 --max-blocks 1`.
**Expected:** Console prints `bits_per_weight=0.8 -> suggested rank=N (d=D)` before block processing begins; resulting checkpoint reflects the computed rank.
**Why human:** Requires a live model load to reach the `nn.Linear` scan in `quantize_model`. The rank formula is correct in code but the printed rank value needs to be a sensible number for the model's hidden dimension.

### Re-verification Summary

Plan 01-03 closed the only gap from the previous verification cycle (UAT test 1). The hardware summary now correctly shows `compute=cuda` separately from `model_load=cpu`, and `HardwarePlan` has a new `dstorage_available` field with Windows DirectStorage detection. The `dstorage-gpu` package is listed in `requirements.txt` with a PEP 508 platform marker (`sys_platform == "win32"`).

Live execution confirmed:
```
Hardware: GPU=10.0GB usable | CPU=29.1GB | disk=234.7GB | compute=cuda, model_load=cpu | dstorage=no
```

No regressions detected in the previously verified artifacts. The two remaining human verification items are unchanged — they require actual GPU pipeline execution to confirm VRAM limits and rank formula output for a live model.

---

_Verified: 2026-02-23_
_Verifier: Claude (gsd-verifier)_
