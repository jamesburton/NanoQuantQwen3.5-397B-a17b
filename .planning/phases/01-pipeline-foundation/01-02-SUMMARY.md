---
phase: 01-pipeline-foundation
plan: 02
subsystem: pipeline-orchestrator
tags: [hardware, checkpoint, safetensors, progress-logging, bits-per-weight, quantize, cli]

requires:
  - 01-01 (hardware.py + checkpoint.py modules)
provides:
  - quantize.py with full pipeline integration: hardware detection, binary factor collection, safetensors save, ETA progress, bits_per_weight
  - run_stage1.py CLI with --bits-per-weight flag and auto hardware detection
affects:
  - All subsequent phases that invoke quantize_model()
  - Phase 2+ that read quantized_factors.safetensors output

tech-stack:
  added: []
  patterns:
    - probe_hardware() called once at pipeline start; device overrides CLI arg
    - all_quantized dict collects binary factors across all blocks; saved in one safetensors file
    - Per-block ETA computed as avg_time * remaining_blocks (rolling average)
    - bits_per_weight -> rank formula: rank = max(1, int(bpw * d / 2)) using first Linear layer d

key-files:
  created: []
  modified:
    - nanoquant/quantize.py
    - scripts/run_stage1.py

key-decisions:
  - "psutil import moved to top-level from inline Phase 3 KD block — consistent with module-level imports"
  - "eval_device in run_stage1.py derives from torch.cuda.is_available() — independent from quantize_model device (which uses probe_hardware)"
  - "Effective bpw computed from U_bin/V_bin numel ratio after all blocks — provides post-hoc verification of bits_per_weight target"

requirements-completed: [PIPE-01, PIPE-03, PIPE-05]

duration: 10min
completed: 2026-02-22
---

# Phase 01 Plan 02: Pipeline Integration Summary

**quantize.py now calls probe_hardware() at startup, collects binary factors block-by-block, saves safetensors checkpoint alongside FP16 model, prints per-block ETA + VRAM/RAM, and accepts bits_per_weight for sub-1-bit rank computation; run_stage1.py exposes --bits-per-weight and removes --device**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-02-22
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `probe_hardware()` replaces manual GPU check block; `device` variable overridden with auto-detected value
- `all_quantized` dict accumulates binary factors from each block's `reconstruct_block()` return value
- `save_quantized_checkpoint()` called after `model.save_pretrained()` — both FP16 and safetensors written
- Per-block progress line: `Block N/M | Xs | ETA Xm Xs | VRAM X.XGB | RAM X.XGB | N layers`
- Effective bpw computed from actual U_bin/V_bin tensor sizes and printed after save
- `bits_per_weight` parameter computes rank via `max(1, int(bpw * d / 2))` using first Linear layer
- `--bits-per-weight` CLI flag added to run_stage1.py; `--device` flag removed
- eval_device for perplexity evaluation derived from `torch.cuda.is_available()`

## Task Commits

1. **Task 1: Integrate hardware detection, binary factor collection, checkpoint save, and progress logging into quantize.py** - `3c39cee` (feat)
2. **Task 2: Update run_stage1.py CLI with --bits-per-weight flag and pass hardware-aware defaults** - `bfc39fa` (feat)

## Files Modified

- `nanoquant/quantize.py` - probe_hardware, all_quantized collection, save_quantized_checkpoint, per-block ETA, bits_per_weight param
- `scripts/run_stage1.py` - --bits-per-weight arg, removed --device arg, eval_device from cuda detection

## Decisions Made

- psutil import moved to top-level from inline Phase 3 KD block
- eval_device in run_stage1.py derives independently from quantize_model hardware detection
- Effective bpw printed post-hoc from actual tensor sizes for verification

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Self-Check: PASSED

- `nanoquant/quantize.py` exists and imports cleanly
- `scripts/run_stage1.py` --help shows --bits-per-weight, no --device
- All four integrations confirmed via grep: probe_hardware, save_quantized_checkpoint, ETA, bits_per_weight
- Commits 3c39cee and bfc39fa exist in git log

---
*Phase: 01-pipeline-foundation*
*Completed: 2026-02-22*
