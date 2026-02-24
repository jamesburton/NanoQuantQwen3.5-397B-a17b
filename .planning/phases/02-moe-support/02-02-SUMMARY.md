---
phase: 02-moe-support
plan: 02
subsystem: quantization
tags: [safetensors, checkpoint, shared-layers, rmsnorm, embeddings, architecture-detection, moe]

requires:
  - phase: 01-pipeline-foundation
    provides: save_quantized_checkpoint, quantize_model pipeline

provides:
  - collect_shared_layers function collecting RMSNorm/LayerNorm/embed_tokens/lm_head as FP16
  - save_quantized_checkpoint with optional shared_layers merge for self-contained files
  - detect_architecture via AutoConfig for model_type logging
  - --parallel-blocks CLI stub in run_stage1.py

affects:
  - 02-moe-support
  - reconstruct phase (will use shared layers for expert reconstruction context)

tech-stack:
  added: []
  patterns:
    - "shared. prefix separates norm/embedding tensors from binary factor tensors in safetensors"
    - "collect_shared_layers always called before save for self-contained checkpoints"
    - "detect_architecture uses AutoConfig.from_pretrained with trust_remote_code=True"

key-files:
  created: []
  modified:
    - nanoquant/checkpoint.py
    - nanoquant/quantize.py
    - scripts/run_stage1.py

key-decisions:
  - "shared. prefix chosen for norm/embedding keys to distinguish from binary factor dotted keys"
  - "collect_shared_layers uses named_modules traversal for RMSNorm/LayerNorm (class name check) to be architecture-agnostic"
  - "lm_head key stored as shared.model.lm_head.weight (model. prefix) for consistency with embed_tokens path"
  - "--parallel-blocks stub added as int default=1 (sequential); parallel logic is Phase 3 scope"

patterns-established:
  - "Self-contained checkpoint pattern: collect_shared_layers -> pass shared_layers= to save_quantized_checkpoint"
  - "Architecture detection at startup via detect_architecture() call after model load"

requirements-completed: [MOE-03]

duration: 10min
completed: 2026-02-24
---

# Phase 02 Plan 02: Self-Contained Checkpoint + Architecture Detection Summary

**Self-contained safetensors checkpoint via collect_shared_layers saving FP16 RMSNorm/embed/lm_head tensors alongside binary factors, plus AutoConfig-based architecture detection logged at model load**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-02-24T00:00:00Z
- **Completed:** 2026-02-24T00:10:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `collect_shared_layers(model)` that traverses `named_modules()` for all RMSNorm/LayerNorm weights/biases, plus `embed_tokens`, `lm_head`, and final `norm` as FP16 CPU tensors with `shared.` prefix
- Updated `save_quantized_checkpoint` to accept optional `shared_layers` dict, merge into safetensors flat dict, and record `shared_layers` key list in manifest JSON
- Added `detect_architecture()` using `AutoConfig.from_pretrained` and integrated call in `quantize_model` with printed `Architecture: {arch_type}` log line
- Added `--parallel-blocks` (int, default 1) CLI stub to `run_stage1.py` for MOE-04

## Task Commits

Each task was committed atomically:

1. **Task 1: Add collect_shared_layers to checkpoint.py** - `b177ebb` (feat)
2. **Task 2: Add architecture detection and shared layer save to quantize.py** - `b42b3ba` (feat)

## Files Created/Modified

- `nanoquant/checkpoint.py` - Added `collect_shared_layers`, `Optional[Dict]` param to `save_quantized_checkpoint`, `shared_layers` in manifest
- `nanoquant/quantize.py` - Added `detect_architecture`, import `AutoConfig` + `collect_shared_layers`, call both in `quantize_model` save section
- `scripts/run_stage1.py` - Added `--parallel-blocks` argparse stub

## Decisions Made

- `shared.` prefix for norm/embedding keys so they are distinguishable from binary factor dotted keys (e.g. `model.layers.0.self_attn.q_proj.U_bin`) in the same flat safetensors file
- Used `named_modules()` with class-name string check (`'RMSNorm' in type(module).__name__`) to stay architecture-agnostic across Qwen2, LLaMA, etc.
- `lm_head` stored as `shared.model.lm_head.weight` (with `model.` prefix) to mirror the embed_tokens path convention
- `--parallel-blocks` stub only — actual parallel expert logic is Phase 3 scope per plan spec

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- MOE-03 complete: quantized safetensors checkpoint is now self-contained with norms and embeddings
- Ready for MOE-04 (expert block grouping) and MOE-05 (MoE-aware Hessian capture)
- `detect_architecture()` in place for future architecture-specific branch logic

---
*Phase: 02-moe-support*
*Completed: 2026-02-24*
