---
phase: 01-pipeline-foundation
plan: 01
subsystem: infra
tags: [hardware, safetensors, checkpoint, accelerate, psutil, quantization]

requires: []
provides:
  - HardwarePlan dataclass and probe_hardware() for auto-detecting GPU/CPU/disk resources
  - save_quantized_checkpoint() / load_quantized_checkpoint() for binary factor persistence via safetensors
affects:
  - 01-02 (quantize.py integration — imports both modules)
  - All subsequent phases that read/write quantized factor checkpoints

tech-stack:
  added: [safetensors, accelerate.utils.get_max_memory, psutil]
  patterns:
    - Block-by-block CPU load strategy (model_load_strategy="cpu") for quantization pipeline
    - Dotted key notation for safetensors (layer_name.tensor_key) enables MoE expert naming
    - HardwarePlan dataclass as typed config object passed through pipeline

key-files:
  created:
    - nanoquant/hardware.py
    - nanoquant/checkpoint.py
  modified: []

key-decisions:
  - "model_load_strategy is always 'cpu' — block-by-block .to(device) is correct for quantization; device_map='auto' scatters layers and breaks sequential processing"
  - "Safetensors dotted key format (layer.tensor_key) chosen for MoE forward-compatibility — no square brackets"
  - "1 GB GPU headroom reserved for activations; 10% CPU headroom for OS — these are hardcoded constants, not configurable"
  - "hessian_on_disk=False hardcoded for Phase 1 — Hessian diagonals are tiny (1D vectors) so no disk offload needed"

patterns-established:
  - "HardwarePlan pattern: probe once at startup, pass typed plan through pipeline — no scattered torch.cuda.is_available() checks"
  - "Safetensors roundtrip: always call .contiguous().cpu() before save_file to avoid layout/device issues"

requirements-completed: [PIPE-02, PIPE-04]

duration: 8min
completed: 2026-02-22
---

# Phase 01 Plan 01: Hardware Detection and Checkpoint Serialization Summary

**HardwarePlan dataclass + probe_hardware() auto-detecting GPU/CPU/disk resources via accelerate, and safetensors-based save/load for binary quantization factors with dotted MoE-compatible key naming**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-02-22T07:26:52Z
- **Completed:** 2026-02-22T07:34:50Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `HardwarePlan` dataclass captures GPU (with headroom reserve), CPU (90% available), disk, and derived strategy fields
- `probe_hardware()` calls `accelerate.utils.get_max_memory()` with psutil fallback; handles CPU-only machines gracefully
- `print_hardware_summary()` outputs a readable one-liner (GPU/CPU/disk GB + strategy)
- `save_quantized_checkpoint()` flattens nested layer dict to dotted safetensors keys with metadata
- `load_quantized_checkpoint()` uses `safe_open` lazy loading; parses layer/tensor split on last dot
- Roundtrip verified: all tensor values (int8 U_bin/V_bin, float s1/s2) survive save/load identically

## Task Commits

Each task was committed atomically:

1. **Task 1: Create hardware auto-detection module** - `7b25d88` (feat)
2. **Task 2: Create safetensors checkpoint module** - `dcf82bb` (feat)

## Files Created/Modified
- `nanoquant/hardware.py` - HardwarePlan dataclass, probe_hardware(), print_hardware_summary()
- `nanoquant/checkpoint.py` - save_quantized_checkpoint(), load_quantized_checkpoint()

## Decisions Made
- model_load_strategy is always "cpu" — confirmed correct for block-by-block quantization; device_map="auto" would scatter layers across devices and break sequential processing
- Dotted safetensors key notation chosen over bracket notation for MoE expert compatibility
- 1 GB GPU headroom and 10% CPU headroom hardcoded as constants (not user-configurable) per plan spec

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both modules are pure additions; no existing files modified
- `nanoquant/hardware.py` and `nanoquant/checkpoint.py` ready to be imported by `quantize.py` in Plan 02
- No blockers

---
*Phase: 01-pipeline-foundation*
*Completed: 2026-02-22*
