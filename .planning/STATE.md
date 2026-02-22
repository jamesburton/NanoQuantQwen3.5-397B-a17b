# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Consumer-friendly sub-1-bit quantization of MoE models via NanoQuant's ADMM-based pipeline
**Current focus:** Phase 1 - Pipeline Foundation

## Current Position

Phase: 1 of 3 (Pipeline Foundation)
Plan: 2 of 2 in current phase
Status: Phase 1 complete
Last activity: 2026-02-22 — Completed plan 01-02 (quantize.py + run_stage1.py integration)

Progress: [██░░░░░░░░] 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

*Updated after each plan completion*

## Accumulated Context

### Decisions

- model_load_strategy always "cpu" — block-by-block .to(device) is correct for quantization; device_map="auto" scatters layers and breaks sequential processing
- Safetensors dotted key format (layer_name.tensor_key) chosen for MoE expert forward-compatibility
- 1 GB GPU headroom + 10% CPU headroom hardcoded as constants, not user-configurable
- psutil import at module top-level in quantize.py (was inline in Phase 3 KD block)
- eval_device in run_stage1.py derives from torch.cuda.is_available() independently of quantize_model hardware detection
- bits_per_weight -> rank: rank = max(1, int(bpw * d / 2)) using first Linear layer's min dimension

### Pending Todos

None yet.

### Blockers/Concerns

- SCALE-04 (Qwen3.5-397B) requires cloud GPU — consumer 12GB constraint applies to phases 1-2 and SCALE-01 through SCALE-03 only

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 01-02-PLAN.md (quantize.py + run_stage1.py pipeline integration)
Resume file: None
