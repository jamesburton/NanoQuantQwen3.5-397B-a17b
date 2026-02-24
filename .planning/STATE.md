# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Consumer-friendly sub-1-bit quantization of MoE models via NanoQuant's ADMM-based pipeline
**Current focus:** Phase 2 - MoE Support

## Current Position

Phase: 2 of 3 (MoE Support)
Plan: 1 of 2 in current phase
Status: In Progress
Last activity: 2026-02-24 — Completed plan 02-01 (FUSED_EXPERT_CLASSES dispatch table for MoE expert types)

Progress: [███░░░░░░░] 30%

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
| Phase 01-pipeline-foundation P03 | 2 | 1 tasks | 2 files |
| Phase 02-moe-support P01 | 5 | 1 tasks | 1 files |

## Accumulated Context

### Decisions

- model_load_strategy always "cpu" — block-by-block .to(device) is correct for quantization; device_map="auto" scatters layers and breaks sequential processing
- Safetensors dotted key format (layer_name.tensor_key) chosen for MoE expert forward-compatibility
- 1 GB GPU headroom + 10% CPU headroom hardcoded as constants, not user-configurable
- psutil import at module top-level in quantize.py (was inline in Phase 3 KD block)
- eval_device in run_stage1.py derives from torch.cuda.is_available() independently of quantize_model hardware detection
- bits_per_weight -> rank: rank = max(1, int(bpw * d / 2)) using first Linear layer's min dimension
- [Phase 01-pipeline-foundation]: print_hardware_summary shows compute= and model_load= as separate fields for clarity
- [Phase 02-moe-support]: Use frozenset dispatch table (FUSED_EXPERT_CLASSES) for fused expert module type matching — enables adding new architectures by updating one set

### Pending Todos

None yet.

### Blockers/Concerns

- SCALE-04 (Qwen3.5-397B) requires cloud GPU — consumer 12GB constraint applies to phases 1-2 and SCALE-01 through SCALE-03 only

## Session Continuity

Last session: 2026-02-24
Stopped at: Completed 02-01-PLAN.md (FUSED_EXPERT_CLASSES dispatch table for MoE expert types)
Resume file: None
