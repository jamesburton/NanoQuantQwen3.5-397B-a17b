# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Consumer-friendly sub-1-bit quantization of MoE models via NanoQuant's ADMM-based pipeline
**Current focus:** Phase 3 - Scaling and Evaluation

## Current Position

Phase: 4 of 4 (Phi MoE Support)
Plan: 2 of 2 in current phase
Status: In Progress
Last activity: 2026-02-28 — Completed plan 04-01 (PhimoeExperts fused gate_up_proj split support)

Progress: [█████████░] 90%

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
| Phase 02-moe-support P02 | 10 | 2 tasks | 3 files |
| Phase 03 P01 | 2 | 2 tasks | 3 files |

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
- [Phase 04-phi-moe-support P01]: FUSED_EXPERT_CLASSES migrated from frozenset to dict with needs_split/split_dim metadata — each architecture declares its own fused tensor layout
- [Phase 04-phi-moe-support P01]: PhimoeExperts gate_up_proj split_dim=0 means dim 0 of per-expert 2D slice (out-dim); gate=[:half,:], up=[half:,:]
- [Phase 04-phi-moe-support P01]: gate_proj[N] and up_proj[N] split views both map to same Hessian key — both halves share the same expert input activations
- [Phase 02-moe-support P02]: shared. prefix chosen for norm/embedding keys to distinguish from binary factor dotted keys in safetensors
- [Phase 02-moe-support P02]: collect_shared_layers uses named_modules traversal with class-name check to stay architecture-agnostic
- [Phase 03]: evaluate_perplexity returns dict (ppl, n_tokens, settings) for structured downstream consumption
- [Phase 03]: dtype=auto default for baseline eval respects model native dtype (e.g. bfloat16 for Qwen3.5-397B)

### Roadmap Evolution

- Phase 4 added: Phi MoE Support — PhimoeExperts fused gate_up_proj layout + Phi-tiny-MoE-instruct local validation

### Pending Todos

None yet.

### Blockers/Concerns

- SCALE-04 (Qwen3.5-397B) requires cloud GPU — consumer 12GB constraint applies to phases 1-2 and SCALE-01 through SCALE-03 only

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 04-01-PLAN.md (PhimoeExperts fused gate_up_proj split support)
Resume file: None
