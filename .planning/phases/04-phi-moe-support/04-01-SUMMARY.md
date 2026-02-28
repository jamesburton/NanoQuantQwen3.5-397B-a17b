---
phase: 04-phi-moe-support
plan: 01
subsystem: moe
tags: [phimoe, fused-experts, gate-up-proj, weight-views, quantization]

requires:
  - phase: 02-moe-support
    provides: FUSED_EXPERT_CLASSES frozenset, get_weight_views, get_hessian_key baseline

provides:
  - FUSED_EXPERT_CLASSES dict with layout metadata (needs_split, split_dim)
  - PhimoeExperts split gate_up_proj WeightView logic in moe.py
  - get_hessian_key support for .gate_proj[N] and .up_proj[N] split view names
  - reconstruct_model.py fused gate_up_proj reassembly from split gate_proj/up_proj checkpoint keys

affects: [05-validation, any future quantization runs on Phi-3.5-MoE]

tech-stack:
  added: []
  patterns:
    - "Layout metadata dict dispatches on needs_split to choose single vs split WeightView emission"
    - "Split views named .gate_proj[N] / .up_proj[N]; Hessian key maps both to parent fused expert module"
    - "Reassembly is exact inverse of split: torch.cat([gate, up], dim=0) -> fused gate_up_proj[idx]"

key-files:
  created: []
  modified:
    - nanoquant/moe.py
    - scripts/reconstruct_model.py

key-decisions:
  - "FUSED_EXPERT_CLASSES migrated from frozenset to dict with needs_split/split_dim metadata to support heterogeneous fused layouts without branching on class name throughout code"
  - "split_dim=0 for PhimoeExperts means split along out-dim (dim 1 of 3D tensor = dim 0 of 2D per-expert slice): gate=[:half,:], up=[half:,:]"
  - "get_hessian_key maps both .gate_proj[N] and .up_proj[N] to the same parent fused expert Hessian key — both halves share the same input Hessian"

patterns-established:
  - "Layout metadata pattern: add new fused architectures by inserting one entry into FUSED_EXPERT_CLASSES dict"
  - "Inverse-op pattern: reconstruct_model.py reassembly must mirror exactly the split done in get_weight_views"

requirements-completed: [PHI-01]

duration: 15min
completed: 2026-02-28
---

# Phase 4 Plan 01: Phi MoE fused gate_up_proj split support Summary

**FUSED_EXPERT_CLASSES migrated from frozenset to dict with needs_split metadata; PhimoeExperts gate_up_proj split into separate gate_proj[i]/up_proj[i] WeightViews for quantization and reassembled in reconstruct_model.py**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-02-28T00:00:00Z
- **Completed:** 2026-02-28T00:15:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Converted FUSED_EXPERT_CLASSES from frozenset to dict with per-class layout metadata (needs_split, split_dim), keeping all existing entries (OlmoeExperts, Qwen*) working identically with needs_split=False
- Added PhimoeExperts entry (needs_split=True, split_dim=0); get_weight_views now yields two WeightViews per expert (gate_proj[i], up_proj[i]) with correct write-back closures targeting the correct half of the fused tensor
- Updated get_hessian_key to map .gate_proj[N] and .up_proj[N] split view names to the parent fused expert module Hessian key
- Added split view reassembly in reconstruct_model.py: detects .gate_proj / .up_proj checkpoint key patterns, concatenates gate+up along out-dim, writes back to fused gate_up_proj[expert_idx]

## Task Commits

1. **Task 1: Migrate FUSED_EXPERT_CLASSES to dict and add PhimoeExperts split logic** - `8d29ac1` (feat)
2. **Task 2: Add fused gate_up_proj reassembly to reconstruct_model.py** - `0221886` (feat)

## Files Created/Modified

- `nanoquant/moe.py` - FUSED_EXPERT_CLASSES dict, split WeightView logic, updated get_hessian_key
- `scripts/reconstruct_model.py` - Split view detection and gate_up_proj reassembly

## Decisions Made

- FUSED_EXPERT_CLASSES converted to dict with `needs_split` / `split_dim` metadata so each architecture declares its own layout — avoids scattered class-name checks
- `split_dim=0` documents the axis convention: dim 0 of the 2D per-expert slice (= out-dim) is halved; gate = first half rows, up = second half rows
- Both gate_proj[N] and up_proj[N] views map to the same Hessian key because both halves of the fused parameter share the same expert input activations

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PhimoeExperts fused gate_up_proj path is fully implemented for quantization and reconstruction
- Phi-tiny-MoE-instruct (SlimMoE, separate nn.Linear experts) already works via the existing nn.Linear path — ready for plan 04-02 local validation

---
*Phase: 04-phi-moe-support*
*Completed: 2026-02-28*
