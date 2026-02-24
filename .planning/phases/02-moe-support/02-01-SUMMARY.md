---
phase: 02-moe-support
plan: 01
subsystem: quantization
tags: [moe, fused-experts, dispatch, olmoe, qwen2, qwen3, qwen3.5]

# Dependency graph
requires:
  - phase: 01-pipeline-foundation
    provides: nanoquant/moe.py with Qwen2MoeExperts WeightView and Hessian key logic
provides:
  - FUSED_EXPERT_CLASSES frozenset covering OlmoeExperts, Qwen2MoeExperts, Qwen3MoeExperts, Qwen3_5MoeExperts
  - get_weight_views dispatches on FUSED_EXPERT_CLASSES (not hardcoded string)
  - Ancestor check in nn.Linear path uses FUSED_EXPERT_CLASSES to prevent double-yielding
affects: [02-02, reconstruct.py, quantize.py]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FUSED_EXPERT_CLASSES frozenset as dispatch table for fused 3D expert module type matching"

key-files:
  created: []
  modified:
    - nanoquant/moe.py

key-decisions:
  - "Use frozenset dispatch table (FUSED_EXPERT_CLASSES) instead of hardcoded string for expert type matching — enables adding new architectures by updating one set"
  - "No logic change to weight slicing, write-back, or Hessian key computation — all four fused expert classes use identical gate_up_proj/down_proj 3D layout"

patterns-established:
  - "FUSED_EXPERT_CLASSES: single source of truth for all known fused expert module class names"

requirements-completed: [MOE-01, MOE-02, MOE-04]

# Metrics
duration: 5min
completed: 2026-02-24
---

# Phase 02 Plan 01: MoE Dispatch Table Summary

**FUSED_EXPERT_CLASSES frozenset replaces hardcoded "Qwen2MoeExperts" string, enabling OlmoeExperts, Qwen3MoeExperts, and Qwen3_5MoeExperts support via a single dispatch table in nanoquant/moe.py**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-24T08:57:32Z
- **Completed:** 2026-02-24T09:02:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added FUSED_EXPERT_CLASSES frozenset at module level with all four fused expert class names
- Replaced `type_name == "Qwen2MoeExperts"` with `type_name in FUSED_EXPERT_CLASSES` in get_weight_views
- Replaced ancestor check `== "Qwen2MoeExperts"` with `in FUSED_EXPERT_CLASSES` to prevent double-yielding for all four architectures
- Updated docstrings for get_weight_views and get_hessian_key to name all four expert class types

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace hardcoded Qwen2MoeExperts with FUSED_EXPERT_CLASSES dispatch table** - `904917d` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `nanoquant/moe.py` - FUSED_EXPERT_CLASSES frozenset added; get_weight_views and ancestor check updated to use in-set dispatch; docstrings updated

## Decisions Made
- Use frozenset for FUSED_EXPERT_CLASSES to get O(1) lookup and immutability — new architectures require only adding a string to the set
- No functional logic changes to weight slicing or Hessian key computation — all four classes share identical 3D parameter layout

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- moe.py now supports OLMoE, Qwen2-MoE, Qwen3-MoE, and Qwen3.5-MoE architectures via dispatch table
- reconstruct.py imports get_weight_views and will automatically benefit from the broader dispatch
- Ready for plan 02-02 (Hessian capture / reconstruct.py MoE path updates)

---
*Phase: 02-moe-support*
*Completed: 2026-02-24*
