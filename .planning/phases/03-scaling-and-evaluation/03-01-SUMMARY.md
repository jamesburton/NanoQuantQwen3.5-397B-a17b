---
phase: 03-scaling-and-evaluation
plan: 01
subsystem: evaluation
tags: [perplexity, wikitext-2, sliding-window, metrics, json]

requires:
  - phase: 02-moe-support
    provides: "MoE-aware quantization pipeline with architecture detection"
provides:
  - "Canonical sliding-window PPL evaluation with -100 context masking"
  - "Unified evaluation runner (run_eval.py) with JSON metrics output"
  - "Auto-generated SUMMARY.md comparison table (build_summary.py)"
affects: [03-02, 03-03]

tech-stack:
  added: []
  patterns: ["HuggingFace canonical sliding-window PPL with -100 masking", "Structured JSON metrics per model"]

key-files:
  created:
    - scripts/run_eval.py
    - scripts/build_summary.py
  modified:
    - scripts/eval_ppl.py

key-decisions:
  - "evaluate_perplexity returns dict (ppl, n_tokens, settings) for structured downstream consumption"
  - "dtype='auto' default for baseline eval respects model native dtype (e.g. bfloat16 for Qwen3.5-397B)"

patterns-established:
  - "Metrics JSON schema: model, hardware, model_size, timing, perplexity, quantization_config"
  - "SUMMARY.md rebuilt idempotently from all results/*/metrics.json via glob"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03]

duration: 2min
completed: 2026-02-25
---

# Phase 3 Plan 1: Evaluation Infrastructure Summary

**Canonical sliding-window PPL evaluation with -100 context masking, unified eval runner with JSON metrics, and auto-generated comparison table**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-24T23:59:56Z
- **Completed:** 2026-02-25T00:02:13Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Rewrote eval_ppl.py to use HuggingFace canonical sliding-window PPL with -100 context masking
- Created run_eval.py unified runner capturing FP16 baseline + quantized PPL + memory/timing metrics
- Created build_summary.py for idempotent SUMMARY.md generation from all metrics.json files

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite eval_ppl.py with canonical sliding-window PPL** - `19e49a0` (feat)
2. **Task 2: Create run_eval.py and build_summary.py** - `b2360c7` (feat)

## Files Created/Modified
- `scripts/eval_ppl.py` - Canonical sliding-window PPL with -100 masking, returns dict, supports --dtype auto/float16
- `scripts/run_eval.py` - Unified evaluation runner: FP16 baseline + quantized PPL + memory + JSON output
- `scripts/build_summary.py` - Generates results/SUMMARY.md comparison table from all metrics.json

## Decisions Made
- evaluate_perplexity() returns dict instead of float for structured metrics consumption
- dtype="auto" as default for baseline eval to respect model native dtype (bfloat16 for Qwen3.5-397B)
- run_eval.py uses try/except around each PPL eval to continue with partial results on failure

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Evaluation infrastructure ready for model-by-model scaling runs (plans 03-02, 03-03)
- Scripts are hardware-agnostic: same code runs locally and on cloud

## Self-Check: PASSED

All files verified present. All commits verified in git log.

---
*Phase: 03-scaling-and-evaluation*
*Completed: 2026-02-25*
