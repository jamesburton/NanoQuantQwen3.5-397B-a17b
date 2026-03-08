---
phase: 04-phi-moe-support
plan: 02
subsystem: quantization
tags: [phi-moe, slimmoe, trust-remote-code, quantization, eval]

requires:
  - phase: 04-01
    provides: PhimoeExperts fused gate_up_proj split support in FUSED_EXPERT_CLASSES
provides:
  - Phi-tiny-MoE-instruct quantized checkpoint (32 blocks, 16 experts, rank=8)
  - SlimMoE transformers>=5.x compatibility patch script
  - trust_remote_code=True on all from_pretrained calls (eval + reconstruct pipelines)
affects: [03-scaling-evaluation]

tech-stack:
  added: []
  patterns:
    - SlimMoE compatibility patching for transformers>=5.x class renames
    - trust_remote_code=True for all custom-code HF models

key-files:
  created:
    - scripts/patch_slimmoe_compat.py
    - output/Phi-tiny-MoE-instruct/quantized_manifest.json
    - output/Phi-tiny-MoE-instruct/config.json
  modified:
    - scripts/eval_ppl.py
    - scripts/reconstruct_model.py
    - scripts/run_eval.py

key-decisions:
  - "trust_remote_code=True added to all from_pretrained calls during 04-01, not 04-02 — Task 3 was a no-op"
  - "SlimMoE uses nn.Linear experts (w1,w2,w3 per expert), not fused 3D tensors — handled by existing nn.Linear path"

patterns-established:
  - "Compatibility patch scripts for upstream HF model repos (patch_slimmoe_compat.py pattern)"

requirements-completed: [PHI-02]

duration: ~45min (quantization run, prior session)
completed: 2026-03-08
---

# Phase 4 Plan 02: Phi-tiny-MoE-instruct Quantization Summary

**Phi-tiny-MoE-instruct (3.8B params, 16 experts) fully quantized end-to-end on consumer GPU; eval deferred pending GPU re-run**

## Performance

- **Duration:** Spread across multiple sessions (quantization on GPU, code verification now)
- **Started:** Prior session (quantization run)
- **Completed:** 2026-03-08T02:35:41Z (code tasks)
- **Tasks:** 3/4 completed (Task 4 deferred - GPU required)
- **Files modified:** 4

## Accomplishments
- Phi-tiny-MoE-instruct quantized end-to-end: 32 blocks, 16 experts per block, rank=8 ADMM
- SlimMoE transformers>=5.x compatibility patch script created (handles class renames in upstream HF repo)
- All `from_pretrained` calls verified to have `trust_remote_code=True` across eval/reconstruct/run_eval pipelines
- Manifest confirms 1957 quantized weight tensors with 444 factor tensors in checkpoint

## Task Commits

Each task was committed atomically:

1. **Task 1: Smoke test - quantize first 2 blocks** - `33d9753` (fix: SlimMoE compat patch)
2. **Task 2: Full quantization of Phi-tiny-MoE-instruct** - `4acd185` (feat: full 32-block quantization)
3. **Task 3: Patch trust_remote_code into eval/reconstruct** - No commit needed (already patched in prior 04-01 work)
4. **Task 4: Measure WikiText-2 perplexity** - DEFERRED (requires GPU + quantized_factors.safetensors)

## Files Created/Modified
- `scripts/patch_slimmoe_compat.py` - Patches SlimMoE HF repo for transformers>=5.x compatibility
- `output/Phi-tiny-MoE-instruct/quantized_manifest.json` - Quantization manifest (32 blocks, 16 experts)
- `output/Phi-tiny-MoE-instruct/config.json` - Model config for checkpoint
- `scripts/eval_ppl.py` - Already had trust_remote_code=True (verified)
- `scripts/reconstruct_model.py` - Already had trust_remote_code=True (verified)
- `scripts/run_eval.py` - No from_pretrained calls (uses eval_ppl.py)

## Decisions Made
- Task 3 was a no-op: all `from_pretrained` calls already had `trust_remote_code=True` from 04-01 work
- SlimMoE uses separate `nn.Linear` experts (w1, w2, w3 per expert), not fused 3D tensors, so existing nn.Linear traversal handles them without PhimoeExperts fused path

## Deviations from Plan

### Task 3 No-Op

**Task 3 (trust_remote_code patch):** All `from_pretrained` calls in eval_ppl.py, reconstruct_model.py, and run_eval.py already included `trust_remote_code=True` from prior commits (04-01 phase work). No code changes were needed. Verification passed - grep confirms all calls are patched.

### Task 4 Deferred

**Task 4 (WikiText-2 eval):** Blocked because:
1. `quantized_factors.safetensors` is not on disk (gitignored large binary, produced during GPU quantization)
2. No CUDA GPU available in current environment
3. Requires re-running quantization or running on original GPU machine

See `deferred-items.md` for resolution steps.

---

**Total deviations:** 1 no-op task (Task 3), 1 deferred task (Task 4)
**Impact on plan:** PHI-02 (quantization) is satisfied. PHI-03 (PPL eval) requires GPU re-run.

## Issues Encountered
- quantized_factors.safetensors is gitignored and not present on disk, blocking eval pipeline
- No Python/CUDA available in current shell environment

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- PHI-02 complete: Phi-tiny-MoE-instruct quantizes end-to-end on consumer GPU
- PHI-03 pending: Run eval pipeline on GPU machine with factors file to measure perplexity
- Command for eval: `python scripts/run_eval.py --model microsoft/Phi-tiny-MoE-instruct --quantized-dir output/Phi-tiny-MoE-instruct --results-dir results`

## Self-Check: PASSED

- FOUND: 04-02-SUMMARY.md
- FOUND: deferred-items.md
- FOUND: commit 4acd185 (Task 2)
- FOUND: commit 33d9753 (Task 1)

---
*Phase: 04-phi-moe-support*
*Completed: 2026-03-08*
