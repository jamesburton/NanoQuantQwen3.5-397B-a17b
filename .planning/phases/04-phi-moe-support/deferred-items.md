# Deferred Items - Phase 04: Phi MoE Support

## From 04-02-PLAN.md (Task 4) — CLOSED

### Task 4: Measure WikiText-2 perplexity (FP16 baseline + quantized)

**Status:** Closed (completed in 04-03)
**Requirement:** PHI-03
**Resolution:** Ran eval on CPU with float32 fallback. Results in `results/Phi-tiny-MoE-instruct/metrics.json`.

**Note:** The CPU eval used zero-refinement quantization settings, producing 141x PPL degradation. See `CUDA-CONTINUATION.md` for instructions to re-run with full refinement on CUDA for production-quality results.
