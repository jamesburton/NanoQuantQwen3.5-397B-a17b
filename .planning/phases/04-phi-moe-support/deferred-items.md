# Deferred Items - Phase 04: Phi MoE Support

## From 04-02-PLAN.md (Task 4)

### Task 4: Measure WikiText-2 perplexity (FP16 baseline + quantized)

**Status:** Blocked on GPU hardware + quantized_factors.safetensors file
**Requirement:** PHI-03

**Issue:** The quantized_factors.safetensors file is not present on disk (gitignored, large binary produced during GPU quantization run). The evaluation pipeline (`scripts/run_eval.py`) requires this file to reconstruct the model and measure perplexity. Additionally, no CUDA GPU is available in the current environment.

**Resolution steps:**
1. On a machine with GPU and the quantized factors file, run:
   ```bash
   python scripts/run_eval.py \
     --model microsoft/Phi-tiny-MoE-instruct \
     --quantized-dir output/Phi-tiny-MoE-instruct \
     --results-dir results
   ```
2. This will produce `results/Phi-tiny-MoE-instruct/metrics.json` and update `results/SUMMARY.md`
3. Commit the results files

**Note:** The `--quantized-dir` argument name differs from the plan's `--checkpoint` — use `--quantized-dir` as that is the actual CLI argument in `run_eval.py`.
