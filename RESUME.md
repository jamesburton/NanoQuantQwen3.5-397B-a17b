# Resume Notes

## Current Status (2026-03-13)

The signed-fit full rerun eval is **complete**. Results are still catastrophically bad.

## Evidence Collected

### Progression of full-run results

| Run | Path | Baseline PPL | Quant PPL | Degradation | Notes |
|-----|------|-------------|-----------|-------------|-------|
| Old stable (rerun_20260311) | `output/Phi-tiny-MoE-instruct_rerun_20260311` | 7.1747 | 1,574,286.75 | 219,420x | Reconstruction semantics fixed; signed-scale bug still present |
| Signed-fit full (20260312) | `output/Phi-tiny-MoE-instruct_signedfit_full_20260312` | 7.1747 | 218,212.22 | 30,413x | Both bugs fixed; full 32 blocks |
| **Signed-fit 2-block ablation** | `output/Phi-tiny-MoE-instruct_ablate_signedfit_mb2` | 7.1747 | **15.15** | **2.11x** | Only 2/32 blocks quantized |

### Why the 2-block ablation is misleading

2/32 blocks quantized means ~94% of model weights are unmodified FP16.
The 2.11x result shows the signed-scale patch works per-layer but errors
compound catastrophically across all 32 blocks. Quantization error from
each block propagates forward as distorted activations into the next block's
ADMM solve, degrading each successive block more than an isolated fit would.

### Bugs Fixed (all committed, no regressions)

1. **Reconstruction semantics** (`nanoquant/reconstruct.py`): `reconstruct_weight()` was
   applying stored preconditioners as inverses; now multiplies, matching how W_tilde is formed.
2. **Signed scale fitting** (`nanoquant/admm.py`): `_fit_balanced_scales()` fit against
   `abs(W_target)`; now fits against the signed target.
3. **Factor key collision** (`nanoquant/quantize.py`): block-relative keys caused all 32 blocks
   to overwrite each other in `all_quantized`; fixed with `block_prefix`.
4. **Fail-fast guard** (`nanoquant/quantize.py`): aborts immediately on non-finite block output
   to prevent silent corruption of downstream checkpoints.

## Root Cause Hypothesis

The signed-scale fix clearly improves per-layer approximation quality (2.11x on 2 blocks).
The full 32-block run still being catastrophic points to **error compounding** — each block's
quantized output diverges from FP16, and subsequent blocks are optimized against distorted
inputs. This is a fundamental challenge in sequential block-wise quantization with no
end-to-end feedback.

Possible contributing factors (unverified):
- **Activation drift**: the block_input fed into ADMM is the FP16 output of the previous
  quantized block, not the true FP16 intermediate. Errors accumulate multiplicatively.
- **Rank too low**: rank=8 binary approximation may not be sufficient for the expert weight
  shapes in Phi-tiny-MoE. Expert W matrices are 2048×(4096 or 2048); rank 8 binary gives
  effective bpw ≈ 8*(m+n)/(m*n) ≈ very sub-1-bit.
- **ADMM quality per-layer**: even with signed scales, the binary factorization may diverge
  badly for some layer shapes. Per-layer reconstruction error is not currently measured.

## Immediate Next Steps

### Step 1: Measure per-layer reconstruction quality

Before changing anything, understand *where* the degradation originates.

```bash
python - <<'EOF'
import torch
from safetensors.torch import load_file

factors = load_file("output/Phi-tiny-MoE-instruct_signedfit_full_20260312/quantized_factors.safetensors")

# Group by layer, compute relative Frobenius error
from collections import defaultdict
import re

layer_data = defaultdict(dict)
for k, v in factors.items():
    # Keys: model.layers.N.layer_name.{U_bin,V_bin,s1,s2,d_in,d_out}
    match = re.match(r'(.+)\.(U_bin|V_bin|s1|s2|d_in|d_out)$', k)
    if match:
        layer_data[match.group(1)][match.group(2)] = v

errors = []
for layer_name, d in layer_data.items():
    if not all(k in d for k in ('U_bin', 'V_bin', 's1', 's2', 'd_in', 'd_out')):
        continue
    from nanoquant.reconstruct import reconstruct_weight
    W_approx = reconstruct_weight(d['U_bin'], d['V_bin'], d['s1'], d['s2'], d['d_in'], d['d_out'])
    # We don't have the original W here, but we can check reconstruction norm
    print(f"{layer_name}: approx_norm={W_approx.norm().item():.3f}, shape={tuple(W_approx.shape)}")
EOF
```

Better: run a short diagnostic script that loads the FP16 model and the factors
simultaneously and computes relative error per layer. This needs to be written.

### Step 2: Compare binary ADMM vs plain SVD on representative layers

Test whether the binary approximation itself is the bottleneck or whether it's
an implementation issue.

Target: pick 3-5 representative weight matrices (1 attention, 2-3 expert w1/w2/w3,
1 large vs 1 small shape). For each:
- Compute rank-8 truncated SVD → Frobenius error
- Run lb_admm rank=8 → Frobenius error
- Check whether signed-scale fit significantly closes the gap

If SVD error >> binary ADMM error: the binary factorization quality is not the bottleneck;
something else (activation drift, reconstruction path) dominates.

If binary ADMM error >> SVD error: the ADMM is not converging well and needs fixes
(more iterations, better initialization, different rho/lam).

### Step 3: Ablation — quantize blocks using FP16 block inputs (offline)

This would isolate whether activation drift is the compounding mechanism.
Instead of sequentially piping quantized block outputs into next block ADMM,
use the FP16 intermediate activations for all blocks.

This requires caching the FP16 block inputs before any quantization, then
running ADMM block-by-block against these clean inputs.

If this produces much better quality: activation drift is the primary cause.
Fix: use error propagation mitigation (the paper describes this in Algorithm 1 —
it subtracts the quantization error at each step before passing to the next block).

### Step 4 (if Step 3 confirms drift): Implement error propagation mitigation

The NanoQuant paper (Section 3.2) describes error propagation mitigation as
part of Phase 2. The current pipeline does NOT implement this — each block is
passed the quantized output of the previous block rather than a corrected signal.

Reference: `scripts/run_stage1.py` passes `block_input = _run_block(block, bi)`
after quantization, rather than `block_input = fp16_block_output - quantization_error`.

## Key Paths

| Path | Contents |
|------|----------|
| `output/Phi-tiny-MoE-instruct_signedfit_full_20260312` | Current best factors (30,413x) |
| `output/Phi-tiny-MoE-instruct_signedfit_full_20260312_reconstructed` | Reconstructed HF model |
| `output/Phi-tiny-MoE-instruct_ablate_signedfit_mb2` | 2-block ablation (2.11x) |
| `output/Phi-tiny-MoE-instruct_rerun_20260311` | Stable but unsigned full run (219,420x) |
| `results/Phi-tiny-MoE-instruct/metrics.json` | Latest eval metrics (signed-fit full run) |
| `results/SUMMARY.md` | Aggregated table |

## Decision Tree

### If per-layer errors (Step 1) are uniformly catastrophic
→ The binary approximation itself is failing. Proceed to Step 2 (SVD comparison).
→ If binary ADMM ≫ SVD error: fix ADMM (increase iters, tune rho/lam, check update equations).
→ If binary ADMM ≈ SVD error: rank is too low; try rank=16 or rank=32 ablation.

### If per-layer errors vary — early blocks fine, later blocks catastrophic
→ Confirms activation drift / error compounding as the primary mechanism.
→ Proceed to Step 3 (offline clean-input ablation) to quantify.
→ If confirmed: implement error propagation mitigation (Step 4).

### If per-layer errors vary — specific layer types are bad
→ Expert layers (w1/w2/w3) have different shapes than attention layers.
→ May need separate rank tuning for expert vs attention layers.
→ Try `NANOQUANT_ATTN_RANK` env var to set a different rank for attention.

## What NOT to Do Next

- Do not re-run another full 32-block quantization without first understanding
  where the error is coming from. A full run takes ~2.5 hours and the evidence
  from Steps 1-3 will be much more informative per unit time.
- Do not increase `admm_iters` blindly — the issue may not be ADMM convergence.
- Do not move to Phase 3 (scaling to other models) until Phi-tiny-MoE PPL is
  at least within 10x of baseline. The pipeline has a fundamental quality issue.
