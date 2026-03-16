---
phase: 03-scaling-and-evaluation
plan: 02b
type: execute
wave: 2
depends_on:
  - 03-02
files_modified:
  - results/OLMoE-1B-7B-0924/metrics.json
  - results/SUMMARY.md
autonomous: false
status: complete
outcome: pipeline_validated_quality_insufficient
requirements:
  - SCALE-01

must_haves:
  truths:
    - "OLMoE-1B-7B-0924 quantizes all 16 blocks successfully without numerical instability"
    - "Quantized OLMoE-1B-7B-0924 produces a loadable checkpoint (loadable_checkpoint: true)"
    - "WikiText-2 PPL degradation ratio is <= 20x FP16 baseline (usable threshold)"
    - "All 16 block checkpoints exist in checkpoints/OLMoE-1B-7B-0924/"
  artifacts:
    - path: "results/OLMoE-1B-7B-0924/metrics.json"
      provides: "Full metrics for OLMoE-1B complete quantization"
      contains: "perplexity.quantized"
    - path: "output/OLMoE-1B-7B-0924/quantized_factors.safetensors"
      provides: "Complete 16-block quantized checkpoint"
---

<objective>
Fix the ADMM numerical instability that stops OLMoE-1B-7B-0924 quantization at block 11/16, then complete all 16 blocks and verify usable PPL.

The previous run (03-02 Task 1) stopped at block 11 with "Non-finite block output detected after block 11 — numerical instability in ADMM at depth". Blocks 0–10 are checkpointed at `checkpoints/OLMoE-1B-7B-0924/`. The resume logic in `quantize.py` will skip to block 11 automatically when a checkpoint_dir is provided.

Purpose: Prove the pipeline works end-to-end on a real MoE model (all blocks, usable PPL) before committing cloud GPU time to larger models. A usable result here is the gate-keeper for plan 03-03.

Success threshold: PPL degradation ratio <= 20x FP16 baseline (FP16 baseline ~6.62 → quantized PPL <= 132).
</objective>

<execution_context>
@C:/Users/james/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/james/.claude/get-shit-done/templates/summary.md
@C:/Users/james/.claude/get-shit-done/references/checkpoints.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/03-scaling-and-evaluation/03-02-SUMMARY.md
@nanoquant/quantize.py
@nanoquant/reconstruct.py
@nanoquant/admm.py
@scripts/run_stage1.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Diagnose block 11 instability</name>
  <files>checkpoints/OLMoE-1B-7B-0924/, nanoquant/quantize.py</files>
  <action>
Read and understand the block-output check at `nanoquant/quantize.py:283-287`. The issue is that after block 11 is quantized, running calibration data through the quantized block 11 produces non-finite activations. This is a forward-pass failure, not an ADMM convergence failure.

Inspect what happened with the previous run:
1. Count existing checkpoints: `ls checkpoints/OLMoE-1B-7B-0924/ | wc -l` — should be 11 (blocks 0–10).
2. Read `results/OLMoE-1B-7B-0924/metrics.json` `notes` field — confirms partial_quantization=true, blocks_quantized=11, failure_reason.

Determine the likely cause:
- OLMoE has 16 transformer blocks. At `--bits-per-weight 0.80`, rank was computed as ~819 (from metrics.json quantization_config.rank). For OLMoE's hidden_dim=2048, rank=819 is aggressive (>50% of d/2=1024).
- Deeper blocks (11+) tend to have more sensitive activations. High-rank ADMM can overfit calibration data and produce degenerate scales.

**Determine the fix to try first:**
- Option A (preferred): Reduce `--bits-per-weight` to 0.55 → lower rank → less aggressive quantization → better stability. This is the primary lever.
- Option B: Reduce `--rho` from 0.5 to 0.1 → softer ADMM penalty → more conservative updates.
- Option C: Both A and B combined.

Since we have blocks 0–10 checkpointed, we do NOT need to re-quantize them. The resume logic (`quantize.py:205-215`) will skip to block 11 automatically. However, lower bits-per-weight means lower rank → different ADMM behavior for block 11 onwards. This is acceptable — the first 11 blocks are already locked in from the previous run.

**Decision: Start with Option A (bits_per_weight=0.55) using resume from block 11.**

Document this decision in the task commit message.
  </action>
  <verify>
    <automated>ls checkpoints/OLMoE-1B-7B-0924/block_*.pt | wc -l</automated>
  </verify>
  <done>Understood root cause, identified fix parameters, confirmed 11 checkpoints present to resume from</done>
</task>

<task type="auto">
  <name>Task 2: Resume quantization with stability fix (blocks 11–15)</name>
  <files>output/OLMoE-1B-7B-0924/, checkpoints/OLMoE-1B-7B-0924/</files>
  <action>
Run Stage 1 quantization resuming from block 11 with reduced bits-per-weight for stability:

```bash
python scripts/run_stage1.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --output output/OLMoE-1B-7B-0924 \
  --bits-per-weight 0.55 \
  --admm-iters 50 \
  --n-calibration 128 \
  --seq-len 2048 \
  --checkpoint-dir checkpoints/OLMoE-1B-7B-0924
```

The `--checkpoint-dir` flag triggers automatic resume: `quantize.py:205-215` detects blocks 0–10 and starts from block 11. Only blocks 11–15 will be quantized in this run.

**If block 11 still fails with non-finite output**, escalate to Option B (add `--rho 0.1`):

```bash
python scripts/run_stage1.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --output output/OLMoE-1B-7B-0924 \
  --bits-per-weight 0.55 \
  --rho 0.1 \
  --admm-iters 50 \
  --n-calibration 128 \
  --seq-len 2048 \
  --checkpoint-dir checkpoints/OLMoE-1B-7B-0924
```

**If block 11 still fails with both options**, investigate `nanoquant/quantize.py` — the block output check could be made recoverable (clamp rather than raise) as a last resort. Document this as a finding before making code changes.

After successful completion, verify the output directory has the new safetensors with all 16 blocks:
```bash
python -c "
import safetensors.torch as st
t = st.load_file('output/OLMoE-1B-7B-0924/quantized_factors.safetensors')
blocks = set(k.split('.')[1] for k in t.keys() if k.startswith('model.layers.'))
print(f'Blocks in checkpoint: {sorted(blocks)}')
print(f'Count: {len(blocks)}')
"
```

Expected: 16 blocks (0–15).
  </action>
  <verify>
    <automated>python -c "
import safetensors.torch as st, os
assert os.path.exists('output/OLMoE-1B-7B-0924/quantized_factors.safetensors'), 'checkpoint missing'
t = st.load_file('output/OLMoE-1B-7B-0924/quantized_factors.safetensors')
blocks = set(int(k.split('.')[1]) for k in t.keys() if k.startswith('model.layers.'))
assert len(blocks) == 16, f'Expected 16 blocks, got {len(blocks)}: {sorted(blocks)}'
print('PASS: all 16 blocks present')
"</automated>
  </verify>
  <done>All 16 blocks quantized, quantized_factors.safetensors updated, all block checkpoints present</done>
</task>

<task type="auto">
  <name>Task 3: Run evaluation and update metrics</name>
  <files>results/OLMoE-1B-7B-0924/metrics.json, results/SUMMARY.md</files>
  <action>
Run evaluation on the complete 16-block checkpoint:

```bash
python scripts/run_eval.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --quantized-dir output/OLMoE-1B-7B-0924 \
  --results-dir results \
  --stride 512 \
  --max-length 2048
```

After evaluation, update `results/OLMoE-1B-7B-0924/metrics.json`:
- Set `notes.partial_quantization` to `false`
- Set `notes.blocks_quantized` to `16`
- Set `notes.blocks_quantized_pct` to `100.0`
- Remove `notes.failure_reason` (or set to null)
- Update `loadable_checkpoint` to `true` if PPL is reasonable
- Record the new `perplexity.quantized` value

Update `results/SUMMARY.md` to reflect the complete run (replace the partial row for OLMoE).

**Usable threshold check**: PPL degradation ratio must be <= 20x.
- FP16 baseline: ~6.62
- Threshold: 6.62 × 20 = 132.4
- If quantized PPL > 132: escalate to Task 4 (additional ADMM tuning needed)
- If quantized PPL <= 132: proceed to checkpoint
  </action>
  <verify>
    <automated>python -c "
import json
m = json.load(open('results/OLMoE-1B-7B-0924/metrics.json'))
assert m['notes']['blocks_quantized'] == 16, 'Not all blocks quantized'
assert m['notes']['blocks_quantized_pct'] == 100.0
assert m['perplexity']['quantized'] is not None, 'No quantized PPL'
ratio = m['perplexity']['degradation_ratio']
print(f'PPL degradation: {ratio:.1f}x (threshold: 20x)')
assert ratio <= 20.0, f'PPL degradation {ratio:.1f}x exceeds usable threshold of 20x'
print('PASS: usable PPL achieved')
"</automated>
  </verify>
  <done>metrics.json updated with complete 16-block eval, SUMMARY.md updated, PPL degradation <= 20x</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: Verify OLMoE complete quantization result</name>
  <action>Human verifies OLMoE-1B-7B-0924 is fully quantized with usable PPL.</action>
  <what-built>Complete 16-block OLMoE-1B-7B-0924 quantization with usable WikiText-2 PPL — validating the pipeline end-to-end before cloud GPU spend</what-built>
  <how-to-verify>
    1. Check `results/OLMoE-1B-7B-0924/metrics.json` — `notes.blocks_quantized` should be 16, `perplexity.degradation_ratio` should be <= 20
    2. Check `output/OLMoE-1B-7B-0924/quantized_factors.safetensors` — should contain all 16 blocks
    3. PPL degradation ratio: was 3237x (partial), target is <= 20x
    4. FP16 baseline: ~6.62 — quantized should be under ~132
  </how-to-verify>
  <resume-signal>Type "approved" to proceed to 03-02c (Qwen1.5-MoE approach), or describe issues</resume-signal>
</task>

</tasks>

<verification>
1. `output/OLMoE-1B-7B-0924/quantized_factors.safetensors` contains all 16 transformer blocks
2. `results/OLMoE-1B-7B-0924/metrics.json` has `notes.blocks_quantized == 16` and `notes.partial_quantization == false`
3. `perplexity.degradation_ratio` <= 20x
4. All 16 `checkpoints/OLMoE-1B-7B-0924/block_*.pt` files exist
</verification>

<success_criteria>
- OLMoE-1B-7B-0924 quantized all 16 blocks with no non-finite block outputs
- Quantized PPL degradation <= 20x FP16 baseline (~6.62 → <= 132)
- metrics.json reflects complete run (blocks_quantized: 16, partial_quantization: false)
- SUMMARY.md updated with complete OLMoE result
</success_criteria>

<output>
After completion, create `.planning/phases/03-scaling-and-evaluation/03-02b-SUMMARY.md`
</output>
