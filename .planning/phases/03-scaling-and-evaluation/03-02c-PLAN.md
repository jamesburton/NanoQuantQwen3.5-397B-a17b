---
phase: 03-scaling-and-evaluation
plan: 02c
type: execute
wave: 3
depends_on:
  - 03-02b
files_modified:
  - results/Qwen1.5-MoE-A2.7B/metrics.json
  - results/SUMMARY.md
autonomous: false
requirements:
  - SCALE-02

must_haves:
  truths:
    - "A clear decision is made on how to handle Qwen1.5-MoE-A2.7B evaluation given the 12GB VRAM constraint"
    - "The chosen approach is executed and results documented in metrics.json"
    - "results/SUMMARY.md reflects the final Qwen1.5 outcome"
  artifacts:
    - path: "results/Qwen1.5-MoE-A2.7B/metrics.json"
      provides: "Final metrics for Qwen1.5-MoE-A2.7B — either measured PPL or documented scope decision"
      contains: "perplexity"
---

<objective>
Decide and execute the approach for Qwen1.5-MoE-A2.7B evaluation, now that OLMoE-1B-7B has proven the pipeline works end-to-end (03-02b gate passed).

Context from 03-02:
- Qwen1.5-MoE-A2.7B is 26.67 GB FP16 — too large to eval on 12GB GPU via CPU offload (~48h at 298s/window)
- 5/24 blocks currently quantized (20.83%), factors file exists at output/Qwen1.5-MoE-A2.7B/
- Compression to 11.28 GB factors (2.36x) was achieved for the 5 completed blocks

The purpose of this model in Phase 3 was to validate the pipeline scales from 1B → 3B before the cloud runs. OLMoE (1B) is now proven. This plan decides whether Qwen1.5 (3B) is also needed before cloud spend, or whether OLMoE is sufficient validation.
</objective>

<execution_context>
@C:/Users/james/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/james/.claude/get-shit-done/templates/summary.md
@C:/Users/james/.claude/get-shit-done/references/checkpoints.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/03-scaling-and-evaluation/03-02b-SUMMARY.md
@results/Qwen1.5-MoE-A2.7B/metrics.json
@scripts/run_stage1.py
</context>

<tasks>

<task type="checkpoint:decision" gate="blocking">
  <name>Task 1: Decide Qwen1.5-MoE-A2.7B approach</name>
  <action>Present the user with the available options for handling Qwen1.5-MoE-A2.7B.</action>
  <what-built>Decision point: how to handle 26.67 GB model that cannot be evaluated on local 12GB GPU</what-built>
  <how-to-verify>
Present these options:

**Option A — Complete full quantization + defer eval to cloud**
- Finish quantizing all 24 blocks (resume from block 5)
- Leave PPL as null/deferred, note eval is cloud-only
- Pro: checkpoint is complete for cloud eval in 03-03
- Con: no local PPL validation for this model size

**Option B — Skip and rely on OLMoE as the local validation**
- Accept that OLMoE-1B-7B (now proven) is sufficient local validation
- Mark Qwen1.5-MoE-A2.7B as "out of scope for local eval" in metrics.json
- Pro: saves time, OLMoE result is the gate that matters
- Con: gap between 1B and 57B cloud models in the scaling curve

**Option C — Complete quantization + eval on cloud in 03-03**
- Same as A but explicitly include Qwen1.5 eval in the 03-03 cloud plan
- Pro: complete scaling curve (1B, 3B, 57B, 397B)
- Con: adds cloud time for a relatively small model

Which option do you want to proceed with?
  </how-to-verify>
  <resume-signal>Reply with "A", "B", or "C" and any notes</resume-signal>
</task>

<task type="auto">
  <name>Task 2: Execute chosen approach</name>
  <files>results/Qwen1.5-MoE-A2.7B/metrics.json, results/SUMMARY.md</files>
  <action>
Execute based on user decision from Task 1:

**If Option A or C (complete quantization):**

Resume Qwen1.5-MoE-A2.7B quantization from block 5:

```bash
python scripts/run_stage1.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --output output/Qwen1.5-MoE-A2.7B \
  --bits-per-weight 0.55 \
  --admm-iters 50 \
  --n-calibration 128 \
  --seq-len 2048 \
  --checkpoint-dir checkpoints/Qwen1.5-MoE-A2.7B
```

Use `--bits-per-weight 0.55` (same stability fix as 03-02b). Resume from block 5 automatically via checkpoint_dir.

After completion, update `results/Qwen1.5-MoE-A2.7B/metrics.json`:
- Set `notes.blocks_quantized` to 24, `notes.partial_quantization` to false
- Keep `perplexity.quantized` as null with `eval_skip_reason` updated to indicate deferred to 03-03 cloud run
- If Option C: add a note that eval is scheduled for 03-03

**If Option B (skip):**

Update `results/Qwen1.5-MoE-A2.7B/metrics.json`:
- Set `notes.scope_decision` to "Local eval skipped — OLMoE-1B-7B is the local validation anchor; Qwen1.5-MoE deferred to cloud 03-03 or out of scope"
- Keep existing partial quantization data

In all cases, update `results/SUMMARY.md` to reflect the decision.
  </action>
  <verify>
    <automated>python -c "
import json
m = json.load(open('results/Qwen1.5-MoE-A2.7B/metrics.json'))
assert 'notes' in m
print(f'Qwen1.5 notes: {json.dumps(m[\"notes\"], indent=2)}')
print('PASS')
"</automated>
  </verify>
  <done>Qwen1.5-MoE-A2.7B metrics.json updated with final decision, SUMMARY.md reflects outcome</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Verify Qwen1.5 outcome and confirm 03-03 readiness</name>
  <action>Human confirms Qwen1.5-MoE outcome is acceptable and cloud phase 03-03 can proceed.</action>
  <what-built>Qwen1.5-MoE-A2.7B handled per chosen approach, SUMMARY.md updated</what-built>
  <how-to-verify>
    1. Check `results/Qwen1.5-MoE-A2.7B/metrics.json` reflects the chosen approach
    2. Check `results/SUMMARY.md` has updated Qwen1.5 row
    3. Confirm you're ready to proceed to 03-03 (cloud GPU: Qwen2-57B + Qwen3.5-397B)
  </how-to-verify>
  <resume-signal>Type "approved" to proceed to 03-03 cloud run, or describe issues</resume-signal>
</task>

</tasks>

<verification>
1. `results/Qwen1.5-MoE-A2.7B/metrics.json` updated with final decision/outcome
2. `results/SUMMARY.md` reflects Qwen1.5 final state
3. Human-approved checkpoint passed before 03-03
</verification>

<success_criteria>
- Clear decision made and documented for Qwen1.5-MoE-A2.7B
- metrics.json updated to reflect decision (no ambiguity about eval status)
- SUMMARY.md updated
- Gate cleared for 03-03 cloud run
</success_criteria>

<output>
After completion, create `.planning/phases/03-scaling-and-evaluation/03-02c-SUMMARY.md`
</output>
