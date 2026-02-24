# Phase 3: Scaling and Evaluation - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate the quantization pipeline across four model sizes (OLMoE-1B-7B → Qwen1.5-MoE-3B → Qwen2-MoE-57B → Qwen3.5-397B-A17B) and produce perplexity baselines + memory/timing reports for each. No new pipeline features — this phase exercises what Phases 1-2 built.

</domain>

<decisions>
## Implementation Decisions

### Scaling Strategy
- Small models (OLMoE-1B, Qwen1.5-3B) run locally on 12GB consumer GPU
- Large models (Qwen2-57B, Qwen3.5-397B) run on rented cloud GPU via RunPod or vast.ai
- Scripts must be hardware-agnostic — same code runs locally and on cloud

### Perplexity Evaluation
- Run our own FP16 baseline perplexity on WikiText-2 for every model (no published numbers)
- Identical evaluation settings for FP16 and quantized versions to ensure apples-to-apples comparison

### Memory & Timing Reporting
- Track three memory metrics: peak VRAM, peak system RAM, model size before/after quantization
- Track wall-clock time: total quantization time + per-layer timing (useful for estimating cloud GPU cost)
- Live VRAM tracking during quantization runs + summary table at the end

### Output Artifacts
- `results/{model_name}/` — JSON metrics per model + human-readable notes
- `results/SUMMARY.md` — aggregated comparison table across all models
- `checkpoints/{model_name}/` — quantized checkpoint files (separate from reports)
- Eval script auto-generates/updates SUMMARY.md after each model evaluation
- JSON format for machine-readable results, Markdown for human consumption

### Claude's Discretion
- Execution order across models (smallest-first recommended but flexible)
- Failure handling strategy (continue vs stop on error)
- WikiText-2 context length and stride for perplexity measurement
- Whether eval is a separate script or integrated (separate script preferred)
- Overall PPL vs per-layer breakdown (overall PPL sufficient for Phase 3 baseline)

</decisions>

<specifics>
## Specific Ideas

- Cloud GPU provider: RunPod or vast.ai for A100/H100 rentals
- Scripts should print clear instructions if run on insufficient hardware rather than silently failing
- SUMMARY.md should have a single comparison table that's easy to paste into a paper or README

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-scaling-and-evaluation*
*Context gathered: 2026-02-24*
