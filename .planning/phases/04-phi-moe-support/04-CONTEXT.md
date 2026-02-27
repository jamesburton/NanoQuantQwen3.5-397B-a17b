# Phase 4: Phi MoE Support - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Add PhimoeExperts support to the NanoQuant pipeline with fused gate_up_proj (2*intermediate) layout handling, and validate end-to-end quantization with Phi-tiny-MoE-instruct (3.8B) and Phi-mini-MoE-instruct (7.6B) on a 12GB consumer GPU. Includes reconstruction support for eval.

</domain>

<decisions>
## Implementation Decisions

### Fused weight layout
- Split gate_up_proj (shape: n_experts, 2*intermediate, hidden) into separate gate_proj and up_proj WeightViews per expert
- Each half gets its own ADMM quantization pass (doubles view count but more precise)
- View naming: `experts.gate_proj[i]` and `experts.up_proj[i]` (matches HF dense model naming convention)
- Shared Hessian: both gate_proj and up_proj views use the same Hessian from the fused module input (consistent with existing expert Hessian sharing)
- Implemented as a special case branch inside the existing `get_weight_views` function

### Model targets
- Primary: `microsoft/Phi-tiny-MoE-instruct` (3.8B total, 1.1B active) — must pass
- Stretch goal: `microsoft/Phi-mini-MoE-instruct` (7.6B total, 2.4B active)
- Exact expert class name (PhimoeExperts or similar) to be confirmed by researcher from HF transformers source
- If Phi MoE uses separate nn.Linear experts instead of fused 3D tensors, adapt approach accordingly (don't force fused pattern)

### Validation approach
- Two-stage: smoke test first (runs without crash, produces loadable checkpoint), then full WikiText-2 eval
- Full eval: FP16 baseline PPL + quantized PPL + memory footprint reporting
- No hard perplexity degradation threshold — report numbers and let them speak
- Results stored in `results/Phi-tiny-MoE-instruct/metrics.json` (same structure as OLMoE results)

### Integration with existing code
- Light refactor: extract common fused-expert pattern for both existing and Phi architectures
- Change `FUSED_EXPERT_CLASSES` from `frozenset` to `dict` mapping class name to layout metadata (e.g., `needs_split=True`, `split_dim=0`)
- Existing OLMoE/Qwen entries get `needs_split=False` to preserve current behavior
- Auto-detect Phi MoE from HF `config.json` `model_type` field (consistent with Qwen/OLMoE detection)
- Update `reconstruct_model.py` to handle split gate_proj/up_proj reassembly back to fused gate_up_proj

### Claude's Discretion
- Phi-mini-MoE eval depth (full or lighter pass based on time/resources)
- Exact refactoring approach for the FUSED_EXPERT_CLASSES dict migration
- Error handling and edge cases during split-view creation

</decisions>

<specifics>
## Specific Ideas

- gate_proj / up_proj naming should match HF dense model conventions for checkpoint key intuitiveness
- The dict-based FUSED_EXPERT_CLASSES should be extensible for future MoE architectures without code changes (just add a new entry)
- Smoke test before full eval — fast iteration loop for debugging weight layout issues

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-phi-moe-support*
*Context gathered: 2026-02-27*
