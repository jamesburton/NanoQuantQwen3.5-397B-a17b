# Phase 2: MoE Support - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

The pipeline correctly handles OLMoE and Qwen MoE expert tensor layouts, treating all layer types uniformly through WeightView. This phase extends the existing Qwen2MoeExperts support to cover OLMoE (individual nn.Linear per expert) and additional Qwen variants (Qwen1.5-MoE, Qwen3.5-MoE). Shared/non-expert layers are included in quantized checkpoints alongside expert layers.

</domain>

<decisions>
## Implementation Decisions

### Architecture Detection
- Auto-detect MoE architecture from model config.json (model_type/architectures field)
- Support both HuggingFace model IDs and local paths (auto-resolve config either way)
- Claude's discretion on fallback behavior when architecture is unknown
- Claude's discretion on whether Qwen variants are one family with internal branching or separate backends (decide based on actual tensor layout similarity)

### Shared Layer Quantization
- LayerNorm/RMSNorm layers must be included in the quantized checkpoint (stored as FP16) for self-contained checkpoints
- Claude's discretion on whether shared layers (attention, embeddings) get same bit-width as experts or higher precision
- Claude's discretion on embedding/lm_head treatment (FP16 vs quantized)
- Claude's discretion on shared expert (e.g., Qwen's shared_expert MLP) treatment

### Expert Reconstruction Grouping
- Shared Hessian across experts in same module (current approach) — confirmed, no per-expert Hessian
- Allow parallel block processing (not just sequential)
- Auto-detect concurrency from available VRAM by default, with --parallel-blocks N flag for user override
- Claude's discretion on whether all experts in a block are one reconstruction group or can be split for VRAM

### OLMoE Tensor Layout
- OLMoE-1B-7B is the primary development/test target — develop and validate against it first, then extend to Qwen variants
- OLMoE experts share Hessian across all experts in a layer (consistent with Qwen fused approach)
- Claude's discretion on WeightView naming convention and structural approach for OLMoE's individual nn.Linear experts
- Claude's discretion on whether OLMoE needs deeper WeightView changes or just correct identification and grouping

### Claude's Discretion
- Unknown architecture fallback behavior (error vs generic fallback)
- Qwen variant unification strategy (one family vs separate backends)
- Shared layer bit-width decisions (same as experts vs higher precision)
- Embedding/lm_head quantization policy
- Shared expert classification (expert vs shared layer)
- Expert reconstruction group sizing for VRAM-constrained scenarios
- OLMoE WeightView naming convention
- OLMoE structural approach depth

</decisions>

<specifics>
## Specific Ideas

- OLMoE-1B-7B (64 experts, top-8 routing) as primary dev target — fits on consumer GPU
- Existing codebase already handles Qwen2MoeExperts fused 3D tensors in moe.py
- Checkpoint must be self-contained (include norms in FP16, not just quantized weights)
- Parallel block processing with auto VRAM detection is a new capability beyond current sequential flow

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-moe-support*
*Context gathered: 2026-02-24*
