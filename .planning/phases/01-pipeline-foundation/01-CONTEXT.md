# Phase 1: Pipeline Foundation - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Stage 1 quantization pipeline (Hessian capture → ADMM initialization → block-wise reconstruction) runs end-to-end on a single model within 12GB VRAM. Produces a serializable quantized checkpoint with sub-1-bit compression. MoE-specific weight handling and multi-model scaling are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Offloading strategy
- Auto-detect available resources (GPU VRAM, CPU RAM, NVMe) at startup and adapt offloading strategy automatically
- No explicit hardware floor — pipeline handles whatever is available gracefully
- No manual device map configuration required from user

### Claude's Discretion
- Weight management approach (block-by-block vs layer streaming vs hybrid) — choose based on memory constraints and implementation simplicity
- Hessian storage strategy (CPU RAM vs disk-backed memory-mapping) — choose based on model size and detected resources
- Checkpoint serialization format — pick what works best for downstream loading
- Pipeline orchestration details (config file vs CLI args, resume behavior)
- Progress/logging implementation (ETA, memory display, log format)

</decisions>

<specifics>
## Specific Ideas

- Pipeline should "just work" on consumer hardware without the user needing to understand offloading internals
- Hardware probing at startup to determine strategy — user shouldn't need to configure memory limits manually

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-pipeline-foundation*
*Context gathered: 2026-02-22*
