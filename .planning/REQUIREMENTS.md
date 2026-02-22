# Requirements: NanoQuant Expansion

**Defined:** 2026-02-22
**Core Value:** Consumer-friendly sub-1-bit quantization of MoE models via NanoQuant's ADMM-based pipeline

## v1 Requirements

Requirements for initial release. Focused on Stage 1 pipeline with progressive model scaling.

### Pipeline Core

- [x] **PIPE-01**: Stage 1 pipeline runs end-to-end (Hessian capture → ADMM init → block-wise reconstruction) on a single model
- [x] **PIPE-02**: Pipeline operates within 12GB VRAM via CPU/NVMe offloading through Accelerate
- [x] **PIPE-03**: Pipeline supports sub-1-bit quantization (0.7–0.9 bits) as primary mode per NanoQuant paper
- [x] **PIPE-04**: Pipeline produces serializable quantized checkpoint (binary matrices + scales)
- [x] **PIPE-05**: Pipeline provides progress logging with ETA and memory usage tracking

### MoE Support

- [ ] **MOE-01**: WeightView abstraction handles OLMoE-1B-7B expert tensor layout
- [ ] **MOE-02**: WeightView abstraction handles Qwen MoE expert tensor layout (Qwen1.5-MoE, Qwen2-MoE, Qwen3.5-MoE)
- [ ] **MOE-03**: Shared attention/embedding layers quantized alongside expert layers
- [ ] **MOE-04**: Expert-aware block grouping for reconstruction (experts in same block processed together)

### Model Scaling

- [ ] **SCALE-01**: Verified working on OLMoE-1B-7B-0924 (smallest, consumer GPU validation)
- [ ] **SCALE-02**: Verified working on Qwen1.5-MoE-3B-14B (small-medium scaling check)
- [ ] **SCALE-03**: Verified working on Qwen2-MoE-57B (medium-large scaling)
- [ ] **SCALE-04**: Verified working on Qwen3.5-397B-A17B (full target, likely requires cloud GPU)

### Evaluation

- [ ] **EVAL-01**: Perplexity evaluation on WikiText-2 for quantized models
- [ ] **EVAL-02**: Comparison baseline: FP16 perplexity for each model
- [ ] **EVAL-03**: Memory footprint reporting (model size before/after, peak VRAM usage)

## v2 Requirements

### Multi-Stage Pipeline

- **STAGE-01**: Stage 2 STE refinement pass after Stage 1
- **STAGE-02**: Stage 3 Knowledge Distillation with teacher model
- **STAGE-03**: Configurable bit-width toggle (1-bit binary vs sub-1-bit)

### Inference Integration

- **INFER-01**: HF Transformers-compatible model loading from quantized checkpoint
- **INFER-02**: vLLM integration for serving quantized models
- **INFER-03**: Integration guide documentation

### Advanced

- **ADV-01**: Mixed bit-width quantization (different rates per layer/expert)
- **ADV-02**: Automated ADMM hyperparameter tuning

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training/fine-tuning | Focus is post-training quantization only |
| Non-MoE dense models | Existing NanoQuant paper already covers these |
| Custom CUDA kernels | Use PyTorch native ops; kernel optimization is v2+ |
| Distributed multi-GPU quantization | Consumer single-GPU focus for v1 |
| Web UI / dashboard | CLI scripts sufficient for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | Phase 1 | Complete |
| PIPE-02 | Phase 1 | Complete |
| PIPE-03 | Phase 1 | Complete |
| PIPE-04 | Phase 1 | Complete |
| PIPE-05 | Phase 1 | Complete |
| MOE-01 | Phase 2 | Pending |
| MOE-02 | Phase 2 | Pending |
| MOE-03 | Phase 2 | Pending |
| MOE-04 | Phase 2 | Pending |
| SCALE-01 | Phase 3 | Pending |
| SCALE-02 | Phase 3 | Pending |
| SCALE-03 | Phase 3 | Pending |
| SCALE-04 | Phase 3 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-22*
*Last updated: 2026-02-22 — traceability populated after roadmap creation*
