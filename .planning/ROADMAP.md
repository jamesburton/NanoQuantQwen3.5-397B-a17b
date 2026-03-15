# Roadmap: NanoQuant Expansion

## Overview

Build a consumer-friendly Stage 1 NanoQuant pipeline for MoE models, starting from the existing nanoquant/ codebase. Phase 1 completes the core pipeline end-to-end. Phase 2 adds MoE architecture support for OLMoE and Qwen families. Phase 3 validates progressive scaling from 1B to 397B parameters and produces perplexity baselines.

## Phases

- [x] **Phase 1: Pipeline Foundation** - Stage 1 runs end-to-end within 12GB VRAM on a single model (completed 2026-02-23)
- [x] **Phase 2: MoE Support** - WeightView abstractions handle OLMoE and Qwen expert tensor layouts (completed 2026-02-24)
- [ ] **Phase 3: Scaling and Evaluation** - Progressive validation across four model sizes with PPL baselines
- [x] **Phase 4: Phi MoE Support** - Add PhimoeExperts fused gate_up layout + validate with Phi-tiny-MoE-instruct (completed 2026-03-08)

## Phase Details

### Phase 1: Pipeline Foundation
**Goal**: Stage 1 quantization (Hessian capture → ADMM init → block-wise reconstruction) runs end-to-end on a single model within 12GB VRAM
**Depends on**: Nothing (first phase)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05
**Success Criteria** (what must be TRUE):
  1. Running `python scripts/run_stage1.py` on a supported model completes without OOM on a 12GB GPU via CPU/NVMe offloading
  2. Output directory contains a serializable checkpoint with binary matrices and scales that can be loaded back
  3. Sub-1-bit quantization (0.7–0.9 bits) is the default mode and produces a quantized model
  4. Console shows real-time progress with ETA and current VRAM usage during quantization
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Hardware auto-detection module + safetensors checkpoint module
- [x] 01-02-PLAN.md — Pipeline integration (hardware probe, binary factor collection, progress logging, sub-1-bit config)
- [ ] 01-03-PLAN.md — Gap closure: fix hardware summary display + dstorage-gpu detection

### Phase 2: MoE Support
**Goal**: The pipeline correctly handles OLMoE and Qwen MoE expert tensor layouts, treating all layer types uniformly through WeightView
**Depends on**: Phase 1
**Requirements**: MOE-01, MOE-02, MOE-03, MOE-04
**Success Criteria** (what must be TRUE):
  1. WeightView correctly maps OLMoE-1B-7B expert tensors so ADMM and reconstruction see uniform weight shapes
  2. WeightView correctly maps Qwen1.5-MoE, Qwen2-MoE, and Qwen3.5-MoE expert tensor layouts
  3. Shared attention and embedding layers are included in the quantized checkpoint alongside expert layers
  4. All experts within the same transformer block are processed together in a single reconstruction group
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md — FUSED_EXPERT_CLASSES dispatch table for OLMoE + Qwen variant WeightView support
- [ ] 02-02-PLAN.md — Self-contained checkpoint (shared layers as FP16) + architecture auto-detection

### Phase 3: Scaling and Evaluation
**Goal**: Pipeline is verified on four model sizes (OLMoE-1B → Qwen1.5-3B → Qwen2-57B → Qwen3.5-397B) and perplexity baselines are documented for each
**Depends on**: Phase 2
**Requirements**: SCALE-01, SCALE-02, SCALE-03, SCALE-04, EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. OLMoE-1B-7B-0924 quantizes and produces a loadable checkpoint on a 12GB consumer GPU
  2. Qwen1.5-MoE-3B and Qwen2-MoE-57B each complete Stage 1 quantization without errors
  3. Qwen3.5-397B-A17B completes Stage 1 quantization (cloud GPU acceptable)
  4. WikiText-2 perplexity is measured and reported for both FP16 baseline and quantized version of each model
  5. Memory footprint report shows model size before/after quantization and peak VRAM for each run
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — Fix eval PPL computation + metrics/reporting infrastructure
- [ ] 03-02-PLAN.md — Local GPU runs: OLMoE-1B-7B + Qwen1.5-MoE-A2.7B
- [ ] 03-03-PLAN.md — Cloud GPU runs: Qwen2-57B + Qwen3.5-397B

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pipeline Foundation | 3/3 | Complete   | 2026-02-23 |
| 2. MoE Support | 2/2 | Complete   | 2026-02-24 |
| 3. Scaling and Evaluation | 2/3 | In Progress|  |
| 4. Phi MoE Support | 3/3 | Complete   | 2026-03-08 |

### Phase 4: Phi MoE Support
**Goal**: Add PhimoeExperts support (fused gate_up_proj 2x layout) and validate pipeline end-to-end with Phi-tiny-MoE-instruct (3.8B total, 1.1B active) on local GPU
**Depends on**: Phase 2
**Requirements**: PHI-01, PHI-02, PHI-03
**Success Criteria** (what must be TRUE):
  1. PhimoeExperts added to FUSED_EXPERT_CLASSES and weight views handle the fused gate_up_proj (2*intermediate) layout
  2. Phi-tiny-MoE-instruct quantizes end-to-end on a 12GB consumer GPU producing a valid checkpoint
  3. WikiText-2 perplexity measured for both FP16 baseline and quantized Phi-tiny-MoE
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md — FUSED_EXPERT_CLASSES dict migration + PhimoeExperts fused gate_up_proj split support
- [x] 04-02-PLAN.md — Phi-tiny-MoE-instruct end-to-end quantization + WikiText-2 eval
- [x] 04-03-PLAN.md — Gap closure: fix factor key collision + run WikiText-2 eval for Phi-tiny-MoE-instruct (PHI-03)
