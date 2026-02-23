# Roadmap: NanoQuant Expansion

## Overview

Build a consumer-friendly Stage 1 NanoQuant pipeline for MoE models, starting from the existing nanoquant/ codebase. Phase 1 completes the core pipeline end-to-end. Phase 2 adds MoE architecture support for OLMoE and Qwen families. Phase 3 validates progressive scaling from 1B to 397B parameters and produces perplexity baselines.

## Phases

- [ ] **Phase 1: Pipeline Foundation** - Stage 1 runs end-to-end within 12GB VRAM on a single model
- [ ] **Phase 2: MoE Support** - WeightView abstractions handle OLMoE and Qwen expert tensor layouts
- [ ] **Phase 3: Scaling and Evaluation** - Progressive validation across four model sizes with PPL baselines

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
**Plans**: TBD

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
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pipeline Foundation | 2/3 | Gap closure | - |
| 2. MoE Support | 0/TBD | Not started | - |
| 3. Scaling and Evaluation | 0/TBD | Not started | - |
