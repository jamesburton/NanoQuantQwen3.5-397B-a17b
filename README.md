# NanoQuant for Qwen3.5-397B-A17B

Research attempt at sub-bit quantization of the Qwen3.5-397B-A17B model using the NanoQuant method.

## References

- **NanoQuant paper**: [Efficient Sub-1-Bit Quantization of Large Language Models (arXiv:2602.06694v1)](https://arxiv.org/html/2602.06694v1)
- **Local Abstract Summary**: [.planning/research/NANOQUANT_PAPER_ABSTRACT.md](.planning/research/NANOQUANT_PAPER_ABSTRACT.md)
- **HuggingFace model page**: [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)

---

## Source Material Summary

### NanoQuant Method

NanoQuant formulates weight compression as a **low-rank binary factorization problem**. Each weight matrix W is decomposed as:

```
W ≈ s1 ⊙ (U±1 · V±1ᵀ) ⊙ s2ᵀ
```

where U, V are binary matrices (±1) and s1, s2 are floating-point scale vectors. This eliminates the per-element metadata overhead that limits classical 1-bit methods to ~2–3 effective bits, achieving genuine **0.55–0.8 bits per weight**.

**Three-phase process:**

| Phase | Description |
|-------|-------------|
| 1. Global Calibration | Compute Hessian-aware diagonal preconditioners (D̃in, D̃out) from 128 WikiText-2 calibration samples (0.26M tokens) with shrinkage regularization |
| 2. Block Reconstruction | For each transformer block: (a) error propagation mitigation, (b) LB-ADMM binary factorization with SVID rank-1 init, (c) STE-based factorized component refinement |
| 3. Model Reconstruction | Global KL-divergence minimization between quantized and full-precision logits to tune all scale vectors |

**Key hyperparameters:**
- Shrinkage coefficient γ: `0.2` for Qwen family
- ADMM penalty ρ, ridge regularization λ (tunable)
- Cholesky decomposition reduces ADMM cost to O(r³/3)

**Published benchmarks (70B model):**
- Compression time: ~13 hours on a single H100 80GB
- 138 GB → 5.35 GB (10× reduction)
- 4.02× inference throughput on RTX 3050
- Accuracy competitive with 2-bit GPTQ baselines

---

### Qwen3.5-397B-A17B Architecture

| Property | Value |
|----------|-------|
| Total parameters | 397B (403B in safetensors) |
| Active parameters per token | ~17B |
| Architecture | Hybrid: Gated DeltaNet + sparse MoE |
| Layers | 60 |
| Hidden dimension | 4,096 |
| Layer pattern | 15 × (3× Gated DeltaNet→MoE + 1× Gated Attention→MoE) |
| Total experts | 512 per MoE layer |
| Active experts | 11 per token (10 routed + 1 shared) |
| Expert intermediate dim | 1,024 |
| Attention heads (Q) | 32; KV: 2 |
| Linear attention heads | 64 (V), 16 (QK) |
| Head dimension | 128 (DeltaNet), 256 (Attention) |
| Vocab size | 248,320 (padded) |
| Context length | 262,144 native; 1,010,000 with YaRN |
| Storage (BF16) | ~794 GB |
| Dtype | BF16 + F32 |

---

## Model Progression Ladder

Rather than targeting the full 397B model immediately, we use a three-stage ladder to validate NanoQuant compatibility with progressively larger and more complex Qwen MoE architectures — stopping early if a stage fails.

| Stage | Model | Total | Active | Experts | Active Exp | Layers | Hidden | BF16 Size | Architecture class |
|-------|-------|-------|--------|---------|------------|--------|--------|-----------|-------------------|
| **1 (local)** | Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 4 | 24 | 2,048 | ~29 GB | `qwen2_moe` — standard MoE |
| **2 (cloud small)** | Qwen3-235B-A22B | 235B | 22B | 128 | 8 | 94 | unknown | ~470 GB | standard transformer MoE + GQA |
| **3 (cloud large)** | Qwen3.5-397B-A17B | 397B | 17B | 512 | 11 | 60 | 4,096 | ~794 GB | hybrid DeltaNet + MoE |

**Gate logic:** Each stage must pass a perplexity gate and code compatibility check before proceeding to the next. Stage 1 is entirely free (local hardware). Stage 2 commits modest cloud spend. Stage 3 is the full goal.

### Why this order matters

- **Stage 1** validates that NanoQuant's block reconstruction pipeline handles MoE expert weight matrices (60 experts per layer, shared expert, `decoder_sparse_step=1`). Architecture is a known `qwen2_moe` type — transformers supports it well. Runs on a single consumer GPU or Mac (≥32GB).
- **Stage 2** validates scale: 128 experts, 94 layers (vs 24 in Stage 1), GQA. Architecture is still standard transformer. Identifies whether block-sequential processing time and memory scale as predicted before committing to Stage 3.
- **Stage 3** adds the novel DeltaNet hybrid: entirely different weight shapes for linear-attention layers (QK heads: 16, V heads: 64, head dim: 128) interleaved with MoE blocks. Highest risk of requiring code changes.

---

## System Requirements Estimate

### Weight volume

| Format | Size |
|--------|------|
| BF16 source weights | ~794 GB |
| Post-NanoQuant (~0.7 bits/weight) | ~35–50 GB |
| Scale vectors (FP16, ~2% overhead) | ~8 GB |
| Working / calibration buffers | ~80–120 GB GPU RAM |

### Scaling from published benchmarks

The paper reports 13 hours for a 70B dense model on one H100 80GB. Qwen3.5-397B-A17B has:
- ~5.7× more total weights
- MoE structure: expert weights dominate (~90% of parameters); these are processed block-by-block like dense layers
- DeltaNet layers add novel linear-attention weight matrices not present in Llama benchmarks

| Estimate | Value |
|----------|-------|
| Naive parameter-ratio scaling | ~74 hours (single H100) |
| Optimistic (MoE sparsity + parallelism) | ~20–40 hours (4× H100) |
| Recommended allocation | 2–4× H100 80GB nodes, 48–80 hours budget |

### CPU / RAM requirements

- Full model must be streamed from disk or CPU RAM layer by layer
- Recommend **≥512 GB CPU RAM** (to hold full BF16 model in memory for fast block iteration)
- Minimum viable: 256 GB CPU RAM + fast NVMe (block loading from disk)
- Storage: **≥1.5 TB NVMe** (source weights + working checkpoints)

### Calibration data

128 samples from WikiText-2 per paper. Fits trivially in memory. Hessian computation is the bottleneck, not data volume.

---

## Implementation Plan

### Stage 1: Local Feasibility — Qwen1.5-MoE-A2.7B

**Hardware:** Single GPU with ≥12 GB VRAM (RTX 3060/4070 or similar) — requires aggressive CPU offloading for the 14.3B (29GB) model.
**Cost:** $0 (local)
**Time budget:** 1–4 hours total

- [ ] Obtain NanoQuant codebase (or implement Algorithm 1 from paper)
- [ ] Download `Qwen/Qwen1.5-MoE-A2.7B` (~29 GB BF16)
- [ ] Prepare 128-sample WikiText-2 calibration set
- [ ] Run Phase 1 (global calibration): Hessian capture with γ=0.2
  - **Note:** Must use `device_map="auto"` or manual layer-wise offloading to fit in 12GB.
  - **Gate:** All 24 layers produce well-conditioned D̃in, D̃out — no NaN/inf
- [ ] Run Phase 2 (block reconstruction) on first 3 blocks
  - **Gate:** Code correctly iterates over 60 expert weight matrices per MoE block without shape errors
  - **Gate:** Shared expert (`shared_expert_intermediate_size=5632`) handled separately from routed experts
- [ ] Run full CUDA rerun for Phi-tiny-MoE-instruct with optimized reconstruction path; connected SlimMoE Phase 3 KD probes (`factor_scales`, `factor_latents`) are implemented but currently fail the FP-cache runtime budget on the local RTX 3060
- [ ] **Pass gate:** WikiText-2 perplexity ≤ 2× BF16 baseline (rough acceptable threshold for 0.7 bit/weight)
- [ ] Record: time per block, peak GPU/CPU RAM, any code changes required (especially offloading logic)

**Abort conditions:** Shape errors in ADMM, perplexity >5× baseline after full run, CUDA OOM that can't be resolved with offloading.

---

### Stage 2: Intermediate Scale — Qwen3-235B-A22B *(conditional on Stage 1 pass)*

**Hardware:** 4× H100 80 GB (cloud)
**Cost estimate:** ~$150–350 (feasibility run: $20–40)
**Time budget:** 20–50 hours

- [ ] Adapt any MoE-handling code changes identified in Stage 1 for `qwen3_moe` model class
- [ ] Verify GQA (64Q / 4KV heads) doesn't require changes to attention Hessian capture
- [ ] Download `Qwen/Qwen3-235B-A22B` (~470 GB BF16) to cloud storage
- [ ] Run calibration + first 5 blocks as a timed feasibility probe (~$20–40 spend)
  - **Gate:** Time per block scales ≤ linearly with parameter count vs Stage 1
  - **Gate:** 4× H100 GPU RAM sufficient with CPU offload (peak ≤ 320 GB GPU total)
- [ ] Run full 94-block reconstruction + scale refinement
- [ ] **Pass gate:** WikiText-2 perplexity ≤ 2× BF16 baseline; MMLU within 5 points of baseline
- [ ] Record: total wall time, cost, blocks/hour throughput

**Abort conditions:** Time per block is >2× the Stage 1 scaling prediction (indicates O(n²) issue), memory overflows that can't be resolved, perplexity diverges.

---

### Stage 3: Full Target — Qwen3.5-397B-A17B *(conditional on Stage 2 pass)*

**Hardware:** 4–8× H100 80 GB (cloud)
**Cost estimate:** ~$300–800
**Time budget:** 40–80 hours

- [ ] Audit DeltaNet linear-attention weight shapes vs NanoQuant assumptions
  - DeltaNet QK heads: 16, V heads: 64, head dim: 128 — asymmetric, likely needs custom Hessian path
  - Layer pattern: 3× DeltaNet→MoE per group, then 1× Attention→MoE — mixed block types
- [ ] Implement/patch NanoQuant for `Qwen3.5MoE` / DeltaNet weight matrices
- [ ] Download `Qwen/Qwen3.5-397B-A17B` (~794 GB BF16) to cloud storage
- [ ] Run calibration + first 5 blocks as timed feasibility probe
  - **Gate:** DeltaNet blocks quantize without errors; MoE blocks (512 experts) complete in reasonable time
- [ ] Run full 60-block reconstruction + scale refinement
  - Checkpoint every block to allow resume
- [ ] **Pass gate:** WikiText-2 perplexity ≤ 2× BF16 baseline
- [ ] Validation: MMLU, GSM8K, multilingual sample
- [ ] Inference benchmark: tokens/sec on target hardware
- [ ] Upload quantized model to HuggingFace

---

## Cloud GPU Options

### Recommended providers (affordable, GPU-cloud)

| Provider | GPU | VRAM | Price/hr (est.) | Notes |
|----------|-----|------|-----------------|-------|
| [Vast.ai](https://vast.ai) | H100 SXM5 | 80GB | $2.00–3.50 | Spot/interruptible available; lowest cost for long runs |
| [RunPod](https://runpod.io) | H100 PCIe | 80GB | $2.69–3.49 | Reliable, persistent storage, preemptible discounts |
| [Lambda Labs](https://lambdalabs.com) | H100 SXM5 | 80GB | $2.49 | On-demand, stable, NFS shared storage |
| [CoreWeave](https://coreweave.com) | H100 | 80GB | ~$2.80–4.00 | Enterprise; bulk discount for long reservations |
| [Modal](https://modal.com) | H100 | 80GB | ~$3.20 (per sec billing) | Best for short feasibility runs; pay only when running |

**Recommendation:**
- **Phase 0 (feasibility)**: Modal — pay-per-second, no minimum, ideal for short gate checks (~$5–20)
- **Phase 1–4 (full run)**: Vast.ai interruptible + checkpointing, or Lambda on-demand for reliability

### Multi-GPU configuration

For 4× H100 (recommended for full run):

| Provider | 4× H100 | Est. cost for 40h run |
|----------|---------|-----------------------|
| Vast.ai | ~$10–14/hr | ~$400–560 |
| Lambda | ~$9.96/hr (cluster) | ~$400 |
| RunPod | ~$11–14/hr | ~$440–560 |

### Cost guard-rails

1. **Time budget cap**: Set a hard wall-clock limit before launching (e.g., `timeout 48h ./run_nanoquant.sh`)
2. **Block-level checkpointing**: Save after every block; restart from last checkpoint if interrupted
3. **Feasibility gate (Phase 0)**: Spend <$30 validating before committing to full run
4. **Perplexity monitoring**: Log PPL after every 5 blocks; abort if diverging
5. **Spot instance strategy**: Use interruptible pricing with auto-checkpoint/resume to cut costs 50–70%
6. **Early exit flag**: `--max-blocks N` to run partial quantization for benchmarking intermediate quality

### Estimated total cost

| Scenario | Config | Time | Cost |
|----------|--------|------|------|
| Feasibility gate only | 1× H100 (Modal) | 2–4h | $6–15 |
| Optimistic full run | 4× H100 (Vast.ai spot) | 20–40h | $200–560 |
| Conservative full run | 2× H100 (Lambda on-demand) | 40–80h | $400–800 |
| Worst case | 1× H100, serial | 74–100h | $185–350 (spot) |

---

## Quick Start

```bash
pip install -r requirements.txt

# Stage 1 — quick 3-block feasibility probe (no GPU required if using --device cpu)
python scripts/run_stage1.py \
  --output ./outputs/qwen1.5-moe-a2.7b-nanoquant \
  --max-blocks 3 \
  --rank 8 \
  --device cuda   # or cpu

# Stage 1 — full run
python scripts/run_stage1.py \
  --output ./outputs/qwen1.5-moe-a2.7b-nanoquant \
  --rank 8 \
  --checkpoint-dir ./checkpoints/stage1

# Evaluate perplexity standalone
python scripts/eval_ppl.py \
  --model ./outputs/qwen1.5-moe-a2.7b-nanoquant \
  --baseline Qwen/Qwen1.5-MoE-A2.7B
```

**Key flags:**
| Flag | Default | Notes |
|------|---------|-------|
| `--rank` | 8 | Factorization rank; lower = more compression, lower quality |
| `--max-blocks` | all | Limit blocks for fast feasibility probes |
| `--checkpoint-dir` | `output/checkpoints` | Auto-resume if interrupted |
| `--admm-iters` | 50 | ADMM convergence iterations |
| `--tpre` / `--tpost` | 200 | FP pre-tune / STE post-refine steps |
| `--tglob` | 1000 | Global KD scale optimization steps |

---

## Open Questions

- Does the published NanoQuant code handle MoE routing (expert dispatch, shared expert) natively?
- Does DeltaNet's linear-attention weight structure (QK/V with different head counts) require custom Hessian handling?
- Is block-sequential processing compatible with the DeltaNet recurrent state propagation?
- Target inference hardware: will compressed model be served on CPU, consumer GPU, or datacenter?

---

## Progress Log

| Date | Milestone | Status |
|------|-----------|--------|
| 2026-02-20 | README / planning | ✅ |
| 2026-02-20 | Core implementation (hessian, admm, reconstruct, refine, moe, quantize) | ✅ |
| — | Stage 1: Qwen1.5-MoE-A2.7B local feasibility | ⬜ |
| — | Stage 1 gate: PPL ≤ 2× baseline | ⬜ |
| — | Stage 2: Qwen3-235B-A22B cloud probe | ⬜ |
| — | Stage 2 gate: PPL ≤ 2× baseline | ⬜ |
| — | Stage 3: Qwen3.5-397B-A17B DeltaNet patch | ⬜ |
| — | Stage 3: full run + validation | ⬜ |
| — | HuggingFace upload | ⬜ |

