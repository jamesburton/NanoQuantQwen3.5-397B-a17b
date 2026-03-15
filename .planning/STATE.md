---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: verifying
stopped_at: Completed 03-02 Tasks 1+2, at checkpoint Task 3 (human verification pending)
last_updated: "2026-03-15T12:00:32.863Z"
last_activity: "2026-03-08 — Completed plan 04-03 (gap closure: factor key fix + WikiText-2 eval)"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 11
  completed_plans: 10
  percent: 91
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Consumer-friendly sub-1-bit quantization of MoE models via NanoQuant's ADMM-based pipeline
**Current focus:** Phase 4 complete. Next: Phase 3 (Scaling and Evaluation)

Phase 3 KD status:
- Reconstruction path optimized: Phi block time reduced from about 329s to about 95s on local RTX 3060 workspace.
- Dead end recorded: hook-based connected-scale KD is too slow on this machine and does not justify current runtime cost.
- Dead end recorded: initial factor-scale prototype was removed because it wrote weights outside autograd.
- Tested implementation target: connected SlimMoE expert override for `block_sparse_moe.experts.*.w1/w2/w3` is now in place. Both `factor_scales` and `factor_latents` probes hit the Phase 3 FP-cache budget limit on this hardware (`sample_too_slow` at about 87-88s for the first KD sample).

## Current Position

Phase: 4 of 4 (Phi MoE Support)
Plan: 3 of 3 in current phase (complete)
Status: Phase 4 complete — all 3 plans done, all verification gaps closed
Last activity: 2026-03-08 — Completed plan 04-03 (gap closure: factor key fix + WikiText-2 eval)

Progress: [█████████░] 91%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

*Updated after each plan completion*
| Phase 01-pipeline-foundation P03 | 2 | 1 tasks | 2 files |
| Phase 02-moe-support P01 | 5 | 1 tasks | 1 files |
| Phase 02-moe-support P02 | 10 | 2 tasks | 3 files |
| Phase 03 P01 | 2 | 2 tasks | 3 files |
| Phase 04-phi-moe-support P02 | - | 3 tasks (1 deferred) | 4 files |
| Phase 04-phi-moe-support P03 | - | 2 tasks | 5 files |
| Phase 03-scaling-and-evaluation P02 | 120 | 2 tasks | 3 files |

## Accumulated Context

### Decisions

- model_load_strategy always "cpu" — block-by-block .to(device) is correct for quantization; device_map="auto" scatters layers and breaks sequential processing
- Safetensors dotted key format (layer_name.tensor_key) chosen for MoE expert forward-compatibility
- 1 GB GPU headroom + 10% CPU headroom hardcoded as constants, not user-configurable
- psutil import at module top-level in quantize.py (was inline in Phase 3 KD block)
- eval_device in run_stage1.py derives from torch.cuda.is_available() independently of quantize_model hardware detection
- bits_per_weight -> rank: rank = max(1, int(bpw * d / 2)) using first Linear layer's min dimension
- [Phase 01-pipeline-foundation]: print_hardware_summary shows compute= and model_load= as separate fields for clarity
- [Phase 02-moe-support]: Use frozenset dispatch table (FUSED_EXPERT_CLASSES) for fused expert module type matching — enables adding new architectures by updating one set
- [Phase 04-phi-moe-support P01]: FUSED_EXPERT_CLASSES migrated from frozenset to dict with needs_split/split_dim metadata — each architecture declares its own fused tensor layout
- [Phase 04-phi-moe-support P01]: PhimoeExperts gate_up_proj split_dim=0 means dim 0 of per-expert 2D slice (out-dim); gate=[:half,:], up=[half:,:]
- [Phase 04-phi-moe-support P01]: gate_proj[N] and up_proj[N] split views both map to same Hessian key — both halves share the same expert input activations
- [Phase 02-moe-support P02]: shared. prefix chosen for norm/embedding keys to distinguish from binary factor dotted keys in safetensors
- [Phase 02-moe-support P02]: collect_shared_layers uses named_modules traversal with class-name check to stay architecture-agnostic
- [Phase 03]: evaluate_perplexity returns dict (ppl, n_tokens, settings) for structured downstream consumption
- [Phase 03]: dtype=auto default for baseline eval respects model native dtype (e.g. bfloat16 for Qwen3.5-397B)
- [Phase 04-phi-moe-support P02]: trust_remote_code=True already present on all from_pretrained calls from 04-01 work — Task 3 was a no-op
- [Phase 04-phi-moe-support P02]: SlimMoE uses nn.Linear experts (w1,w2,w3), not fused 3D tensors — existing nn.Linear traversal path handles them
- [Phase 04-phi-moe-support P03]: Factor key collision bug — block-relative keys caused all 32 blocks to overwrite each other; fixed with block_prefix
- [Phase 04-phi-moe-support P03]: CPU float32 fallback for eval pipeline (float16 matmuls produce NaN on CPU)
- [Phase 03-scaling-and-evaluation]: Document partial quantization in metrics.json notes field with blocks_quantized/blocks_total when ADMM stops early
- [Phase 03-scaling-and-evaluation]: When consumer GPU eval infeasible (CPU offload too slow), record null PPL with eval_skip_reason and defer to cloud GPU

### Roadmap Evolution

- Phase 4 added: Phi MoE Support — PhimoeExperts fused gate_up_proj layout + Phi-tiny-MoE-instruct local validation
- Phase 4 completed: All 3 plans done, PHI-01/02/03 satisfied
- Phase 3 restructured (2026-03-15): 03-02 partial results unacceptable as cloud gate; added 03-02b (fix OLMoE all 16 blocks + usable PPL) and 03-02c (Qwen1.5 approach decision) before 03-03 can proceed

### Pending Todos

- Phase 03 plans must execute sequentially (03-02 before 03-03) — resource contention + 03-02 results feed 03-03 summary
- Review upstream origin/main changes backed up under backups/origin-main-2026-03-10/ before next merge attempt:
- nanoquant/quantize.py: upstream CPU thread tuning and float32 CPU load path reviewed; deferred because local Phase 3 KD refactor diverged substantially and needs targeted benchmarking before cherry-picking
- scripts/run_stage1.py: upstream CPU thread configuration and float32 eval load path reviewed; deferred because local CLI/workflow diverged (--skip-eval) and needs manual merge if CPU-only eval becomes a priority
- scripts/patch_slimmoe_compat.py: reviewed; local version already contains stronger idempotent repair logic than upstream
- scripts/monitor_quant.py: upstream helper integrated locally
- Merge decision for now: keep local working versions because local experimental changes are broader and current eval/test signal is not strong enough to justify replacing them blindly

### Blockers/Concerns

- SCALE-04 (Qwen3.5-397B) requires cloud GPU — consumer 12GB constraint applies to phases 1-2 and SCALE-01 through SCALE-03 only
- Phase 4 PPL degradation (141x) is high due to zero-refinement settings; CUDA re-run with defaults expected to improve to ~1.5-3x
- Local branch is behind origin/main by 2 commits; upstream overlap has been backed up rather than merged because verification coverage remains weak and recent eval outcomes are not yet convincing

## Session Continuity

Last session: 2026-03-15T12:00:32.857Z
Stopped at: Phase 3 restructured — executing 03-02b (fix OLMoE all blocks + usable PPL)
Resume file: None



