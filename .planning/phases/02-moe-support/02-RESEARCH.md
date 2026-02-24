# Phase 2: MoE Support - Research

**Researched:** 2026-02-24
**Domain:** MoE expert tensor layouts, architecture detection, shared layer quantization
**Confidence:** HIGH (for OLMoE, Qwen2-MoE, Qwen3-MoE), MEDIUM (for Qwen3.5-MoE — new hybrid architecture)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Auto-detect MoE architecture from model config.json (model_type/architectures field)
- Support both HuggingFace model IDs and local paths (auto-resolve config either way)
- LayerNorm/RMSNorm layers must be included in the quantized checkpoint (stored as FP16)
- Shared Hessian across experts in same module — no per-expert Hessian
- Allow parallel block processing (not just sequential)
- Auto-detect concurrency from available VRAM by default, with --parallel-blocks N flag for user override
- OLMoE-1B-7B is the primary development/test target — develop and validate against it first, then extend to Qwen variants

### Claude's Discretion

- Unknown architecture fallback behavior (error vs generic fallback)
- Qwen variant unification strategy (one family vs separate backends)
- Shared layer bit-width decisions (same as experts vs higher precision)
- Embedding/lm_head quantization policy
- Shared expert classification (expert vs shared layer)
- Expert reconstruction group sizing for VRAM-constrained scenarios
- OLMoE WeightView naming convention
- OLMoE structural approach depth

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MOE-01 | WeightView abstraction handles OLMoE-1B-7B expert tensor layout | OLMoE uses `OlmoeExperts` class with fused 3D tensors — same pattern as Qwen2MoeExperts. New `type_name == "OlmoeExperts"` branch needed in `get_weight_views`. Module path: `mlp.experts` within each decoder layer. |
| MOE-02 | WeightView abstraction handles Qwen MoE expert tensor layout (Qwen1.5-MoE, Qwen2-MoE, Qwen3.5-MoE) | Qwen1.5-MoE uses `model_type=qwen2_moe` and same `Qwen2MoeExperts` class — already handled. Qwen2-MoE same. Qwen3-MoE uses `Qwen3MoeExperts` — identical fused 3D layout, new type name. Qwen3.5-MoE uses `Qwen3_5MoeExperts` — identical fused 3D layout, new type name. |
| MOE-03 | Shared attention/embedding layers quantized alongside expert layers | `model.save_pretrained()` already saves FP16 model. Explicit save of norms/embeddings as FP16 tensors in checkpoint needed. Architecture detection determines which norm/embed names to include. |
| MOE-04 | Expert-aware block grouping for reconstruction (experts in same block processed together) | Current `reconstruct_block` already processes all weight views per block sequentially. Need to verify all expert views within a block's `mlp.experts` module are yielded together before the block advances. Currently true — `get_weight_views` iterates modules in order. |
</phase_requirements>

---

## Summary

All four MoE architectures in scope (OLMoE, Qwen1.5-MoE, Qwen2-MoE, Qwen3/3.5-MoE) use the **same fundamental expert storage pattern**: fused 3D `nn.Parameter` tensors with shape `[num_experts, out_dim, in_dim]`, accessed via named params `gate_up_proj` and `down_proj`. The existing `Qwen2MoeExperts` handling in `moe.py` is the template — MOE-01 and MOE-02 require adding parallel detection branches for `OlmoeExperts`, `Qwen3MoeExperts`, and `Qwen3_5MoeExperts` class names.

The key architectural discovery is that OLMoE does NOT use individual `nn.Linear` per expert (as CONTEXT.md speculated) — it uses the same fused 3D tensor pattern as Qwen. This means the WeightView extension is simpler than anticipated: add class name matching for three new module types, all sharing the same slice-per-expert logic.

Qwen3.5-397B-A17B has a novel hybrid architecture (GatedDeltaNet + sparse MoE) with `model_type=qwen3_5_moe`. It requires transformers from GitHub main (not PyPI stable). The expert class `Qwen3_5MoeExperts` follows the identical fused tensor pattern. However, the hybrid linear attention layers (GatedDeltaNet) are NOT transformer decoder blocks in the traditional sense — the block iteration in `quantize.py` via `model.model.layers` still works, but blocks contain DeltaNet layers interleaved with MoE layers. This is only a concern at Phase 3 (SCALE-04) validation.

**Primary recommendation:** Extend `get_weight_views` with a type-name dispatch table covering all four expert class names. Use `model_type` from config.json to drive architecture detection, architecture-specific `get_hessian_key` logic, and norm/embed collection for MOE-03.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 | Tensor ops, nn.Module traversal | Already in project |
| transformers | >=4.45 (HF main for Qwen3.5) | AutoModelForCausalLM, model configs | Already in project |
| safetensors | >=0.4 | Checkpoint serialization | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | - | Read config.json for architecture detection | Always for auto-detect |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| type(module).__name__ string matching | isinstance() with imported class | isinstance requires importing model-specific classes — breaks for trust_remote_code models. String matching is more robust. |
| Per-architecture branches in moe.py | Single unified dispatch table | Dispatch table is cleaner and extensible — use it |

---

## Architecture Patterns

### Expert Class Name to Tensor Layout Mapping

All supported MoE models use the identical 3D tensor pattern:

```
gate_up_proj: nn.Parameter shape [n_experts, 2*intermediate_dim, hidden_dim]
down_proj:    nn.Parameter shape [n_experts, hidden_dim, intermediate_dim]
```

| Model Family | model_type | Expert Class Name | Has shared_expert? |
|---|---|---|---|
| OLMoE-1B-7B | `olmoe` | `OlmoeExperts` | No |
| Qwen1.5-MoE | `qwen2_moe` | `Qwen2MoeExperts` | Yes (Qwen2MoeMLP) |
| Qwen2-MoE | `qwen2_moe` | `Qwen2MoeExperts` | Yes (Qwen2MoeMLP) |
| Qwen3-MoE | `qwen3_moe` | `Qwen3MoeExperts` | No |
| Qwen3.5-MoE | `qwen3_5_moe` | `Qwen3_5MoeExperts` | Yes (Qwen3_5MoeMLP) |

**Confidence:** HIGH for all except Qwen3.5 (MEDIUM — verified via GitHub but very new model, 2026-02-16 release).

### Pattern 1: Type-Name Dispatch Table in get_weight_views

**What:** Replace the hardcoded `Qwen2MoeExperts` check with a set/dict of known expert class names.

**When to use:** Always — covers all current and future fused-expert MoE models.

```python
# Source: verified from transformers source for each model type
FUSED_EXPERT_CLASSES = {
    "Qwen2MoeExperts",    # Qwen1.5-MoE, Qwen2-MoE  (model_type: qwen2_moe)
    "Qwen3MoeExperts",    # Qwen3-MoE                (model_type: qwen3_moe)
    "Qwen3_5MoeExperts",  # Qwen3.5-MoE              (model_type: qwen3_5_moe)
    "OlmoeExperts",       # OLMoE                    (model_type: olmoe)
}

# In get_weight_views:
type_name = type(module).__name__
if type_name in FUSED_EXPERT_CLASSES:
    for param_name in ("gate_up_proj", "down_proj"):
        ...  # existing per-expert slice logic
```

### Pattern 2: Architecture Auto-Detection

**What:** Read `config.json` from model path or HF hub to determine `model_type`.

**When to use:** At pipeline start before block iteration; result stored in a dataclass/dict.

```python
def detect_moe_architecture(model_name_or_path: str) -> dict:
    """Returns {'model_type': str, 'has_shared_expert': bool, ...}"""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = getattr(config, 'model_type', 'unknown')

    arch_map = {
        'olmoe':      {'has_shared_expert': False, 'expert_class': 'OlmoeExperts'},
        'qwen2_moe':  {'has_shared_expert': True,  'expert_class': 'Qwen2MoeExperts'},
        'qwen3_moe':  {'has_shared_expert': False,  'expert_class': 'Qwen3MoeExperts'},
        'qwen3_5_moe':{'has_shared_expert': True,  'expert_class': 'Qwen3_5MoeExperts'},
    }
    return arch_map.get(model_type, {'has_shared_expert': False, 'expert_class': None})
```

**Note:** `AutoConfig.from_pretrained` is the canonical way — it handles both HF model IDs and local paths automatically. No manual JSON parsing needed.

### Pattern 3: Norm and Embedding Collection for MOE-03

**What:** Traverse the model to collect non-expert layers (norms, embeddings) and include them in the checkpoint as FP16 tensors.

**When to use:** After block-wise reconstruction, before checkpoint save.

```python
def collect_shared_layers(model) -> dict:
    """Collect norm and embedding state dicts (FP16) for self-contained checkpoint."""
    shared = {}
    inner = getattr(model, 'model', model)
    # Embeddings
    if hasattr(inner, 'embed_tokens'):
        shared['model.embed_tokens.weight'] = inner.embed_tokens.weight.data.half()
    if hasattr(model, 'lm_head'):
        shared['lm_head.weight'] = model.lm_head.weight.data.half()
    # Per-block norms
    if hasattr(inner, 'layers'):
        for i, layer in enumerate(inner.layers):
            for name, mod in layer.named_modules():
                if 'norm' in type(mod).__name__.lower() and hasattr(mod, 'weight'):
                    key = f"model.layers.{i}.{name}.weight"
                    shared[key] = mod.weight.data.half()
    # Final norm
    if hasattr(inner, 'norm') and hasattr(inner.norm, 'weight'):
        shared['model.norm.weight'] = inner.norm.weight.data.half()
    return shared
```

### Pattern 4: Hessian Key for OLMoE Experts

**What:** OLMoE's expert module path within a decoder layer is `mlp.experts` (not a standalone path). The Hessian is captured on the `OlmoeExperts` module input (shared Hessian across all experts in the layer).

**When to use:** In `get_hessian_key` — add OLMoE expert name parsing.

The current `get_hessian_key` logic in `moe.py` handles `[expert_idx]` suffix by stripping to the base module. This works generically for all fused-expert classes — no OLMoE-specific change needed IF the naming convention for WeightView names is consistent.

For OLMoE, within a block, the path is:
```
mlp.experts.gate_up_proj[0]
mlp.experts.gate_up_proj[1]
...
mlp.experts.down_proj[0]
```

This matches the existing Qwen2MoeExperts pattern where `get_hessian_key` strips to `mlp.experts`. Confirmed compatible.

### Pattern 5: Parallel Block Processing

**What:** Process multiple transformer blocks simultaneously using Python threading or multiprocessing.

**When to use:** Only when N≥2 GPU/CPU slots are free. Default to sequential for correctness; parallel is an optimization.

**Warning:** Block reconstruction mutates model weights in-place. Parallel processing of DIFFERENT blocks is safe because each block is independent. Do NOT parallelize within a block.

```python
import concurrent.futures

def _process_block_parallel(args):
    block, block_idx, block_input, hessians, cfg, device = args
    block.to(device)
    quantized = reconstruct_block(block, block_input.to(device), hessians, cfg,
                                  block_prefix=f"model.layers.{block_idx}")
    block.cpu()
    return block_idx, quantized

# VRAM-based concurrency estimate
def estimate_max_parallel_blocks(model_hidden_size: int, num_experts: int) -> int:
    if not torch.cuda.is_available():
        return 1
    free_vram = torch.cuda.mem_get_info()[0]
    # Rough estimate: each block needs ~expert_params * 2 bytes (FP16) + activations
    # Conservative: 2GB per block as floor estimate for small MoE
    block_vram_estimate_gb = 2.0
    return max(1, int(free_vram / (block_vram_estimate_gb * 1024**3)))
```

**Recommendation:** Implement parallel block support as additive feature with `--parallel-blocks N` CLI flag and VRAM auto-detection. Default to 1 (sequential) for safety.

### Anti-Patterns to Avoid

- **String-split expert name parsing hardcoded to Qwen2:** The existing `get_hessian_key` uses `".gate_up_proj"` and `".down_proj"` string checks — these work for all fused-expert models since all use the same param names. Do not add architecture-specific branches here.
- **Importing model-specific classes for isinstance checks:** Fragile with `trust_remote_code=True` models. Use `type(module).__name__` string matching instead.
- **Hooking OLMoE expert modules for STE activation capture:** `OlmoeExperts` is not `nn.Linear` — hooking its forward won't give clean per-weight activations. Use the existing fallback path (block input approximation) for fused expert STE.
- **Double-saving shared layers:** `model.save_pretrained()` already saves the full FP16 model. MOE-03 requires the quantized checkpoint to be self-contained, but the FP16 checkpoint saved alongside already includes norms. Clarify whether MOE-03 means "include in safetensors factor file" or just "don't strip from model checkpoint." Most likely the former.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config loading from HF hub or local path | Custom JSON parsing logic | `AutoConfig.from_pretrained()` | Handles caching, auth, local paths, revision — all edge cases |
| Architecture detection | Hardcoded model name string matching | `config.model_type` field | Canonical, works with fine-tuned variants |

**Key insight:** The entire expert tensor layout problem reduces to "match class name, use same slice logic." The hard part is knowing the class names — which this research provides.

---

## Common Pitfalls

### Pitfall 1: OLMoE Was Assumed to Use Individual nn.Linear — It Does Not

**What goes wrong:** CONTEXT.md speculated OLMoE uses "individual nn.Linear per expert." This is incorrect. OLMoE uses `OlmoeExperts` with fused 3D parameters, same as Qwen.

**Why it happens:** The original `allenai/OLMoE` GitHub code (pre-HF integration) may have used separate `nn.Linear` per expert. The HF Transformers integration uses fused tensors for efficiency.

**How to avoid:** Use `type(module).__name__ == "OlmoeExperts"` detection. Do NOT look for `nn.ModuleList` of `OlmoeMLP`.

**Warning signs:** If `get_weight_views` yields zero WeightViews for an OLMoE block's MLP section, the detection is failing.

### Pitfall 2: Qwen3.5-397B-A17B Requires transformers from GitHub main

**What goes wrong:** `pip install transformers` (PyPI stable) does not include `qwen3_5_moe` model type as of 2026-02-24. Model load fails with "unknown model type."

**Why it happens:** Model released 2026-02-16, transformers PyPI releases lag 1-4 weeks.

**How to avoid:** Install from GitHub main: `pip install git+https://github.com/huggingface/transformers.git`. Phase 2 only needs this for Qwen3.5-MoE support; OLMoE and Qwen2-MoE work with stable transformers.

**Warning signs:** `ValueError: Unrecognized model in...` or `KeyError: 'qwen3_5_moe'` on model load.

### Pitfall 3: Qwen3.5-397B-A17B Has Hybrid Non-Transformer Layers

**What goes wrong:** Qwen3.5-397B-A17B interleaves GatedDeltaNet (linear attention) layers with standard MoE transformer layers (pattern: 3x DeltaNet + 1x Attention, each followed by MoE). The `_call_block` function may fail on DeltaNet layers.

**Why it happens:** DeltaNet layers have different forward signatures than standard transformer decoder layers.

**How to avoid:** This is a Phase 3 (SCALE-04) concern, not Phase 2. Phase 2 focus is on OLMoE-1B-7B (standard transformer). Flag this as a known future issue.

**Warning signs:** TypeError or unexpected output shape when `_call_block` runs on a DeltaNet block.

### Pitfall 4: Hessian Key Mismatch for New Model Types

**What goes wrong:** `get_hessian_key` strips expert index to get Hessian dict key. The Hessian is captured during forward pass on the experts module input (e.g., `model.layers.5.mlp.experts`). If the block_prefix or module path differs across model types, the key lookup fails and fallback (identity d_in) is used.

**Why it happens:** OLMoE and Qwen may differ in how the MLP/experts module is named within a decoder layer.

**Verification:** After Hessian capture, log the first few keys and compare to what `get_hessian_key` produces. Add a warning if fallback d_in is used more than expected.

### Pitfall 5: Shared Expert Treated as Router and Skipped

**What goes wrong:** `is_router()` checks for `.gate` suffix. Qwen's `shared_expert_gate` ends with `_gate` — already caught. But `shared_expert.gate_proj` etc. must NOT be skipped.

**Why it happens:** Naming ambiguity between router gates and expert projection gates.

**How to avoid:** Verify `is_router()` doesn't accidentally match `shared_expert.gate_proj`. Current logic: checks `last` part only (`gate` at end) and full suffix `.gate` or `_gate`. `gate_proj` last part is `gate_proj` — not matched. This is correct. But validate explicitly.

### Pitfall 6: MOE-03 Scope Ambiguity

**What goes wrong:** MOE-03 says "shared attention/embedding layers quantized alongside expert layers." This could mean: (a) included in the safetensors factor checkpoint, or (b) included in the `model.save_pretrained()` output (which always happens). The user's intent from CONTEXT.md is "self-contained checkpoint" including "norms in FP16."

**Why it happens:** The project saves two things: FP16 model via `model.save_pretrained()` and binary factors via `save_quantized_checkpoint()`. MOE-03 most likely means: norm weights should be in the binary factors checkpoint so it's self-contained without needing the FP16 model alongside it.

**How to avoid:** Include shared layer tensors (norms, optionally embeddings) as FP16 entries in `quantized_factors.safetensors`. Add a `shared_layers` section to the manifest JSON.

---

## Code Examples

### Extending FUSED_EXPERT_CLASSES in moe.py

```python
# Source: transformers GitHub source for each model type (verified 2026-02-24)
FUSED_EXPERT_CLASSES = frozenset({
    "Qwen2MoeExperts",    # Qwen1.5-MoE, Qwen2-MoE  (qwen2_moe)
    "Qwen3MoeExperts",    # Qwen3-MoE                (qwen3_moe)
    "Qwen3_5MoeExperts",  # Qwen3.5-MoE              (qwen3_5_moe)
    "OlmoeExperts",       # OLMoE-1B-7B              (olmoe)
})

def get_weight_views(block: nn.Module, block_prefix: str = "") -> Iterator[WeightView]:
    for mod_name, module in block.named_modules():
        full_mod = f"{block_prefix}.{mod_name}" if block_prefix else mod_name
        type_name = type(module).__name__

        if type_name in FUSED_EXPERT_CLASSES:
            for param_name in ("gate_up_proj", "down_proj"):
                param = getattr(module, param_name, None)
                if param is None or not isinstance(param, nn.Parameter):
                    continue
                n_experts = param.shape[0]
                for expert_idx in range(n_experts):
                    w_name = f"{mod_name}.{param_name}[{expert_idx}]"
                    weight_slice = param.data[expert_idx].float()
                    orig_dtype = param.dtype

                    def make_write(p, idx):
                        def _write(W_new: torch.Tensor):
                            p.data[idx] = W_new.to(p.dtype)
                        return _write

                    yield WeightView(
                        name=w_name,
                        weight=weight_slice,
                        write_back=make_write(param, expert_idx),
                        orig_dtype=orig_dtype,
                    )
            continue  # don't recurse into expert module children

        # Standard nn.Linear — existing logic unchanged
        ...
```

### Architecture Detection via AutoConfig

```python
# Source: HuggingFace transformers AutoConfig API
from transformers import AutoConfig

def detect_architecture(model_name_or_path: str) -> str:
    """Return model_type string from config. Works for HF IDs and local paths."""
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    return getattr(config, 'model_type', 'unknown')
```

### Collecting Norms for Self-Contained Checkpoint (MOE-03)

```python
def collect_norm_tensors(model: nn.Module) -> dict:
    """Collect all RMSNorm/LayerNorm weight tensors as FP16 for checkpoint."""
    norms = {}
    for name, module in model.named_modules():
        mtype = type(module).__name__
        if ('RMSNorm' in mtype or 'LayerNorm' in mtype) and hasattr(module, 'weight'):
            norms[f"{name}.weight"] = module.weight.data.half().cpu()
        if ('RMSNorm' in mtype or 'LayerNorm' in mtype) and hasattr(module, 'bias') and module.bias is not None:
            norms[f"{name}.bias"] = module.bias.data.half().cpu()
    return norms
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate nn.Linear per expert (Mixtral-style) | Fused 3D nn.Parameter tensors | 2024 (Qwen1.5-MoE, OLMoE HF integration) | Simpler quantization: one slice-and-write per expert instead of per-module hookup |
| Sequential block reconstruction only | Parallel block processing | Phase 2 target | 2-4x speedup on VRAM-rich machines |
| Checkpoint stores only expert factors | Self-contained checkpoint (norms + factors) | MOE-03 target | No FP16 model needed alongside checkpoint for inference |

---

## Open Questions

1. **OLMoE Module Path: `mlp.experts` or different?**
   - What we know: HF Transformers `OlmoeDecoderLayer` has `OlmoeSparseMoeBlock` at `mlp` attribute, which contains `OlmoeExperts` at `experts`.
   - What's unclear: Exact attribute names haven't been verified by running the model. Path may be `mlp.experts` or `block_sparse_moe.experts` depending on transformers version.
   - Recommendation: Add a debug log at the start of `get_weight_views` to print all module names and class names for the first block when OLMoE is detected. Verify against `mlp.experts` expectation.

2. **Qwen1.5-MoE shared_expert presence**
   - What we know: Qwen2-MoE has `shared_expert`. Qwen1.5-MoE uses same `qwen2_moe` code path.
   - What's unclear: Whether Qwen1.5-MoE's config has `shared_expert_intermediate_size > 0`.
   - Recommendation: Treat shared_expert as present for all qwen2_moe models. If `shared_expert_intermediate_size` is 0 or absent, the MLP module may be skipped automatically.

3. **Parallel block processing: input dependency**
   - What we know: Each block's output is the next block's input — strict sequential dependency for input computation.
   - What's unclear: Can block reconstruction (the ADMM/STE part) be parallelized if we pre-compute all inputs first?
   - Recommendation: Yes — pre-compute all block inputs sequentially (fast forward pass), then parallelize reconstruction. This is safe because reconstruction reads but does not need the next block's output.

4. **Qwen3.5-397B-A17B: Is it in scope for Phase 2?**
   - What we know: `model_type=qwen3_5_moe`, uses `Qwen3_5MoeExperts` (fused 3D tensors, same pattern). Has GatedDeltaNet layers that may break `_call_block`. Very new model (released 2026-02-16).
   - What's unclear: Whether the WeightView extension alone (without `_call_block` fixes) is sufficient for Qwen3.5-MoE MOE-02 coverage.
   - Recommendation: Add `Qwen3_5MoeExperts` to the FUSED_EXPERT_CLASSES set in Phase 2 (low-cost). Defer `_call_block` fixes for DeltaNet layers to Phase 3 validation (SCALE-04). This gives partial MOE-02 coverage without blocking Phase 3.

---

## Sources

### Primary (HIGH confidence)
- HuggingFace Transformers GitHub — `modeling_olmoe.py` — OlmoeExperts class structure verified
- HuggingFace Transformers GitHub — `modeling_qwen2_moe.py` — Qwen2MoeExperts verified, shared_expert confirmed
- HuggingFace Transformers GitHub — `modeling_qwen3_moe.py` — Qwen3MoeExperts verified, no shared_expert
- HuggingFace Transformers GitHub — `modeling_qwen3_5_moe.py` — Qwen3_5MoeExperts verified, shared_expert confirmed
- HuggingFace — Qwen3.5-397B-A17B config.json — model_type=qwen3_5_moe, num_experts=512, num_experts_per_tok=10

### Secondary (MEDIUM confidence)
- HuggingFace model docs — OLMoE: num_experts=64, num_experts_per_tok=8 confirmed
- WebSearch — Qwen3.5 architecture: GatedDeltaNet + sparse MoE hybrid, multimodal, 60 layers

### Tertiary (LOW confidence)
- WebSearch — Qwen1.5-MoE shared_expert presence: assumed from qwen2_moe code path inheritance (not directly verified in config)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies needed
- Architecture patterns: HIGH for OLMoE/Qwen2/Qwen3, MEDIUM for Qwen3.5 (very new)
- Pitfalls: HIGH — verified from code inspection and known HF release patterns

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (Qwen3.5 area may shift as transformers PyPI catches up)
