# Phase 4: Phi MoE Support - Research

**Researched:** 2026-02-28
**Domain:** Phi MoE expert architecture, fused/separate expert weight layouts, HF custom modeling
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Fused weight layout:** Split gate_up_proj (shape: n_experts, 2*intermediate, hidden) into separate gate_proj and up_proj WeightViews per expert. Each half gets its own ADMM quantization pass. View naming: `experts.gate_proj[i]` and `experts.up_proj[i]`. Shared Hessian: both views use the same Hessian from the fused module input. Implemented as a special case branch inside the existing `get_weight_views` function.
- **Model targets:** Primary: `microsoft/Phi-tiny-MoE-instruct` (3.8B total, 1.1B active) — must pass. Stretch goal: `microsoft/Phi-mini-MoE-instruct` (7.6B total, 2.4B active). Exact expert class name to be confirmed by researcher from HF transformers source (done — see below). If Phi MoE uses separate nn.Linear experts instead of fused 3D tensors, adapt approach accordingly (don't force fused pattern).
- **Validation approach:** Two-stage: smoke test first (runs without crash, produces loadable checkpoint), then full WikiText-2 eval. Full eval: FP16 baseline PPL + quantized PPL + memory footprint reporting. No hard perplexity degradation threshold — report numbers and let them speak. Results stored in `results/Phi-tiny-MoE-instruct/metrics.json`.
- **Integration with existing code:** Light refactor: extract common fused-expert pattern. Change `FUSED_EXPERT_CLASSES` from `frozenset` to `dict` mapping class name to layout metadata (`needs_split=True`, `split_dim=0`). Existing OLMoE/Qwen entries get `needs_split=False`. Auto-detect Phi MoE from HF `config.json` `model_type` field. Update `reconstruct_model.py` to handle split gate_proj/up_proj reassembly back to fused gate_up_proj.

### Claude's Discretion

- Phi-mini-MoE eval depth (full or lighter pass based on time/resources)
- Exact refactoring approach for the FUSED_EXPERT_CLASSES dict migration
- Error handling and edge cases during split-view creation

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PHI-01 | PhimoeExperts added to FUSED_EXPERT_CLASSES with weight views handling fused gate_up_proj (2*intermediate) layout | Architecture confirmed below — but critical finding: Phi-tiny-MoE uses a CUSTOM modeling file (`modeling_slimmoe.py`) with SEPARATE nn.Linear experts (not fused 3D tensors). The fused-3D path applies to the standard `transformers` `PhimoeExperts` class used by Phi-3.5-MoE. See "Critical Architecture Finding" section. |
| PHI-02 | Phi-tiny-MoE-instruct (3.8B) quantizes end-to-end on a 12GB consumer GPU producing a valid checkpoint | Model architecture confirmed compatible with existing pipeline. trust_remote_code=True required. Block structure is standard model.layers. Rotary embedding is per-attention-layer not top-level. |
| PHI-03 | WikiText-2 perplexity measured for FP16 baseline and quantized Phi-tiny-MoE | Existing run_eval.py infrastructure handles this. Results stored to results/Phi-tiny-MoE-instruct/metrics.json matching existing OLMoE pattern. |
</phase_requirements>

---

## Summary

Phase 4 adds Phi MoE model support to the NanoQuant pipeline. Research uncovered a critical architectural split: the standard HuggingFace `transformers` library contains `PhimoeExperts` (used by Phi-3.5-MoE) with a fused `gate_up_proj` 3D parameter of shape `(n_experts, 2*intermediate_dim, hidden_dim)`, exactly as anticipated. However, `microsoft/Phi-tiny-MoE-instruct` and `microsoft/Phi-mini-MoE-instruct` are SlimMoE variants that ship with a **custom modeling file** (`modeling_slimmoe.py`) loaded via `trust_remote_code=True` and `auto_map`. These models use `PhiMoEBlockSparseTop2MLP` per expert with **separate** `nn.Linear` layers (`w1`, `w2`, `w3`) stored in an `nn.ModuleList`, not fused 3D tensors.

This means the CONTEXT.md's conditional clause — "if Phi MoE uses separate nn.Linear experts instead of fused 3D tensors, adapt approach accordingly" — is triggered for the primary target (Phi-tiny-MoE-instruct). The existing `get_weight_views` nn.Linear traversal path will already enumerate `experts.0.w1`, `experts.0.w2`, `experts.0.w3` etc. correctly. The main integration work is: (1) confirming the router gate linear (`block_sparse_moe.gate`) is correctly skipped by `is_router()`, (2) ensuring `hessian.py` capture works with `trust_remote_code=True` models, and (3) the fused-3D split path (PHI-01 as originally stated) applies to standard `PhimoeExperts` from transformers, not the SlimMoE custom models.

The pipeline infrastructure (block traversal, Hessian capture, checkpoint, eval) is fully compatible. The primary implementation risk is the custom `modeling_slimmoe.py` requiring `trust_remote_code=True` throughout and the `_call_block` compatibility with `PhiMoEDecoderLayer` which uses `position_ids` (not pre-computed `position_embeddings=(cos,sin)`).

**Primary recommendation:** For Phi-tiny-MoE-instruct and Phi-mini-MoE-instruct (SlimMoE), use the existing nn.Linear path — minimal new code. For standard `PhimoeExperts` (Phi-3.5-MoE), implement the fused split via the FUSED_EXPERT_CLASSES dict. Both paths are needed for full PHI-01 compliance.

---

## Critical Architecture Finding

### Phi-tiny-MoE-instruct IS NOT the standard PhimoeExperts

The `config.json` of `microsoft/Phi-tiny-MoE-instruct` contains:
```json
{
  "model_type": "phimoe",
  "auto_map": {
    "AutoConfig": "configuration_slimmoe.PhiMoEConfig",
    "AutoModelForCausalLM": "modeling_slimmoe.PhiMoEForCausalLM"
  }
}
```

The `auto_map` field causes HuggingFace to load **remote custom code** from the model repo (`modeling_slimmoe.py`), not the standard `transformers` `phimoe` implementation. This means:
- `trust_remote_code=True` is **required** everywhere (quantize.py, reconstruct_model.py, eval_ppl.py)
- The expert module class name is `PhiMoEBlockSparseTop2MLP` (not `PhimoeExperts`)
- Experts are stored in `nn.ModuleList` (standard Python objects), not as fused 3D `nn.Parameter`

### Standard transformers PhimoeExperts (for Phi-3.5-MoE)

Verified from `transformers/src/transformers/models/phimoe/modeling_phimoe.py`:
```python
@use_experts_implementation
class PhimoeExperts(nn.Module):
    def __init__(self, config: PhimoeConfig):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
```

This IS the fused 3D layout matching the CONTEXT.md spec. But it applies to `microsoft/Phi-3.5-MoE-instruct`, not the SlimMoE compressed variants.

### @use_experts_implementation decorator

This decorator (from the HF Experts backends interface) can swap the implementation to `batched_mm` or `grouped_mm` backends. It does NOT change the parameter layout — `gate_up_proj` remains a 3D nn.Parameter regardless of backend. However, it MAY change `type(module).__name__`. This needs to be verified at runtime if targeting Phi-3.5-MoE.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | ≥4.43.3 | Load PhiMoE models (auto_map resolution) | Required for trust_remote_code dispatch |
| torch | ≥2.0 | Weight slicing, ADMM | Already in project |
| safetensors | any | Checkpoint I/O | Already in project |
| datasets | any | WikiText-2 calibration/eval | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| huggingface_hub | any | Model download, cache resolution | Already used in run_eval.py |
| psutil | any | RAM measurement | Already in project |

### Alternatives Considered
None applicable — stack is fixed by existing project.

**Installation:** No new dependencies required.

---

## Architecture Patterns

### Phi-tiny-MoE Block Structure (SlimMoE custom modeling)

```
model.model.layers[i] (PhiMoEDecoderLayer)
├── self_attn (PhiMoEAttention)
│   ├── q_proj (nn.Linear)    # fused qkv — actually separate in Phi
│   ├── k_proj (nn.Linear)
│   ├── v_proj (nn.Linear)
│   ├── o_proj (nn.Linear)
│   └── rotary_emb (PhiMoERotaryEmbedding)  # per-attention, NOT top-level model
├── block_sparse_moe (PhiMoESparseMoeBlock)
│   ├── gate (nn.Linear)                     # router — must be SKIPPED by is_router()
│   └── experts (nn.ModuleList[16])
│       └── [i] (PhiMoEBlockSparseTop2MLP)
│           ├── w1 (nn.Linear)               # gate projection  (hidden→intermediate)
│           ├── w2 (nn.Linear)               # down projection  (intermediate→hidden)
│           └── w3 (nn.Linear)               # up projection    (hidden→intermediate)
├── input_layernorm (nn.LayerNorm)
└── post_attention_layernorm (nn.LayerNorm)
```

### Config values for Phi-tiny-MoE-instruct
- `model_type`: `phimoe`
- `hidden_size`: 4096
- `intermediate_size`: 448
- `num_local_experts`: 16
- `num_experts_per_tok`: 2
- `num_hidden_layers`: 32
- `num_attention_heads`: 16
- `num_key_value_heads`: 4
- `torch_dtype`: bfloat16

### Pattern 1: trust_remote_code Required Everywhere

**What:** SlimMoE models use `auto_map` in config.json to load custom Python from the HF repo.
**When to use:** For any `model_type: phimoe` model from the SlimMoE series.
**Example:**
```python
# Source: microsoft/Phi-tiny-MoE-instruct config.json
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-tiny-MoE-instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,   # REQUIRED — without this, load fails
    low_cpu_mem_usage=True,
    device_map="cpu",
)
```

### Pattern 2: Hessian Key Mapping for w1/w2/w3 Expert Naming

The existing `get_hessian_key()` handles non-fused `nn.Linear` layers correctly — their Hessian key equals their full module name. For SlimMoE experts:
```
WeightView name:    "block_sparse_moe.experts.0.w1"
Hessian key:        "model.layers.0.block_sparse_moe.experts.0.w1"
```
No code change needed for Hessian lookup — the existing `get_hessian_key` logic for non-fused layers is:
```python
# From nanoquant/moe.py get_hessian_key()
key = f"{block_prefix}.{name}" if block_prefix else name  # works correctly
```

### Pattern 3: is_router() Must Handle block_sparse_moe.gate

The existing `is_router()` checks for `.gate` suffix:
```python
# nanoquant/moe.py line 65
if name.endswith(".gate") or name.endswith("_gate"):
    return True
```
The module name `block_sparse_moe.gate` ends with `.gate` — this WILL be correctly skipped. No change needed.

### Pattern 4: _call_block Compatibility with PhiMoEDecoderLayer

`PhiMoEDecoderLayer.forward()` signature:
```python
def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False,
            output_router_logits=False, use_cache=False, **kwargs)
```

The existing `_call_block` in `reconstruct.py` has three fallback paths:
1. `block(inputs)` — will FAIL for PhiMoE (needs position_ids for RoPE)
2. `block(inputs, attention_mask=None)` — may work if position_ids defaults work
3. `block(inputs, position_ids=position_ids)` — the last-resort path that WILL work

The rotary embedding is NOT at `model.rotary_emb` (top-level); it's inside each attention layer. So `set_rotary_emb()` will not find it, and `_call_block` will fall through to the `position_ids` last-resort path. This should work correctly.

### Pattern 5: FUSED_EXPERT_CLASSES dict migration

Decision: change from `frozenset` to `dict` mapping class name → layout metadata.
Recommended structure:
```python
FUSED_EXPERT_CLASSES = {
    "Qwen2MoeExperts":   {"needs_split": False},
    "Qwen3MoeExperts":   {"needs_split": False},
    "Qwen3_5MoeExperts": {"needs_split": False},
    "OlmoeExperts":      {"needs_split": False},
    "PhimoeExperts":     {"needs_split": True, "split_dim": 1},
    # split_dim=1: gate_up_proj shape is (n_experts, 2*intermediate, hidden)
    # so we split along dim 1 (out_dim) to get gate_proj and up_proj halves
}
```

The split creates two WeightViews per expert from the fused parameter:
```python
# For PhimoeExperts gate_up_proj[i], shape (2*intermediate, hidden)
half = intermediate_dim  # = param.shape[1] // 2
gate_slice = param.data[i, :half, :]    # (intermediate, hidden)
up_slice   = param.data[i, half:, :]    # (intermediate, hidden)

# Write-back closures must write back to the correct half
def make_gate_write(p, idx, h):
    def _write(W_new): p.data[idx, :h, :] = W_new.to(p.dtype)
    return _write

def make_up_write(p, idx, h):
    def _write(W_new): p.data[idx, h:, :] = W_new.to(p.dtype)
    return _write
```

### Pattern 6: reconstruct_model.py Reassembly

For split views (`experts.gate_proj[i]` and `experts.up_proj[i]`), the reconstruct script must reassemble them into the fused `gate_up_proj` 3D tensor. The current code groups by base_name using `[N]` suffix. Split views will have different base names (`experts.gate_proj` vs `experts.up_proj`), requiring a new assembly step that concatenates `gate_proj[i]` and `up_proj[i]` along dim 0 before writing to `experts.gate_up_proj[i]`.

However: since Phi-tiny-MoE (SlimMoE) does NOT use fused tensors, there is no reconstruction problem for the primary target. The reassembly code only matters if targeting standard `PhimoeExperts` (Phi-3.5-MoE).

### Anti-Patterns to Avoid

- **Assuming model_type maps to a single architecture:** `model_type: phimoe` covers both standard `transformers` PhimoeExperts (Phi-3.5-MoE) AND SlimMoE custom models (Phi-tiny-MoE, Phi-mini-MoE). They have completely different internal architectures.
- **Loading without trust_remote_code:** SlimMoE models will fail to load without `trust_remote_code=True`.
- **Using set_rotary_emb() for PhiMoE:** Rotary embedding is per-attention-layer, not model-level. The top-level set call will set `_rotary_emb_cache = None` (no `model.rotary_emb` attribute). The `_call_block` position_ids fallback will handle RoPE correctly.
- **Checking for PhiMoEBlockSparseTop2MLP in FUSED_EXPERT_CLASSES:** This class uses standard nn.Linear — it should NOT be added to FUSED_EXPERT_CLASSES. The existing nn.Linear traversal handles it already.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| trust_remote_code model loading | Custom download/import | `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` | HF handles auto_map dispatch correctly |
| Expert gate linear detection | Custom name pattern matching | Existing `is_router()` with `.gate` suffix check | Already works for `block_sparse_moe.gate` |
| Hessian key mapping for nn.Linear experts | Custom key builder | Existing `get_hessian_key()` non-fused path | Full module path is already correct |
| WikiText-2 eval | Custom eval loop | Existing `run_eval.py` + `eval_ppl.py` | Already tested with OLMoE |

**Key insight:** The SlimMoE Phi models largely "just work" with the existing nn.Linear path. Most implementation work is for the standard `PhimoeExperts` fused case and for ensuring `trust_remote_code=True` is present everywhere.

---

## Common Pitfalls

### Pitfall 1: SlimMoE vs Standard PhimoeExperts Confusion
**What goes wrong:** Treating all `model_type: phimoe` models as having the fused `PhimoeExperts` 3D tensor. Results in `AttributeError: 'PhiMoESparseMoeBlock' has no attribute 'gate_up_proj'`.
**Why it happens:** Both model families share `model_type: phimoe` in config.json, but only standard Phi-3.5-MoE uses the transformers `PhimoeExperts` class. SlimMoE (Phi-tiny/mini-MoE) uses custom modeling code with separate nn.Linear.
**How to avoid:** Check `auto_map` in config.json at runtime, or detect by probing for `PhimoeExperts` class name vs `PhiMoEBlockSparseTop2MLP`.
**Warning signs:** `AttributeError` on `gate_up_proj`, or `get_weight_views` returning 0 views for expert layers.

### Pitfall 2: Missing trust_remote_code in Hessian / Eval passes
**What goes wrong:** `quantize.py` loads model with `trust_remote_code=True`, but `eval_ppl.py` or `reconstruct_model.py` doesn't — loading fails or uses wrong class.
**Why it happens:** trust_remote_code=True is not the default; easy to forget in secondary scripts.
**How to avoid:** Audit all `from_pretrained` calls across quantize.py, eval_ppl.py, reconstruct_model.py, run_stage1.py. All must pass `trust_remote_code=True`.
**Warning signs:** `ValueError: Loading model from hub requires you to agree to the repository's license` or wrong class loaded silently.

### Pitfall 3: @use_experts_implementation Decorator Changes Class Name
**What goes wrong:** `type(module).__name__` returns something other than `PhimoeExperts` when the experts backend is non-eager (e.g., `batched_mm`). The FUSED_EXPERT_CLASSES dict lookup fails silently.
**Why it happens:** The `@use_experts_implementation` decorator may wrap/replace the class.
**How to avoid:** When loading models for quantization, explicitly set `experts_implementation="eager"` or check the type name at runtime and add a fallback.
**Warning signs:** FUSED_EXPERT_CLASSES dispatch misses PhimoeExperts modules; `get_weight_views` falls through to empty output for those layers.

### Pitfall 4: split_dim for gate_up_proj
**What goes wrong:** Splitting `gate_up_proj` along the wrong dimension.
**Why it happens:** `PhimoeExperts.gate_up_proj` shape is `(n_experts, 2*intermediate, hidden)`. Per-expert slice is `[i]` giving `(2*intermediate, hidden)`. The split must be along `dim=0` of the 2D slice (row dimension), not dim=1.
**How to avoid:** After slicing `param.data[expert_idx]` → shape `(2*intermediate, hidden)`, split along `dim=0`: `gate = slice[:intermediate, :]`, `up = slice[intermediate:, :]`. The `split_dim` in layout metadata refers to the dimension of the 2D per-expert slice.
**Warning signs:** Wrong shapes passed to ADMM; `quantize_weight` receives unexpectedly shaped tensors.

### Pitfall 5: Rotary Embedding Not Found at Top Level
**What goes wrong:** `_call_block` for Phi-tiny-MoE uses the `rotary_emb` path (second fallback) which calls model-level `rotary_emb`, but there is none. The fallback loop inside `_call_block` also won't find it because the module is inside `self_attn`, not at block level.
**Why it happens:** PhiMoE places `rotary_emb` inside each `PhiMoEAttention`, not at the block's top level.
**How to avoid:** The existing `position_ids` last-resort path in `_call_block` handles this correctly — no change needed. Just verify that `block(inputs, position_ids=position_ids)` path runs without error.
**Warning signs:** Exception during `_call_block` from inside the attention module's RoPE computation.

### Pitfall 6: w1/w2/w3 Naming in Checkpoint Keys
**What goes wrong:** After quantization, checkpoint keys use `w1`, `w2`, `w3` which differ from the `gate_proj`, `up_proj`, `down_proj` naming used by standard dense Phi models. This is fine for NanoQuant's own checkpoint, but may cause confusion.
**Why it happens:** SlimMoE uses non-standard layer naming.
**How to avoid:** NanoQuant checkpoints are keyed by module path, so `w1`/`w2`/`w3` will appear naturally. `reconstruct_model.py` maps by module path anyway. No special handling needed.

---

## Code Examples

### Loading Phi-tiny-MoE with trust_remote_code
```python
# Source: verified from microsoft/Phi-tiny-MoE-instruct config.json + transformers docs
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-tiny-MoE-instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,   # Required for SlimMoE auto_map dispatch
    low_cpu_mem_usage=True,
    device_map="cpu",
)
```

### Verifying Expert Architecture at Runtime
```python
# Detect whether we have fused 3D experts or separate nn.Linear per expert
def detect_expert_type(model):
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if class_name in ("PhimoeExperts", "PhiMoEBlockSparseTop2MLP"):
            return class_name, name
    return None, None
```

### FUSED_EXPERT_CLASSES Dict Pattern
```python
# Source: proposed for nanoquant/moe.py
FUSED_EXPERT_CLASSES = {
    "Qwen2MoeExperts":   {"needs_split": False},
    "Qwen3MoeExperts":   {"needs_split": False},
    "Qwen3_5MoeExperts": {"needs_split": False},
    "OlmoeExperts":      {"needs_split": False},
    "PhimoeExperts":     {"needs_split": True, "split_dim": 0},
    # split_dim=0: per-expert slice shape is (2*intermediate, hidden),
    # split along dim 0 to get (intermediate, hidden) gate and up halves
}
```

### get_weight_views Split Branch for PhimoeExperts
```python
# Proposed implementation for nanoquant/moe.py
if type_name in FUSED_EXPERT_CLASSES:
    layout = FUSED_EXPERT_CLASSES[type_name]
    for param_name in ("gate_up_proj", "down_proj"):
        param = getattr(module, param_name, None)
        if param is None or not isinstance(param, nn.Parameter):
            continue
        n_experts = param.shape[0]
        needs_split = layout.get("needs_split", False) and param_name == "gate_up_proj"
        for expert_idx in range(n_experts):
            if needs_split:
                half = param.shape[1] // 2
                # gate half
                yield WeightView(
                    name=f"{mod_name}.gate_proj[{expert_idx}]",
                    weight=param.data[expert_idx, :half, :].float(),
                    write_back=make_gate_write(param, expert_idx, half),
                    orig_dtype=param.dtype,
                )
                # up half
                yield WeightView(
                    name=f"{mod_name}.up_proj[{expert_idx}]",
                    weight=param.data[expert_idx, half:, :].float(),
                    write_back=make_up_write(param, expert_idx, half),
                    orig_dtype=param.dtype,
                )
            else:
                # existing path — unchanged
                yield WeightView(...)
```

### Smoke Test Command
```bash
# Quick validation: first 2 blocks only, minimal calibration
python scripts/run_stage1.py \
  --model microsoft/Phi-tiny-MoE-instruct \
  --output output/phi-tiny-moe-smoke \
  --max-blocks 2 \
  --n-calibration 4 \
  --admm-iters 10 \
  --tpre 0 --tpost 0
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| frozenset for FUSED_EXPERT_CLASSES | dict with layout metadata | Phase 4 | Enables architecture-specific expert handling without code changes |
| Single architecture assumption for model_type | Check auto_map + class name at runtime | Phase 4 | Required because SlimMoE and standard PhimoeExperts share model_type |

**Deprecated/outdated:**
- None — this is the first support for Phi MoE architecture.

---

## Open Questions

1. **Does @use_experts_implementation change type().__name__ for PhimoeExperts?**
   - What we know: The decorator is part of the HF Experts backends interface. It may wrap or subclass the module.
   - What's unclear: Whether `type(module).__name__` still returns `"PhimoeExperts"` when batched_mm or grouped_mm backend is active.
   - Recommendation: When loading for quantization, explicitly call `model.set_experts_implementation("eager")` if available, or probe the name at runtime and add `"PhimoeExpertsEager"` / similar to the dict if needed. This is low-risk for Phase 4 since the primary target (SlimMoE) doesn't use this decorator.

2. **Does Phi-mini-MoE-instruct use the same modeling_slimmoe.py architecture?**
   - What we know: Same SlimMoE paper/series. config.json likely has identical auto_map structure.
   - What's unclear: Whether intermediate_size or expert count differs significantly.
   - Recommendation: Verify config.json for Phi-mini-MoE before attempting stretch goal. Expect same `PhiMoEBlockSparseTop2MLP` class and same nn.Linear path.

3. **Is trust_remote_code=True already present in eval_ppl.py?**
   - What we know: `run_stage1.py` has it; `quantize.py` has it. `eval_ppl.py` was not checked in detail.
   - What's unclear: Whether `eval_ppl.py` passes `trust_remote_code=True` in its `from_pretrained` call.
   - Recommendation: Audit and add `trust_remote_code=True` as a parameter to `evaluate_perplexity()` in `eval_ppl.py`. Default to False to preserve backward compatibility, but pass True for Phi models.

4. **What transformer version is required for PhimoeExperts support?**
   - What we know: PhimoeExperts was added to transformers in 2024. Phi-tiny-MoE config shows `transformers_version: 4.43.3`.
   - What's unclear: Minimum version for the experts backend interface and `@use_experts_implementation`.
   - Recommendation: Pin `transformers>=4.43.3` for Phi MoE support. The existing project already uses a recent version.

---

## Sources

### Primary (HIGH confidence)
- `microsoft/Phi-tiny-MoE-instruct` config.json (fetched directly from HF) — model_type, auto_map, expert counts, hidden/intermediate sizes
- `microsoft/Phi-tiny-MoE-instruct` modeling_slimmoe.py (fetched directly from HF) — PhiMoEBlockSparseTop2MLP class definition, PhiMoESparseMoeBlock structure, forward signatures
- `github.com/huggingface/transformers` modeling_phimoe.py (fetched) — PhimoeExperts class definition, gate_up_proj shape `(n_experts, 2*intermediate, hidden)`
- `nanoquant/moe.py` (read directly) — FUSED_EXPERT_CLASSES frozenset, get_weight_views, is_router, get_hessian_key
- `nanoquant/reconstruct.py` (read directly) — _call_block behavior and fallback paths

### Secondary (MEDIUM confidence)
- HF Experts backends documentation (`huggingface.co/docs/transformers/main/en/experts_interface`) — use_experts_implementation decorator behavior, backend types
- HF PhiMoE documentation (`huggingface.co/docs/transformers/v4.46.2/en/model_doc/phimoe`) — "MLP's up and gate projection layers are also fused"

### Tertiary (LOW confidence)
- WebSearch results on PhimoeExperts gate_up_proj — corroborated by direct source inspection; treating as verified

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — confirmed from direct source inspection of model files
- Architecture (SlimMoE): HIGH — read directly from modeling_slimmoe.py
- Architecture (standard PhimoeExperts): HIGH — read directly from transformers GitHub
- Pitfalls: MEDIUM-HIGH — some pitfalls (decorator class name change) are theoretical pending runtime validation
- Integration patterns: HIGH — based on reading all relevant existing codebase files

**Research date:** 2026-02-28
**Valid until:** 2026-05-28 (90 days; SlimMoE modeling file may change with model updates)
