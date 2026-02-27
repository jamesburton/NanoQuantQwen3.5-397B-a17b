---
status: complete
phase: 02-moe-support
source: [02-01-SUMMARY.md, 02-02-SUMMARY.md]
started: 2026-02-27T20:54:08Z
updated: 2026-02-27T20:58:00Z
---

## Current Test

[testing complete]

## Tests

### 1. FUSED_EXPERT_CLASSES dispatch table
expected: `nanoquant/moe.py` contains a `FUSED_EXPERT_CLASSES` frozenset with all four expert class names: OlmoeExperts, Qwen2MoeExperts, Qwen3MoeExperts, Qwen3_5MoeExperts
result: pass

### 2. get_weight_views uses dispatch table
expected: `get_weight_views()` in `nanoquant/moe.py` checks `type_name in FUSED_EXPERT_CLASSES` instead of hardcoded `"Qwen2MoeExperts"` string
result: pass

### 3. Ancestor check uses dispatch table
expected: The nn.Linear ancestor check in `get_weight_views()` uses `in FUSED_EXPERT_CLASSES` to prevent double-yielding fused expert weights for all four architectures
result: pass

### 4. collect_shared_layers captures norms and embeddings
expected: `collect_shared_layers(model)` in `nanoquant/checkpoint.py` returns a dict with `shared.` prefixed keys containing RMSNorm/LayerNorm weights, embed_tokens, lm_head, and final norm as FP16 tensors
result: pass

### 5. Self-contained checkpoint with shared layers
expected: `save_quantized_checkpoint` accepts a `shared_layers` dict and merges it into the safetensors file alongside binary factors. Manifest JSON includes a `shared_layers` key list.
result: pass

### 6. Architecture detection at model load
expected: `detect_architecture()` uses `AutoConfig.from_pretrained` and prints `Architecture: {model_type}` during `quantize_model` execution
result: pass

### 7. --parallel-blocks CLI stub
expected: `python scripts/run_stage1.py --help` shows a `--parallel-blocks` argument (int, default 1)
result: pass

## Summary

total: 7
passed: 7
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
