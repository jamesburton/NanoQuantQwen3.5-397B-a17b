---
status: complete
phase: 01-pipeline-foundation
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md]
started: 2026-02-22T22:42:06Z
updated: 2026-02-23T00:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Hardware auto-detection
expected: `probe_hardware()` returns a HardwarePlan dataclass and `print_hardware_summary()` prints GPU/CPU/disk GB + strategy in a readable one-liner
result: issue
reported: "Summary shows strategy=cpu even with 10GB GPU. Should display device=cuda and clarify what is on cpu vs gpu. Also need dstorage-gpu (https://pypi.org/project/dstorage-gpu/) for Windows DirectStorage direct-to-GPU loading."
severity: major

### 2. CPU-only graceful fallback
expected: If no GPU available, `probe_hardware()` still succeeds with gpu_free_bytes=0 and device=cpu
result: pass

### 3. Checkpoint save/load roundtrip
expected: `save_quantized_checkpoint()` writes a .safetensors file with dotted keys; `load_quantized_checkpoint()` reads it back with identical tensor values (int8 and float)
result: pass

### 4. Quantize pipeline runs with hardware detection
expected: Running `python scripts/run_stage1.py --help` shows `--bits-per-weight` flag and does NOT show `--device` flag
result: pass

### 5. Per-block progress logging
expected: During quantization, each block prints a progress line like `Block N/M | Xs | ETA Xm Xs | VRAM X.XGB | RAM X.XGB | N layers`
result: pass

### 6. Safetensors checkpoint written after quantization
expected: After `quantize_model()` completes, a `quantized_factors.safetensors` file is written alongside the FP16 model output
result: pass

### 7. Effective bits-per-weight reported
expected: After saving, the effective bpw computed from actual U_bin/V_bin tensor sizes is printed to console
result: pass

## Summary

total: 7
passed: 6
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "print_hardware_summary() shows clear GPU vs CPU usage and device assignment"
  status: failed
  reason: "User reported: Summary shows strategy=cpu even with 10GB GPU. Should display device=cuda and clarify what is on cpu vs gpu. Also need dstorage-gpu for Windows DirectStorage direct-to-GPU loading."
  severity: major
  test: 1
  artifacts: []
  missing: []
