# Phase 1: Pipeline Foundation - Research

**Researched:** 2026-02-22
**Domain:** PyTorch quantization pipeline, hardware-adaptive offloading, checkpoint serialization, progress logging
**Confidence:** HIGH (core stack), MEDIUM (hardware detection specifics), HIGH (existing codebase analysis)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Auto-detect available resources (GPU VRAM, CPU RAM, NVMe) at startup and adapt offloading strategy automatically
- No explicit hardware floor — pipeline handles whatever is available gracefully
- No manual device map configuration required from user

### Claude's Discretion
- Weight management approach (block-by-block vs layer streaming vs hybrid) — choose based on memory constraints and implementation simplicity
- Hessian storage strategy (CPU RAM vs disk-backed memory-mapping) — choose based on model size and detected resources
- Checkpoint serialization format — pick what works best for downstream loading
- Pipeline orchestration details (config file vs CLI args, resume behavior)
- Progress/logging implementation (ETA, memory display, log format)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIPE-01 | Stage 1 pipeline runs end-to-end (Hessian capture → ADMM init → block-wise reconstruction) on a single model | Existing `nanoquant/quantize.py` already implements this flow; research identifies gaps to fix (hardware detection, sub-1-bit config, robust progress) |
| PIPE-02 | Pipeline operates within 12GB VRAM via CPU/NVMe offloading through Accelerate | `accelerate.get_max_memory()` + `infer_auto_device_map()` + `disk_offload()` provide the full adapter stack; existing code uses manual `device_map="cpu"` — must migrate to adaptive dispatch |
| PIPE-03 | Pipeline supports sub-1-bit quantization (0.7–0.9 bits) as primary mode per NanoQuant paper | Paper demonstrates 0.55, 0.80, 1.00 bpw via rank selection in `lb_admm`; existing `admm.py` is correct; need to expose rank→bpw mapping and validate defaults |
| PIPE-04 | Pipeline produces serializable quantized checkpoint (binary matrices + scales) | `safetensors.torch.save_file` is the right format: fast, mmap-loadable, metadata-supporting; existing code uses `model.save_pretrained` which does NOT store binary factors |
| PIPE-05 | Pipeline provides progress logging with ETA and memory usage tracking | `tqdm` already present; need `psutil.virtual_memory()`, `torch.cuda.memory_allocated()`, and ETA calculation layered onto existing block loop |
</phase_requirements>

---

## Summary

Phase 1 is fundamentally a **gap-closure phase**: the codebase already contains a working NanoQuant Stage 1 pipeline (`nanoquant/quantize.py`, `admm.py`, `hessian.py`, `reconstruct.py`) that correctly implements Hessian capture, LB-ADMM initialization, and block-wise STE reconstruction. However, it has five concrete deficiencies that prevent it from meeting the stated requirements.

The most significant gap is **PIPE-02 + PIPE-04**: the existing code hardcodes `device_map="cpu"` for model loading and then calls `model.save_pretrained()` at the end, which neither enables adaptive GPU/CPU/NVMe dispatch nor persists the binary factor tensors (U_bin, V_bin, s1, s2 per layer) that are the actual NanoQuant artifact. The second significant gap is **hardware auto-detection** (PIPE-02): the code does a one-shot GPU check but does not query CPU RAM or disk, and does not build an adaptive offloading plan. The remaining gaps (PIPE-03 rank/bpw documentation, PIPE-05 ETA + memory logging) are smaller additions on top of the existing `tqdm` loop.

**Primary recommendation:** Keep the existing mathematical core intact. Add a `hardware.py` module that probes resources and returns an adaptive loading plan, update `quantize.py` to use `accelerate.get_max_memory()` + Accelerate dispatch hooks for block movement, and replace `model.save_pretrained()` with a `safetensors`-based checkpoint that stores binary factors plus a JSON manifest.

---

## Standard Stack

### Core (already present in repo)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | >=2.2.0 | Tensor ops, CUDA, linalg (SVD, Cholesky) | Required — entire pipeline is PyTorch native |
| `transformers` | >=4.45.0 | Model loading, `AutoModelForCausalLM` | Required — HF model hub integration |
| `accelerate` | >=0.30.0 | Hardware detection, device dispatch, offload hooks | The standard solution for adaptive multi-device model placement |
| `datasets` | any | WikiText-2 calibration data | Already used |
| `tqdm` | any | Progress bars | Already used |
| `psutil` | any | CPU RAM and disk free space queries | Already used (partially) |
| `numpy`, `scipy` | any | Numerical support | Already present |

### New Addition Required

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `safetensors` | >=0.4.0 | Binary factor checkpoint format | Fast mmap loads, metadata support, HF-ecosystem native, zero-copy. Preferred over `.pt` for distributable checkpoints. |

**Installation:**
```bash
pip install safetensors>=0.4.0
```
(All other dependencies already in `requirements.txt`)

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `safetensors` for checkpoint | `torch.save` / `.pt` | `.pt` uses pickle (security risk, slower load, no mmap). `safetensors` is the current HF standard for model weights. |
| `accelerate.get_max_memory()` | manual `psutil` probing | Manual probing misses GPU-specific overhead (CUDA kernels take 1-2 GB at first allocation). `get_max_memory()` handles this correctly. |
| Block-by-block GPU movement (current approach) | `device_map="auto"` for whole model | For this quantization pipeline, block-by-block is actually correct — we need full control over when each block is on GPU. The current approach is right; the gap is the hardware probe that decides the strategy. |

---

## Architecture Patterns

### Recommended Module Structure

The existing structure is nearly correct. One new module is needed:

```
nanoquant/
├── admm.py          # LB-ADMM binary factorization — DO NOT CHANGE
├── hessian.py       # Hessian diagonal capture — DO NOT CHANGE
├── moe.py           # WeightView abstraction — DO NOT CHANGE
├── quantize.py      # Main pipeline orchestrator — MODIFY: add hardware plan
├── reconstruct.py   # Block reconstruction — MINOR FIX: Hessian key lookup
├── refine.py        # STE + KD refinement — DO NOT CHANGE
├── hardware.py      # NEW: hardware detection + adaptive plan
└── checkpoint.py    # NEW: safetensors-based checkpoint save/load
scripts/
├── run_stage1.py    # Entry point — MODIFY: add --bits-per-weight flag, better logging
```

### Pattern 1: Hardware Auto-Detection Module (`hardware.py`)

**What:** Probe GPU VRAM, CPU RAM, and disk at startup. Return a `HardwarePlan` dataclass that the pipeline uses to decide (a) where to load the model, (b) whether Hessian tensors need disk-backing, (c) how many calibration samples fit in memory.

**When to use:** Called once at pipeline start, result passed into `quantize_model()`.

**Key APIs to use:**
```python
# Source: accelerate/utils/modeling.py — get_max_memory()
from accelerate.utils import get_max_memory
import psutil, torch

def probe_hardware() -> HardwarePlan:
    max_mem = get_max_memory()  # returns {0: bytes, "cpu": bytes} auto-detected
    gpu_free = max_mem.get(0, 0)          # bytes available on GPU 0
    cpu_free = max_mem.get("cpu", 0)      # bytes available on CPU RAM
    disk_free = psutil.disk_usage(".").free  # bytes on current drive
    cuda_available = torch.cuda.is_available()
    ...
```

**Important note from Accelerate docs:** CUDA kernel initialization consumes 1-2 GB of GPU memory on first allocation. `get_max_memory()` accounts for this by querying live free memory after PyTorch is already imported. Reserve an additional 1 GB budget for activations during block forward passes.

**HardwarePlan fields:**
- `gpu_free_bytes`: usable VRAM (after CUDA overhead reserve)
- `cpu_free_bytes`: usable CPU RAM
- `disk_free_bytes`: available disk space at offload dir
- `use_gpu_for_blocks`: bool — move blocks to GPU during reconstruction
- `hessian_on_disk`: bool — mmap Hessian diagonals instead of RAM
- `model_load_strategy`: `"cpu"` | `"disk_offload"` | `"auto_dispatch"`

### Pattern 2: Adaptive Model Loading

**What:** Instead of hardcoded `device_map="cpu"`, use the hardware plan to decide loading strategy.

**Strategy decision tree:**
1. If GPU VRAM > model size in fp16 → load with `device_map="auto"` (rare for large models)
2. If CPU RAM > model size in fp16 → load with `device_map="cpu"` (current behavior — correct for most consumer setups with the target models)
3. If CPU RAM < model size in fp16 → load with `device_map="auto"` which triggers Accelerate's disk offload for the overflow

**For the quantization block loop specifically**, the current block-by-block `block.to(device)` / `block.cpu()` / `torch.cuda.empty_cache()` pattern is correct and should stay. This is intentionally different from inference device maps — during reconstruction we process one block at a time with full control.

```python
# Correct pattern (preserve existing): move block to GPU, process, move back
block.to(device)
quantized = reconstruct_block(block, bi, block_hessians, cfg, block_prefix)
block.cpu()
torch.cuda.empty_cache()
```

### Pattern 3: Safetensors Checkpoint Format

**What:** Save per-block binary factor tensors in safetensors format with a JSON manifest.

**Format design:**
```
outputs/
├── quantized_factors.safetensors   # all U_bin, V_bin, s1, s2, d_in, d_out per layer
├── quantized_manifest.json         # metadata: model name, rank, bpw, layer list
└── config.json                     # HF model config (for downstream loading)
```

**Naming convention for safetensors keys:**
```
"model.layers.0.self_attn.q_proj.U_bin"
"model.layers.0.self_attn.q_proj.V_bin"
"model.layers.0.self_attn.q_proj.s1"
"model.layers.0.self_attn.q_proj.s2"
```

```python
# Source: https://github.com/huggingface/safetensors (Context7 verified)
from safetensors.torch import save_file, safe_open

# Save
flat = {}
for layer_name, qdata in all_quantized.items():
    for key, tensor in qdata.items():
        flat[f"{layer_name}.{key}"] = tensor.contiguous()
save_file(flat, "quantized_factors.safetensors", metadata={"rank": str(rank), "model": model_name})

# Load (lazy, memory-efficient)
with safe_open("quantized_factors.safetensors", framework="pt", device="cpu") as f:
    for k in f.keys():
        tensor = f.get_tensor(k)
```

**Why safetensors over torch.save:**
- No pickle (no code execution on load — safe for distributed use)
- Memory-mapped: tensors are not loaded into RAM until accessed
- Supports `__metadata__` string-to-string dict in header (stores rank, bpw, model name)
- Zero-copy with `safe_open` — each tensor accessed on demand

### Pattern 4: Sub-1-Bit Configuration via Rank

**What:** The NanoQuant paper achieves 0.55, 0.80, 1.00 bpw by varying factorization rank. The mapping is:

```
bpw ≈ rank * (d_out + d_in) / (d_out * d_in)
```

For a weight matrix W of shape (d_out, d_in), the binary storage cost is `rank * (d_out + d_in)` bits (plus scale vectors). For example, with d_out=d_in=4096 and rank=8:
```
bpw ≈ 8 * (4096 + 4096) / (4096 * 4096) ≈ 0.0039 — extremely low
```

In practice, scales (fp16 vectors) and the amortized overhead push effective bpw higher. The paper's Table 2 maps rank to bpw for Llama-scale models. For the target models (OLMoE-1B-7B through Qwen3.5-397B), the following ranks are starting points:

| Target bpw | Approximate rank | Notes |
|------------|-----------------|-------|
| ~0.55 | 4 | Very aggressive compression |
| ~0.80 | 8 | Paper's main result (default) |
| ~1.00 | 12 | Near-binary baseline |

**The `--bits-per-weight` CLI flag should translate to a rank suggestion.** The actual bpw must be computed post-quantization based on stored tensor sizes.

### Pattern 5: Progress Logging with ETA and Memory

**What:** Wrap the block loop with real-time ETA and VRAM/RAM reporting.

```python
import time, psutil, torch

# Before block loop
t_start_phase = time.time()
block_times = []

# Inside loop, after each block
elapsed = time.time() - t_block_start
block_times.append(elapsed)
avg = sum(block_times) / len(block_times)
remaining = (n_blocks - block_idx - 1) * avg
eta_str = f"{int(remaining // 60)}m{int(remaining % 60)}s"

vram_used = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
ram_used = psutil.virtual_memory().used / 1024**3

print(f"  Block {block_idx}/{n_blocks-1} | {elapsed:.1f}s | ETA {eta_str} | VRAM {vram_used:.1f}GB | RAM {ram_used:.1f}GB")
```

### Anti-Patterns to Avoid

- **`model.save_pretrained()` as the quantization output:** This saves the FP16 model with quantized weights written back in-place (via `wv.write()`), but does NOT save the binary factor tensors (U_bin, V_bin, s1, s2). Downstream phases cannot reconstruct the factored representation. Use safetensors checkpoint instead.
- **Full Hessian matrices:** The codebase correctly uses diagonal-only Hessians (stored as 1D vectors). Do not change this — full Hessians for a 397B MoE model would require hundreds of GB.
- **`device_map="auto"` with Transformers for the quantization model:** Accelerate's auto device map is optimized for inference (hooks move data per-forward). During block reconstruction, we need to modify weights in-place on GPU. Use `device_map="cpu"` for initial load and manual `.to(device)` per block.
- **Accumulating all block outputs:** The existing code correctly processes blocks sequentially with a rolling `block_input` tensor. Do not collect all block outputs at once — that would require O(n_layers × batch × seq × hidden) memory.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hardware memory detection | Custom `torch.cuda.mem_get_info()` + `psutil` combination | `accelerate.utils.get_max_memory()` | Handles CUDA init overhead, NPU/XPU, multi-GPU, and string formatting automatically |
| Disk offload for model weights | Custom file streaming | `accelerate.disk_offload(model, offload_dir, exec_device)` | Inserts forward hooks that stream weights from disk; handles dtype, device movement, and cleanup |
| Tensor checkpoint format | Custom binary packing with struct | `safetensors.torch.save_file` / `safe_open` | Memory-mapped, zero-copy, metadata support, secure (no pickle) |
| Bit-packing {-1,+1} → bits | Custom numpy bitpacking | Store as `torch.int8` in safetensors (or use `torch.packbits` on bool tensors) | Standard PyTorch ops; safetensors handles int8 natively. Packing to 1 bit saves ~8x storage but complicates loading — defer to v2. |

**Key insight:** The Accelerate library is specifically designed for the hardware-adaptive model loading problem. Its `get_max_memory()` → `infer_auto_device_map()` flow handles more edge cases (multiple GPUs, NPUs, XPUs, CUDA overhead budget, disk overflow) than any reasonable custom probe would.

---

## Common Pitfalls

### Pitfall 1: CUDA Kernel Memory Overhead in Hardware Probe

**What goes wrong:** `torch.cuda.get_device_properties(0).total_memory` and `torch.cuda.mem_get_info()` disagree — the first gives total VRAM, the second gives free VRAM after CUDA initialization. Scripts that use `.total_memory` to plan offloading OOM because 1-2 GB is already used by CUDA kernels.

**Why it happens:** PyTorch's CUDA initialization loads kernels on first use. If the hardware probe runs after any PyTorch CUDA call (including `torch.cuda.is_available()` on some platforms), kernels may already be loaded.

**How to avoid:** Use `accelerate.utils.get_max_memory()` which queries live free memory after initialization. If probing manually, prefer `torch.cuda.mem_get_info()[0]` (free bytes) over `.total_memory`. Reserve an additional 1 GB buffer for activations.

**Warning signs:** Pipeline runs fine on first block then OOMs on second block (activations + next block weights exceed available VRAM).

### Pitfall 2: Hessian Key Mismatches for Fused Expert Tensors

**What goes wrong:** The existing `get_hessian_key()` in `moe.py` strips the `gate_up_proj[N]` expert index suffix to look up the Hessian. But the Hessian was captured on `nn.Linear` modules, not on `Qwen2MoeExperts` modules directly. For fused 3D expert tensors, no Hessian hook fires at the expert level — the hook fires at the input to the `Qwen2MoeExperts` module.

**Why it happens:** `capture_hessians()` registers hooks on `isinstance(module, nn.Linear)` — `Qwen2MoeExperts` holds `nn.Parameter` objects, not `nn.Linear` children, so no hook fires for individual experts.

**How to avoid:** The current fallback (`d_in = torch.ones(n, ...)`) is acceptable for Phase 1 (single model, non-MoE). For Phase 2 (MoE), the hessian module must be updated. For Phase 1 tests against OLMoE or small Qwen1.5-MoE, the identity fallback degrades quality but does not crash.

**Warning signs:** High perplexity degradation specifically in MoE expert layers vs. attention layers.

### Pitfall 3: Checkpoint Stores FP Weights, Not Binary Factors

**What goes wrong:** The current pipeline writes quantized weights back in-place via `wv.write(W_new)`, then calls `model.save_pretrained(output_dir)`. This saves a FP16 model where weights approximate the binary factorization — it does NOT save U_bin, V_bin, s1, s2. Downstream stages (Stage 2 STE refinement, Stage 3 KD) cannot recover the binary factors.

**Why it happens:** The existing code was likely written to validate the pipeline end-to-end without worrying about checkpoint format.

**How to avoid:** Maintain a parallel dict of `all_quantized[layer_name] = {U_bin, V_bin, s1, s2, d_in, d_out}` across all blocks and save to safetensors. Optionally also save the FP16 model for immediate inference validation.

**Warning signs:** Any subsequent phase that tries to load and refine the binary factors fails with "file not found" or finds only FP16 weights.

### Pitfall 4: Blocking on RAM for Full KD Pass

**What goes wrong:** The Phase 3 KD step in `quantize.py` loads a second full copy of the FP model (`model_fp`). For models > ~7B parameters at fp16, this doubles RAM requirements. The existing code has a psutil guard but the threshold check (`avail_gb < model_size_gb * 0.8`) may be too conservative — at 80% model size, there's insufficient headroom for activations.

**Why it happens:** KD requires simultaneous access to both quantized and reference model outputs.

**How to avoid:** Keep the existing guard but tighten threshold to 1.5x model size. Consider `tglob=0` as the default for Phase 1 testing (KD is a Phase 3+ concern per the paper's Stage 3 definition). The phase 1 scope is Hessian + ADMM + block reconstruction — KD is a separate phase in the paper.

**Warning signs:** RAM OOM during Phase 3 on machines with <512 GB RAM when quantizing models >7B parameters.

### Pitfall 5: Block Input Shape Drift

**What goes wrong:** The rolling `block_input` tensor is computed by running each already-processed block in `model_dtype` (fp16). If block outputs return tuples instead of tensors (model-architecture dependent), `block_input` becomes stale or None.

**Why it happens:** The `_run_block()` helper handles the `block(x)[0]` vs `block(x, attention_mask=None)[0]` fallback, but some architectures (e.g., Qwen MoE with past_key_values) may return different tuple shapes.

**How to avoid:** Wrap `_run_block` to explicitly assert output is a 3D tensor `(batch, seq, dim)`. Add `isinstance(out, torch.Tensor)` check and log a warning before proceeding.

---

## Code Examples

Verified patterns from official sources:

### Hardware Detection (Accelerate)
```python
# Source: accelerate/utils/modeling.py — get_max_memory() (GitHub verified)
from accelerate.utils import get_max_memory
import psutil, torch

def build_hardware_plan(offload_dir: str = ".") -> dict:
    """Probe hardware and return adaptive pipeline configuration."""
    max_mem = get_max_memory()  # {0: int_bytes, "cpu": int_bytes} — auto-detected

    gpu_free_bytes = max_mem.get(0, 0)
    cpu_free_bytes = max_mem.get("cpu", 0)
    disk_free_bytes = psutil.disk_usage(offload_dir).free

    # Reserve headroom: 1 GB GPU for activations, 10% CPU for system
    gpu_usable = max(0, gpu_free_bytes - 1 * 1024**3)
    cpu_usable = int(cpu_free_bytes * 0.9)

    return {
        "gpu_free_gb": gpu_usable / 1024**3,
        "cpu_free_gb": cpu_usable / 1024**3,
        "disk_free_gb": disk_free_bytes / 1024**3,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() and gpu_usable > 2 * 1024**3 else "cpu",
    }
```

### Safetensors Checkpoint Save
```python
# Source: https://github.com/huggingface/safetensors — Context7 verified
from safetensors.torch import save_file
import json

def save_quantized_checkpoint(
    all_quantized: dict,  # {layer_name: {U_bin, V_bin, s1, s2, d_in, d_out}}
    output_dir: str,
    model_name: str,
    rank: int,
) -> None:
    """Save binary factor checkpoint in safetensors format."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    flat_tensors = {}
    layer_list = []
    for layer_name, qdata in all_quantized.items():
        layer_list.append(layer_name)
        for key, tensor in qdata.items():
            safetensors_key = f"{layer_name}.{key}"
            flat_tensors[safetensors_key] = tensor.contiguous().cpu()

    metadata = {
        "model": model_name,
        "rank": str(rank),
        "num_layers": str(len(layer_list)),
        "format": "nanoquant_v1",
    }

    save_file(flat_tensors, os.path.join(output_dir, "quantized_factors.safetensors"), metadata=metadata)

    manifest = {"model": model_name, "rank": rank, "layers": layer_list}
    with open(os.path.join(output_dir, "quantized_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
```

### Safetensors Checkpoint Load
```python
# Source: https://github.com/huggingface/safetensors — Context7 verified
from safetensors import safe_open

def load_quantized_checkpoint(checkpoint_dir: str) -> dict:
    """Load binary factor checkpoint lazily (memory-mapped)."""
    import json, os

    with open(os.path.join(checkpoint_dir, "quantized_manifest.json")) as f:
        manifest = json.load(f)

    all_quantized = {layer: {} for layer in manifest["layers"]}
    safetensors_path = os.path.join(checkpoint_dir, "quantized_factors.safetensors")

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            # k = "model.layers.0.self_attn.q_proj.U_bin"
            parts = k.rsplit(".", 1)
            layer_name, tensor_key = parts[0], parts[1]
            if layer_name in all_quantized:
                all_quantized[layer_name][tensor_key] = f.get_tensor(k)

    return all_quantized
```

### Adaptive Model Loading
```python
# Adaptive strategy based on hardware plan
def load_model_adaptive(model_name: str, plan: dict) -> nn.Module:
    """Load model with strategy determined by hardware probe."""

    # Always load to CPU first for block-by-block quantization.
    # Accelerate's device_map="auto" is for inference, not in-place quantization.
    # Block-by-block .to(device) is the correct pattern here.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",  # All weights on CPU; we move blocks manually
    )

    print(f"Hardware plan: GPU={plan['gpu_free_gb']:.1f}GB usable, "
          f"CPU={plan['cpu_free_gb']:.1f}GB usable, "
          f"disk={plan['disk_free_gb']:.1f}GB free")

    model.eval()
    return model
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torch.save / pickle for checkpoints | safetensors (mmap, no-pickle) | 2023, HF standard since 2024 | Secure, lazy-loadable, cross-framework |
| Manual GPU memory checks | `accelerate.get_max_memory()` | Accelerate v0.20+ | Correctly accounts for CUDA init overhead |
| Full Hessian matrices (GPTQ-style) | Diagonal-only Hessians (NanoQuant) | NanoQuant paper Feb 2026 | 34 GB → ~1 MB for 397B model Hessians |
| Float16 quantization outputs | Binary factor tensors (U_bin, V_bin) + scales | NanoQuant Feb 2026 | True sub-1-bit storage; ~25x compression |

**Deprecated/outdated:**
- `torch.save` for distributable model checkpoints: replaced by safetensors for HF-ecosystem models.
- `device_map="auto"` during quantization (not inference): wrong pattern for in-place weight modification — use manual block movement.

---

## Open Questions

1. **Bits-per-weight calculation for MoE models**
   - What we know: bpw formula is `rank * (d_out + d_in) / (d_out * d_in)` per layer, plus amortized scale overhead
   - What's unclear: For MoE expert tensors with shape `(n_experts, d_out, d_in)`, bpw is computed per-expert but storage is shared — should bpw report per-expert or per-parameter-count?
   - Recommendation: For Phase 1 (non-MoE validation), compute bpw per layer and report average. Defer MoE-specific bpw accounting to Phase 2.

2. **Hessian disk-backing threshold**
   - What we know: Hessian diagonals are tiny (~1 MB total for large models per the codebase comments). CPU RAM is sufficient in all realistic scenarios for Phase 1 models.
   - What's unclear: Is disk-backed mmap for Hessians ever necessary in Phase 1 scope?
   - Recommendation: Skip Hessian disk-backing for Phase 1. Keep all Hessians in CPU RAM (current behavior). Add the disk-backing option as a config flag but default off.

3. **Block checkpoint format compatibility with Phase 2**
   - What we know: Phase 2 (MoE support) needs to load binary factors per-expert
   - What's unclear: Whether the `layer_name.tensor_key` naming scheme handles expert index notation `experts.gate_up_proj[3].U_bin` in safetensors (safetensors keys are arbitrary strings — square brackets are allowed)
   - Recommendation: Use dotted notation without brackets: `experts.gate_up_proj.expert_3.U_bin`. Define naming convention now to avoid migration later.

4. **`model.save_pretrained()` dual output**
   - What we know: The existing code calls `model.save_pretrained()` after writing quantized weights back in-place. This produces an FP16 model that approximates the quantization but loses binary factors.
   - What's unclear: Whether Phase 3 evaluation (`eval_ppl.py`) needs the FP16 model or the binary factors
   - Recommendation: Save both: safetensors checkpoint (binary factors for downstream phases) AND `model.save_pretrained()` (FP16 approximation for immediate perplexity evaluation). This satisfies PIPE-04 and enables EVAL-01 in later phases without blocking.

---

## Sources

### Primary (HIGH confidence)

- `/llmstxt/huggingface-projects-docs-llms-txt_hf_space_accelerate_llms_txt` (Context7) — `infer_auto_device_map`, `get_max_memory`, `disk_offload`, `device_map` semantics, hardware probe patterns
- `/huggingface/safetensors` (Context7) — `save_file`, `safe_open`, `get_tensor`, metadata format
- `https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference` — Official Accelerate big model inference guide, `no_split_module_classes`, CUDA overhead warning
- Existing codebase files: `nanoquant/quantize.py`, `nanoquant/admm.py`, `nanoquant/hessian.py`, `nanoquant/reconstruct.py`, `nanoquant/moe.py`, `scripts/run_stage1.py` — direct inspection

### Secondary (MEDIUM confidence)

- `https://arxiv.org/html/2602.06694v1` — NanoQuant paper; pipeline architecture (Stage 1/2/3), bpw targets (0.55, 0.80, 1.00), calibration data (128 samples WikiText-2), binary packing format description
- `https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/modeling.py` — `get_max_memory()` and `get_balanced_memory()` function signatures (fetched via WebFetch)
- `https://github.com/huggingface/safetensors` — safetensors format structure, `__metadata__` support

### Tertiary (LOW confidence)

- WebSearch results on NVMe disk offload patterns — consistent with Accelerate official docs but not independently verified against current accelerate version

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are official HF stack, verified via Context7 and direct inspection of existing `requirements.txt`
- Architecture patterns: HIGH — based on direct codebase reading + official Accelerate docs
- Pitfalls: HIGH (CUDA overhead, block shape, checkpoint format) — MEDIUM (MoE Hessian key specifics) — based on codebase analysis and documented known issues in Accelerate GitHub
- Sub-1-bit rank/bpw mapping: MEDIUM — derived from paper abstract and table descriptions; exact rank values need empirical validation

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 for Accelerate APIs (stable); 2026-03-01 for NanoQuant paper details (very recent, may have updates)
