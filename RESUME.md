# Resume: Running NanoQuant on a CPU-Only Machine

## Current Status (2026-03-16)

Phase 03-02b is at Task 4 (human checkpoint).

- **Pipeline smoke test: PASSED** — end-to-end flow works
- **Quality test: FAILED** — bpw=0.80 with 50 ADMM iterations produces unusable PPL on OLMoE-1B-7B-0924

The quantization pipeline is mechanically correct. The remaining question
is whether higher bpw or more ADMM iterations can achieve usable quality
(PPL degradation ≤ 20x).

## Why Transfer to CPU Machine

The block reconstruction phase (Phase 2) is **100% CPU-bound**:
- VRAM usage: 0.1 GB throughout all 16 blocks
- RAM usage: 50-68 GB, peaking at ~72 GB during final save
- The previous run hit MemoryError at 68 GB RAM during safetensors serialization

A high-memory CPU machine avoids the RAM bottleneck and runs Phase 2
at the same speed (no GPU needed for this phase).

## Machine Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 72 GB | 96+ GB |
| CPU cores | 8 | 16+ (fast single-thread matters most) |
| Disk | 100 GB free | 150 GB free |
| GPU | Not required | Not required |
| Python | 3.10+ | 3.12 |

## Setup

```bash
git clone https://github.com/jamesburton/NanoQuantQwen3.5-397B-a17b.git
cd NanoQuantQwen3.5-397B-a17b

# Install dependencies
pip install torch transformers datasets safetensors tqdm psutil

# Verify installation
python -c "import nanoquant; print('OK')"
```

## What Runs Where

| Phase | GPU? | CPU? | RAM | Time (est.) |
|-------|------|------|-----|-------------|
| Phase 1: Hessian capture | Preferred but optional | Yes (8-10x slower) | ~20 GB | 1.5h (GPU) / 10-15h (CPU) |
| Phase 2: Block reconstruction | No (0.1 GB VRAM) | **Yes — bottleneck** | 50-68 GB, peak ~72 GB at save | 2.5-3h |
| Phase 3: KD refinement | Optional | Yes | ~30 GB | Usually skipped on CPU |
| Eval: PPL computation | Preferred but optional | Yes | ~15 GB | 40 min (GPU) / 3-5h (CPU) |

Phase 2 runs ADMM optimization (N iterations) + STE refinement (200 iterations)
on each of 196 layers per block, across 16 blocks. All CPU matrix operations.

## Recommended Runs

Run **Option B first** (higher bpw), then **Option C** if B shows improvement.

### Option B: Higher bpw (test quality knee-point)

```bash
# Clean start
rm -rf checkpoints/OLMoE-1B-7B-0924-bpw2

# bpw=2.0 → rank=2048 (much more approximation capacity than rank=819)
python scripts/run_stage1.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --output output/OLMoE-1B-7B-0924-bpw2 \
  --bits-per-weight 2.0 \
  --admm-iters 50 \
  --n-calibration 128 \
  --seq-len 2048 \
  --checkpoint-dir checkpoints/OLMoE-1B-7B-0924-bpw2
```

### Option C: More ADMM iterations at bpw=0.80

```bash
rm -rf checkpoints/OLMoE-1B-7B-0924-admm200

python scripts/run_stage1.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --output output/OLMoE-1B-7B-0924-admm200 \
  --bits-per-weight 0.80 \
  --admm-iters 200 \
  --n-calibration 128 \
  --seq-len 2048 \
  --checkpoint-dir checkpoints/OLMoE-1B-7B-0924-admm200
```

## CPU-Only Considerations

### No GPU Detected

The script auto-detects hardware. Without CUDA:
- Hessian capture runs on CPU (~10-15h instead of 1.5h)
- Block reconstruction is unaffected (already CPU-bound)
- Model loads with `device_map="cpu"`

### Float16 Warning

**Never evaluate with float16 on CPU** — it produces NaN due to matmul
overflow. The eval scripts handle this automatically when `--device cpu`,
but if running `eval_ppl.py` directly, always use `--dtype float32`:

```bash
# CORRECT for CPU:
python scripts/eval_ppl.py \
  --model output/OLMoE-1B-7B-0924-bpw2_reconstructed \
  --device cpu \
  --dtype float32 \
  --stride 512 \
  --max-length 2048

# WRONG — will produce NaN on CPU:
# python scripts/eval_ppl.py --model ... --device cpu --dtype float16
```

### MemoryError at Save Step

If the final save fails with MemoryError, use the recovery script:

```bash
python scripts/recover_checkpoint.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --checkpoint-dir checkpoints/OLMoE-1B-7B-0924-bpw2 \
  --output-dir output/OLMoE-1B-7B-0924-bpw2 \
  --rank 2048
```

The `--rank` value must match `bits_per_weight`: `rank = int(bpw * 2048 / 2)`.
- bpw=0.80 → rank=819
- bpw=2.0 → rank=2048

## Running Evaluation

After quantization completes:

```bash
python scripts/run_eval.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --quantized-dir output/OLMoE-1B-7B-0924-bpw2 \
  --results-dir results \
  --device cpu \
  --stride 512 \
  --max-length 2048
```

This writes `results/OLMoE-1B-7B-0924/metrics.json` and updates `results/SUMMARY.md`.

### Quick Activation Sanity Check

Before running the full 3-5h CPU eval, do a fast activation check (~5 min):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Use the _reconstructed dir (created by run_eval.py), or the output dir
# if model.save_pretrained() completed successfully
path = 'output/OLMoE-1B-7B-0924-bpw2_reconstructed'
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                              trust_remote_code=True, device_map='cpu')
model.eval()

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ids = tokenizer(text, return_tensors='pt')['input_ids'][:, :256]

# Expected baseline maxes for reference
baselines = {0: 0.36, 1: 5.6, 2: 47.8, 3: 54.7, 4: 55.3, 5: 55.3,
             6: 55.6, 7: 55.6, 8: 58.8, 9: 65.2, 10: 68.5, 11: 66.3,
             12: 66.3, 13: 66.2, 14: 55.0, 15: 12.6}

for i in range(16):
    def hook(idx):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            mx = o.float().abs().max().item()
            ratio = mx / baselines[idx]
            flag = ' *** PROBLEM' if ratio > 5 or ratio < 0.1 else ''
            print(f'  Block {idx:2d}: max={mx:10.3f}  (baseline={baselines[idx]:6.1f}, ratio={ratio:.2f}x){flag}')
        return fn
    model.model.layers[i].register_forward_hook(hook(i))

with torch.no_grad():
    model(ids)
```

If any block shows ratio >5x or <0.1x, the quantization has quality issues
and full eval will produce unusable PPL.

## Success Criteria

| Metric | Target |
|--------|--------|
| PPL degradation | ≤ 20x (quantized PPL ≤ 132, baseline is 6.62) |
| Forward pass | No NaN/inf in float32 |
| Block activations | All within 3x of baseline max |

## Current Results (bpw=0.80, for reference)

| Metric | Value |
|--------|-------|
| FP16 Baseline PPL | 6.6181 |
| Quantized PPL (partial: blocks 0-10 quant + 11-15 FP16) | 21,425 |
| Quantized PPL (full: 16 blocks, Hessian-fix attempt) | 881,014 |
| Degradation | 3,236x (partial) / 133,000x (full) |
| Target | ≤ 20x |

## Previous Findings (Phi-tiny-MoE, Phase 02)

| Run | Blocks | Degradation | Notes |
|-----|--------|-------------|-------|
| Phi-tiny full (signed-fit) | 32/32 | 30,413x | Same error compounding pattern |
| Phi-tiny 2-block ablation | 2/32 | 2.11x | Per-layer quality is good |

The 2-block ablation proves ADMM per-layer quality is fine — the issue is
multiplicative error compounding across blocks.

## Disk Space Management

| Path | Size | Safe to Delete? |
|------|------|-----------------|
| `checkpoints/*/block_*.pt` | ~1.9 GB each | Yes, after factors file exists |
| `output/*_reconstructed/` | ~13 GB each | Yes, can rebuild from factors |
| `output/*/model-*.safetensors` | ~13 GB total | Yes, if factors file exists |
| `output/*/quantized_factors.safetensors` | ~30 GB | **Keep** — core output |
| `~/.cache/huggingface/` | ~13 GB per model | Can clear if disk-constrained |

## Key Files

| Path | Description |
|------|-------------|
| `scripts/run_stage1.py` | Main quantization entry point |
| `scripts/run_eval.py` | Evaluation runner (baseline + quantized PPL) |
| `scripts/recover_checkpoint.py` | Rebuild factors from block checkpoints |
| `scripts/eval_ppl.py` | Standalone PPL evaluator |
| `nanoquant/quantize.py` | Core quantization loop |
| `nanoquant/reconstruct.py` | Block reconstruction (ADMM + STE) |
| `nanoquant/admm.py` | LB-ADMM binary factorization |
| `.planning/phases/03-scaling-and-evaluation/.continue-here.md` | Full session state |
