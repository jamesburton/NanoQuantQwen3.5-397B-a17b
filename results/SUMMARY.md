# NanoQuant Scaling Results

*Auto-generated from 3 model evaluation(s)*

| Model | FP16 PPL | Quant PPL | Degradation | FP16 Size | Quant Size | Ratio | Peak VRAM | Quant Time |
|-------|----------|-----------|-------------|-----------|------------|-------|-----------|------------|
| OLMoE-1B-7B-0924 | 6.62 | unusable | 3,236x (partial) / 133,000x (full) | 12.9 GB | 30.1 GB | 0.4x | 11.2 GB | - |
| Phi-tiny-MoE-instruct | 7.17 | 218212.22 | 30413.96x | 7.0 GB | 0.8 GB | 9.0x | 8.1 GB | - |
| Qwen1.5-MoE-A2.7B | - | - | - | 26.7 GB | 11.3 GB | 2.4x | - | - |

## Notes

### OLMoE-1B-7B-0924 (bpw=0.80, rank=819, 50 ADMM iters)

Pipeline smoke test **PASSED** — end-to-end flow works mechanically. Quality test **FAILED**.

Two Hessian approaches tested:
- **Pre-captured Hessians**: activation explosion starting block 2 (7.8x to 314x by block 10). PPL with 10 quantized + 6 FP16 blocks: 21,425 (3,236x degradation).
- **Actual-layer-input Hessians** (attempted fix, reverted): activation collapse (50x below baseline). PPL all 16 blocks: 881,014.

Root cause: binary factorization at bpw=0.80 lacks approximation capacity for OLMoE weight matrices. Error compounds multiplicatively through blocks. Sub-1-bit may need larger models with more redundancy, higher bpw (2.0-4.0), or significantly more ADMM iterations.
