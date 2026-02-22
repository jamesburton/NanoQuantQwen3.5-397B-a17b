# Technology Stack

**Analysis Date:** 2025-02-21

## Languages

**Primary:**
- Python 3.10+ - Used for the entire quantization pipeline and evaluation scripts.

## Runtime

**Environment:**
- PyTorch 2.2.0+ - Primary deep learning runtime for model loading, inference, and quantization.

**Package Manager:**
- pip - Managed via `requirements.txt`.
- Lockfile: missing - No `package-lock.json` or `poetry.lock` detected.

## Frameworks

**Core:**
- Hugging Face Transformers 4.45.0+ - Model architecture, loading, and tokenizer management.
- Hugging Face Accelerate - Multi-GPU/offload orchestration.
- Hugging Face Datasets - Calibration and evaluation data loading (e.g., WikiText-2).

**Testing:**
- Not detected - No explicit testing framework (e.g., pytest) in `requirements.txt`.

**Build/Dev:**
- pip - Core installation tool.

## Key Dependencies

**Critical:**
- `torch` >= 2.2.0 - Core compute engine.
- `transformers` >= 4.45.0 - Model logic and weights handling.
- `datasets` - Calibration data (WikiText-2) source.
- `einops` - Used for specialized tensor manipulations (e.g., in MoE and DeltaNet).

**Infrastructure:**
- `scipy` / `numpy` - Numerical computations for Hessian matrix analysis and ADMM.
- `psutil` - System resource monitoring.
- `tqdm` - Progress monitoring for long-running quantization tasks.

## Configuration

**Environment:**
- Environment variables - Not explicitly used in the core `nanoquant` module, but standard HF variables like `HF_HOME` or `HF_TOKEN` are relevant for model downloads.

**Build:**
- `requirements.txt` - Dependency specification.

## Platform Requirements

**Development:**
- Local GPU/CPU - RTX 3060/4070 (12GB VRAM) or higher.
- Requires aggressive CPU/NVMe offloading via `accelerate` for local model loading.
- ≥256 GB CPU RAM (Recommended 512 GB) for streaming large models.

**Production:**
- Cloud GPU Cluster - 4× to 8× H100 80GB for large-scale quantization (Stage 2/3).
- Platform targets: Vast.ai, RunPod, Lambda Labs, Modal.

---

*Stack analysis: 2025-02-21*
