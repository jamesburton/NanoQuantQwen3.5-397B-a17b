# External Integrations

**Analysis Date:** 2025-02-21

## APIs & External Services

**Model & Data Hub:**
- Hugging Face Hub - Used for downloading source model weights (Qwen family) and tokenizers.
  - SDK/Client: `transformers`, `huggingface_hub`
  - Auth: `HF_TOKEN` (for private or gated models like Qwen)

**Datasets:**
- Hugging Face Datasets - Source for WikiText-2 calibration and evaluation data.
  - SDK/Client: `datasets`

## Data Storage

**Databases:**
- None detected.

**File Storage:**
- Local filesystem - Used for saving quantized model weights, checkpoints, and logs.
  - Location: `./outputs/`, `./checkpoints/`, `./logs/`

**Caching:**
- Hugging Face Cache - Default local storage for downloaded models and datasets.

## Authentication & Identity

**Auth Provider:**
- Hugging Face Token - Used via CLI (`huggingface-cli login`) or environment variable.

## Monitoring & Observability

**Error Tracking:**
- None detected.

**Logs:**
- Local file logging - Progress and status logged to files (e.g., `logs/stage1-probe.log`).
- Console output - Real-time progress tracking via `tqdm`.

## CI/CD & Deployment

**Hosting:**
- Research/Dev environment - Intended for execution on local hardware or cloud GPU instances (Vast.ai, RunPod, etc.).

**CI Pipeline:**
- None detected.

## Environment Configuration

**Required env vars:**
- `HF_TOKEN` (optional but recommended for gated models).

**Secrets location:**
- Not applicable (managed via environment variables or CLI auth).

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None.

---

*Integration audit: 2025-02-21*
