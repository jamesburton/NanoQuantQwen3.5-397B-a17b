# Coding Conventions

**Analysis Date:** 2026-02-21

## Naming Patterns

**Files:**
- `snake_case`: `admm.py`, `quantize.py`, `run_stage1.py`, `eval_ppl.py`

**Functions:**
- `snake_case`: `svid()`, `lb_admm()`, `quantize_model()`, `evaluate_perplexity()`
- Private functions start with an underscore: `_load_calibration_data()`, `_make_dataloader()`, `_get_blocks()`, `_get_embed_and_norm()`

**Variables:**
- `snake_case`: `W_target`, `residual`, `u1`, `v1`, `U_cont`, `V_bin`

**Types:**
- Explicit type hints are used for parameters and return values.
- `torch.Tensor` is the primary type for data.
- `typing` module is used for complex types: `Tuple`, `Optional`, `Dict`, `Any`.

## Code Style

**Formatting:**
- Indentation: 4 spaces
- Docstrings: Google/Sphinx style (Args, Returns)
- Long lines: Folded or wrapped, though no specific formatter like Black or Prettier is explicitly configured in the repo.

**Linting:**
- Not detected. No `.eslintrc`, `pyproject.toml`, or `.flake8` files.

## Import Organization

**Order:**
1. Standard library: `import os`, `import time`
2. Third-party: `import torch`, `from transformers import ...`, `from datasets import ...`
3. Local: `from nanoquant.hessian import ...`, `from nanoquant.reconstruct import ...`

**Path Aliases:**
- None detected. Uses standard package-relative imports (e.g., `from nanoquant.hessian ...`).

## Error Handling

**Patterns:**
- Explicitly raises errors for invalid states: `raise ValueError("Cannot find transformer blocks in model architecture")`

## Logging

**Framework:**
- Standard `print()` statements for progress and status.
- `tqdm` for progress bars during long-running operations.

**Patterns:**
- Summary statistics and durations are printed to stdout: `print(f"NanoQuant Stage 1")`, `print(f"  Model: {args.model}")`.

## Comments

**When to Comment:**
- Briefly explain specific logic or rare cases: `# Fix zeros (rare but possible)`
- High-level descriptions in module docstrings: `"""Main orchestrator for NanoQuant quantization pipeline."""`

**JSDoc/TSDoc:**
- Not applicable (Python project). Uses Python docstrings.

## Function Design

**Size:**
- Generally focused. Complex logic (like quantization) is broken down into helper functions (`_get_blocks`, `reconstruct_block`, etc.).

**Parameters:**
- Uses a mix of positional and keyword arguments.
- Default values are provided for hyperparameters (`rank=8`, `rho=0.5`).

**Return Values:**
- Explicitly typed. Returns `None` for orchestrators, `torch.Tensor` or `Tuple` for mathematical operations.

## Module Design

**Exports:**
- Explicitly exports functions and classes via `__init__.py`.
- Internal logic is kept private via the `_` prefix.

**Barrel Files:**
- `nanoquant/__init__.py` is used but currently contains minimal exports.

---

*Convention analysis: 2026-02-21*
