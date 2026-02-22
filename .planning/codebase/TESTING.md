# Testing Patterns

**Analysis Date:** 2026-02-21

## Test Framework

**Runner:**
- None detected. No `pytest`, `unittest`, or `tox` configured.
- Functional tests are run as standalone scripts.

**Assertion Library:**
- None. Relies on `print()` output and `ppl` (perplexity) metrics for quality evaluation.

**Run Commands:**
```bash
python scripts/run_stage1.py --model Qwen/Qwen1.5-MoE-A2.7B --output outputs/stage1-probe # Run quantization pipeline
python scripts/eval_ppl.py --model outputs/stage1-probe # Evaluate output quality
```

## Test File Organization

**Location:**
- Functional and integration tests are in the `scripts/` directory.

**Naming:**
- `run_stage*.py` (e.g., `run_stage1.py`)
- `eval_*.py` (e.g., `eval_ppl.py`)

**Structure:**
```
scripts/
├── eval_ppl.py
└── run_stage1.py
```

## Test Structure

**Suite Organization:**
- No formal suite organization. Each script is a self-contained execution path.

**Patterns:**
- **Setup pattern:** Load model/tokenizer using `transformers.AutoModelForCausalLM`.
- **Execution pattern:** Run quantization or evaluation on a dataset (WikiText-2).
- **Assertion pattern:** Print results to stdout.

## Mocking

**Framework:**
- None detected. Uses real models and datasets.

**Patterns:**
- Not applicable.

**What to Mock:**
- Not applicable.

**What NOT to Mock:**
- LLM weights, attention masks, and calibration data (WikiText-2).

## Fixtures and Factories

**Test Data:**
- WikiText-2 dataset loaded via `datasets.load_dataset()`.

**Location:**
- `nanoquant/quantize.py` (`_load_calibration_data()`)
- `scripts/run_stage1.py` (`evaluate_perplexity()`)
- `scripts/eval_ppl.py` (`evaluate_perplexity()`)

## Coverage

**Requirements:**
- None enforced.

**View Coverage:**
- Not applicable.

## Test Types

**Unit Tests:**
- Not detected.

**Integration Tests:**
- `scripts/run_stage1.py`: Full quantization and basic evaluation on WikiText-2.

**E2E Tests:**
- `scripts/eval_ppl.py`: Standalone quality assessment (perplexity) on quantized models.

## Common Patterns

**Async Testing:**
- Not applicable.

**Error Testing:**
- Not explicitly tested in scripts.

---

*Testing analysis: 2026-02-21*
