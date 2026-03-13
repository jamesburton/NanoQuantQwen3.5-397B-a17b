#!/usr/bin/env python3
"""Stage 1: NanoQuant quantization + evaluation."""

import argparse
import os
import time
import torch
from nanoquant.quantize import quantize_model


def _configure_cpu_threads():
    """Set optimal thread counts for CPU-bound workloads."""
    n_cores = os.cpu_count() or 1
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(min(n_cores, 4))
    # MKL/OpenMP environment (must be set before first BLAS call)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_cores))
    print(f"  CPU threads: {n_cores} (intra-op), {min(n_cores, 4)} (inter-op)")


def evaluate_perplexity(model_path: str, device: str = "cuda") -> float:
    """Evaluate WikiText-2 perplexity."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Use float32 on CPU (float16 matmuls produce NaN on CPU)
    load_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=load_dtype, trust_remote_code=True, device_map=device
    )
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=32768)
    input_ids = encodings["input_ids"].to(device)

    seq_len = input_ids.shape[1]
    max_len = 2048
    nlls = []

    with torch.no_grad():
        for i in range(0, seq_len - 1, max_len):
            end = min(i + max_len, seq_len - 1)
            ids = input_ids[:, i : end + 1]
            outputs = model(input_ids=ids[:, :-1])
            logits = outputs.logits
            targets = ids[:, 1:]

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits = logits.reshape(-1, logits.size(-1))
            shift_labels = targets.reshape(-1)
            nll = loss_fct(shift_logits.float(), shift_labels)
            nlls.append(nll)

    nlls = torch.cat(nlls)
    ppl = torch.exp(nlls.mean()).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(description="NanoQuant Stage 1: Quantize + Evaluate")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
        help="HuggingFace model name or path"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--rank", type=int, default=8, help="ADMM rank")
    parser.add_argument("--gamma", type=float, default=0.2, help="Hessian shrinkage")
    parser.add_argument("--rho", type=float, default=0.5, help="ADMM penalty")
    parser.add_argument("--lam", type=float, default=1e-4, help="Regularization")
    parser.add_argument("--admm-iters", type=int, default=50, help="ADMM iterations")
    parser.add_argument("--tpre", type=int, default=200, help="Pre-tune FP iters")
    parser.add_argument("--tpost", type=int, default=200, help="Post STE iters")
    parser.add_argument("--tglob", type=int, default=1000, help="Global KD iters")
    parser.add_argument("--n-calibration", type=int, default=128, help="Calibration samples")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--max-blocks", type=int, default=None, help="Max blocks to process")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument(
        "--bits-per-weight", type=float, default=None,
        help="Target bits per weight (0.55, 0.80, 1.00). Overrides --rank with computed value. Default: use --rank directly."
    )
    parser.add_argument(
        "--parallel-blocks", type=int, default=1,
        help="Number of blocks to process in parallel (MOE-04 stub, default 1 = sequential)."
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip inline perplexity evaluation after quantization. Use scripts/run_eval.py separately."
    )
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output + "/checkpoints"

    print(f"NanoQuant Stage 1")
    if not torch.cuda.is_available():
        _configure_cpu_threads()
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Rank: {args.rank}")
    if args.bits_per_weight:
        print(f"  Target bpw: {args.bits_per_weight}")
    print(f"  Max blocks: {args.max_blocks or 'all'}")
    print()

    t_start = time.time()

    quantize_model(
        model_name_or_path=args.model,
        output_dir=args.output,
        rank=args.rank,
        gamma=args.gamma,
        rho=args.rho,
        lam=args.lam,
        admm_iters=args.admm_iters,
        tpre=args.tpre,
        tpost=args.tpost,
        tglob=args.tglob,
        n_calibration=args.n_calibration,
        seq_len=args.seq_len,
        checkpoint_dir=args.checkpoint_dir,
        max_blocks=args.max_blocks,
        bits_per_weight=args.bits_per_weight,
    )

    t_quant = time.time() - t_start
    print(f"\nQuantization took {t_quant:.1f}s ({t_quant/60:.1f}min)")

    if not args.skip_eval:
        # Evaluate perplexity using auto-detected device
        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n=== Evaluating WikiText-2 Perplexity ===")
        ppl = evaluate_perplexity(args.output, device=eval_device)
        print(f"Quantized model perplexity: {ppl:.2f}")

        # Perplexity gate: warn if suspiciously high
        if args.max_blocks and args.max_blocks >= 3 and ppl > 100:
            print(
                f"\nWARNING: Perplexity {ppl:.2f} is very high after {args.max_blocks} blocks. "
                "This may indicate quantization quality issues."
            )

        # Try baseline comparison
        try:
            print(f"\nComputing baseline perplexity for {args.model}...")
            baseline_ppl = evaluate_perplexity(args.model, device=eval_device)
            print(f"Baseline perplexity: {baseline_ppl:.2f}")
            print(f"Degradation: {ppl / baseline_ppl:.2f}x")
            if ppl > 5 * baseline_ppl:
                print(
                    f"\nWARNING: Quantized PPL ({ppl:.2f}) is >{5}x baseline ({baseline_ppl:.2f}). "
                    "Consider increasing rank or calibration samples."
                )
        except Exception as e:
            print(f"Could not compute baseline: {e}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
