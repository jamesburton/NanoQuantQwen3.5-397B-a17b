#!/usr/bin/env python3
"""Standalone WikiText-2 perplexity evaluator using HuggingFace canonical sliding-window method."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def evaluate_perplexity(
    model_path: str,
    device: str = "cuda",
    max_length: int = 2048,
    stride: int = 512,
    dtype: str = "auto",
) -> dict:
    """Compute WikiText-2 test set perplexity using canonical sliding-window method.

    Uses -100 context masking so only newly-predicted tokens contribute to loss,
    matching the HuggingFace recommended approach for comparable PPL numbers.

    Args:
        model_path: HF model name or local path to model directory.
        device: Compute device ("cuda", "cpu", etc.).
        max_length: Maximum context window per evaluation step.
        stride: Number of new tokens to score per step.
        dtype: Model loading dtype - "auto" (model's native dtype) or "float16".

    Returns:
        Dict with keys: ppl, n_tokens, settings.
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()

    print("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"]

    seq_len = input_ids.shape[1]
    print(f"Evaluating on {seq_len} tokens (stride={stride}, max_length={max_length})...")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids_window = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids_window.clone()
            target_ids[:, :-trg_len] = -100  # mask context tokens

            outputs = model(input_ids_window, labels=target_ids)
            neg_log_likelihood = outputs.loss

            # Count non-masked tokens that contributed to loss.
            # model() internally shifts labels by 1, so actual loss tokens =
            # (number of non -100 labels) - 1 (for the shift).
            num_valid = (target_ids != -100).sum().item()
            num_loss_tokens = num_valid - target_ids.size(0)  # subtract batch_size for shift
            nll_sum += neg_log_likelihood.item() * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.tensor(nll_sum / n_tokens)).item()
    print(f"Tokens evaluated: {n_tokens}, Perplexity: {ppl:.4f}")

    return {
        "ppl": ppl,
        "n_tokens": n_tokens,
        "settings": {
            "stride": stride,
            "max_length": max_length,
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="WikiText-2 Perplexity Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF name")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline model for comparison")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max-length", type=int, default=2048, help="Max context window")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16"],
        help="Model dtype: 'auto' (native) or 'float16'",
    )
    args = parser.parse_args()

    result = evaluate_perplexity(
        args.model,
        device=args.device,
        max_length=args.max_length,
        stride=args.stride,
        dtype=args.dtype,
    )
    print(f"\nModel: {args.model}")
    print(f"WikiText-2 Perplexity: {result['ppl']:.2f}")
    print(f"Tokens evaluated: {result['n_tokens']}")

    if args.baseline:
        print(f"\nEvaluating baseline: {args.baseline}")
        baseline_result = evaluate_perplexity(
            args.baseline,
            device=args.device,
            max_length=args.max_length,
            stride=args.stride,
            dtype=args.dtype,
        )
        print(f"\nBaseline: {args.baseline}")
        print(f"Baseline Perplexity: {baseline_result['ppl']:.2f}")
        print(f"Quantized Perplexity: {result['ppl']:.2f}")
        print(f"Degradation: {result['ppl'] / baseline_result['ppl']:.2f}x")


if __name__ == "__main__":
    main()
