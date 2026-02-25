#!/usr/bin/env python3
"""Generate SUMMARY.md from all results/*/metrics.json files."""

import argparse
import glob
import json
import os


def rebuild_summary(results_dir: str = "results") -> str:
    """Read all metrics.json files and generate a comparison SUMMARY.md.

    Args:
        results_dir: Directory containing per-model result subdirectories.

    Returns:
        Path to the generated SUMMARY.md file.
    """
    pattern = os.path.join(results_dir, "*/metrics.json")
    metrics_files = sorted(glob.glob(pattern))

    if not metrics_files:
        print(f"No metrics.json files found in {results_dir}/*/")
        return ""

    rows = []
    for metrics_path in metrics_files:
        with open(metrics_path) as f:
            rows.append(json.load(f))

    lines = [
        "# NanoQuant Scaling Results\n",
        "\n",
        f"*Auto-generated from {len(rows)} model evaluation(s)*\n",
        "\n",
        "| Model | FP16 PPL | Quant PPL | Degradation | FP16 Size | Quant Size | Ratio | Peak VRAM | Quant Time |\n",
        "|-------|----------|-----------|-------------|-----------|------------|-------|-----------|------------|\n",
    ]

    for m in rows:
        ppl = m.get("perplexity", {})
        mem = m.get("model_size", {})
        hw = m.get("hardware", {})
        t = m.get("timing", {})

        name = m.get("model_short_name", m.get("model", "?"))
        fp16_ppl = ppl.get("fp16_baseline")
        quant_ppl = ppl.get("quantized")
        degradation = ppl.get("degradation_ratio")
        fp16_size = mem.get("fp16_gb")
        quant_size = mem.get("quantized_gb")
        ratio = mem.get("compression_ratio")
        peak_vram = hw.get("peak_vram_gb")

        # Quantization time: use total if available, else sum of eval times
        total_quant = t.get("total_quantization_seconds")
        if total_quant is not None:
            time_str = f"{total_quant / 60:.0f} min"
        else:
            time_str = "-"

        def fmt(val, suffix="", decimals=2):
            if val is None:
                return "-"
            return f"{val:.{decimals}f}{suffix}"

        lines.append(
            f"| {name} "
            f"| {fmt(fp16_ppl)} "
            f"| {fmt(quant_ppl)} "
            f"| {fmt(degradation, 'x')} "
            f"| {fmt(fp16_size, ' GB', 1)} "
            f"| {fmt(quant_size, ' GB', 1)} "
            f"| {fmt(ratio, 'x', 1)} "
            f"| {fmt(peak_vram, ' GB', 1)} "
            f"| {time_str} |\n"
        )

    summary_path = os.path.join(results_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.writelines(lines)

    print(f"Generated {summary_path} with {len(rows)} model(s)")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Build SUMMARY.md from metrics.json files")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    args = parser.parse_args()
    rebuild_summary(args.results_dir)


if __name__ == "__main__":
    main()
