#!/usr/bin/env python3
"""Monitor a running NanoQuant quantization process."""

import argparse
import os
import time
from datetime import datetime

INTERVAL = 300  # 5 minutes


def find_process():
    """Find the run_stage1 process and return (pid, rss_gb, n_threads) or None."""
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = p.info.get("cmdline") or []
                if "run_stage1" in " ".join(cmd) and "python" in (p.info.get("name") or "").lower():
                    mem = p.memory_info()
                    if mem.rss > 1e9:
                        return p.info["pid"], mem.rss / 1024**3, p.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass
    return None


def count_checkpoints(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return 0
    return len([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])


def factors_info(factors_file: str):
    if os.path.exists(factors_file):
        stat = os.stat(factors_file)
        return stat.st_size / 1024**2, datetime.fromtimestamp(stat.st_mtime)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Monitor NanoQuant quantization progress")
    parser.add_argument("--checkpoint-dir", default="output/Phi-tiny-MoE-instruct/checkpoints", help="Checkpoint directory to watch")
    parser.add_argument("--factors-file", default="output/Phi-tiny-MoE-instruct/quantized_factors.safetensors", help="Factors file to watch")
    parser.add_argument("--interval", type=int, default=INTERVAL, help="Polling interval in seconds")
    parser.add_argument("--expected-blocks", type=int, default=32, help="Expected number of blocks for completion")
    args = parser.parse_args()

    print(f"Monitoring NanoQuant quantization (every {args.interval // 60}min)")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Factors: {args.factors_file}")
    print()

    prev_blocks = -1
    start_time = time.time()

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        proc = find_process()
        n_blocks = count_checkpoints(args.checkpoint_dir)
        fsize, ftime = factors_info(args.factors_file)

        if proc:
            pid, rss, threads = proc
            phase = "Phase 1 (Hessians)" if n_blocks == 0 else f"Phase 2 (block {n_blocks}/{args.expected_blocks})"
            status = f"PID {pid} | {rss:.1f}GB RAM | {threads} threads | {phase}"
        else:
            status = "process not found"

        changed = n_blocks != prev_blocks
        marker = " *NEW*" if changed and prev_blocks >= 0 else ""
        print(f"[{now}] {status} | blocks: {n_blocks}/{args.expected_blocks}{marker}")

        if changed and n_blocks > 0 and prev_blocks >= 0:
            elapsed = time.time() - start_time
            avg_per_block = elapsed / n_blocks if n_blocks > 0 else 0
            remaining = max(args.expected_blocks - n_blocks, 0) * avg_per_block
            print(f"         ETA: ~{remaining / 3600:.1f}h remaining ({avg_per_block / 60:.1f}min/block)")

        prev_blocks = n_blocks

        if not proc and n_blocks >= args.expected_blocks:
            print(f"\nQuantization complete! {n_blocks} blocks done.")
            if fsize:
                print(f"Factors: {fsize:.0f} MB (updated {ftime})")
            break

        if not proc and n_blocks < args.expected_blocks:
            print(f"\nWARNING: Process gone but only {n_blocks}/{args.expected_blocks} blocks done.")
            print("Check for errors or restart.")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
