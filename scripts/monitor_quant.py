#!/usr/bin/env python3
"""Monitor a running NanoQuant quantization process. Checks every 5 minutes."""

import os
import sys
import time
from datetime import datetime

CHECKPOINT_DIR = "output/Phi-tiny-MoE-instruct/checkpoints"
FACTORS_FILE = "output/Phi-tiny-MoE-instruct/quantized_factors.safetensors"
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
                    if mem.rss > 1e9:  # skip wrapper process (~0 RSS)
                        return p.info["pid"], mem.rss / 1024**3, p.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass
    return None


def count_checkpoints():
    """Count completed block checkpoints."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0
    return len([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])


def factors_info():
    """Get factors file size and mtime if it exists."""
    if os.path.exists(FACTORS_FILE):
        stat = os.stat(FACTORS_FILE)
        return stat.st_size / 1024**2, datetime.fromtimestamp(stat.st_mtime)
    return None, None


def main():
    print(f"Monitoring NanoQuant quantization (every {INTERVAL // 60}min)")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Factors: {FACTORS_FILE}")
    print()

    prev_blocks = -1
    start_time = time.time()

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        proc = find_process()
        n_blocks = count_checkpoints()
        fsize, ftime = factors_info()

        if proc:
            pid, rss, threads = proc
            phase = "Phase 1 (Hessians)" if n_blocks == 0 else f"Phase 2 (block {n_blocks}/32)"
            status = f"PID {pid} | {rss:.1f}GB RAM | {threads} threads | {phase}"
        else:
            status = "process not found"

        # Only print on change or every check
        changed = n_blocks != prev_blocks
        marker = " *NEW*" if changed and prev_blocks >= 0 else ""

        print(f"[{now}] {status} | blocks: {n_blocks}/32{marker}")

        if changed and n_blocks > 0 and prev_blocks >= 0:
            elapsed = time.time() - start_time
            avg_per_block = elapsed / n_blocks if n_blocks > 0 else 0
            remaining = (32 - n_blocks) * avg_per_block
            print(f"         ETA: ~{remaining / 3600:.1f}h remaining ({avg_per_block / 60:.1f}min/block)")

        prev_blocks = n_blocks

        # Check if done
        if not proc and n_blocks >= 32:
            print(f"\nQuantization complete! {n_blocks} blocks done.")
            if fsize:
                print(f"Factors: {fsize:.0f} MB (updated {ftime})")
            break

        if not proc and n_blocks < 32:
            print(f"\nWARNING: Process gone but only {n_blocks}/32 blocks done.")
            print("Check for errors or restart.")
            break

        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
