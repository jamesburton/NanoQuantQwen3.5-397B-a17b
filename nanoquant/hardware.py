"""Hardware auto-detection and adaptive plan for NanoQuant pipeline."""

from dataclasses import dataclass

import psutil
import torch

_GB = 1024 ** 3
_GPU_HEADROOM_BYTES = 1 * _GB   # 1 GB reserved for activations
_CPU_HEADROOM_FRAC = 0.10       # 10% reserved for OS/system
_MIN_GPU_USABLE_BYTES = 2 * _GB  # threshold to use GPU for block computation


@dataclass
class HardwarePlan:
    """Describes detected hardware resources and the derived execution plan."""

    gpu_free_bytes: int
    cpu_free_bytes: int
    disk_free_bytes: int
    cuda_available: bool
    use_gpu_for_blocks: bool
    hessian_on_disk: bool      # False for Phase 1 — Hessian diagonals are tiny
    model_load_strategy: str   # "cpu" | "disk_offload" | "auto_dispatch"
    device: str                # "cuda" or "cpu"


def probe_hardware(offload_dir: str = ".") -> HardwarePlan:
    """Probe available hardware resources and return an adaptive HardwarePlan.

    Args:
        offload_dir: Directory used for disk offloading; disk free space is
                     measured at this path.

    Returns:
        A populated HardwarePlan instance.
    """
    cuda_available = torch.cuda.is_available()

    # ---- GPU memory ----
    if cuda_available:
        try:
            from accelerate.utils import get_max_memory  # type: ignore
            max_mem = get_max_memory()
            # get_max_memory returns bytes keyed by device index (int) and "cpu"
            # Sum all GPU entries (typically just device 0 for consumer machines)
            gpu_raw = sum(
                v for k, v in max_mem.items() if isinstance(k, int)
            )
        except Exception:
            # Fallback: use torch directly
            gpu_raw = torch.cuda.get_device_properties(0).total_memory
        gpu_free_bytes = max(0, gpu_raw - _GPU_HEADROOM_BYTES)
    else:
        gpu_free_bytes = 0

    # ---- CPU memory ----
    vm = psutil.virtual_memory()
    cpu_available = vm.available
    cpu_free_bytes = int(cpu_available * (1.0 - _CPU_HEADROOM_FRAC))

    # ---- Disk ----
    disk_free_bytes = psutil.disk_usage(offload_dir).free

    # ---- Derived plan ----
    use_gpu_for_blocks = cuda_available and (gpu_free_bytes > _MIN_GPU_USABLE_BYTES)
    device = "cuda" if use_gpu_for_blocks else "cpu"

    # Block-by-block manual .to(device) is the correct strategy for quantization.
    # Never load the model with device_map="auto" for reconstruction — it scatters
    # layers across devices and breaks sequential block processing.
    model_load_strategy = "cpu"

    return HardwarePlan(
        gpu_free_bytes=gpu_free_bytes,
        cpu_free_bytes=cpu_free_bytes,
        disk_free_bytes=disk_free_bytes,
        cuda_available=cuda_available,
        use_gpu_for_blocks=use_gpu_for_blocks,
        hessian_on_disk=False,
        model_load_strategy=model_load_strategy,
        device=device,
    )


def print_hardware_summary(plan: HardwarePlan) -> None:
    """Print a one-line human-readable hardware summary.

    Example output:
        Hardware: GPU=10.2GB usable, CPU=28.4GB usable, disk=145.3GB free, strategy=cpu
    """
    gpu_gb = plan.gpu_free_bytes / _GB
    cpu_gb = plan.cpu_free_bytes / _GB
    disk_gb = plan.disk_free_bytes / _GB
    print(
        f"Hardware: GPU={gpu_gb:.1f}GB usable, "
        f"CPU={cpu_gb:.1f}GB usable, "
        f"disk={disk_gb:.1f}GB free, "
        f"strategy={plan.model_load_strategy}"
    )
