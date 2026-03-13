"""Global KD refinement strategies for NanoQuant."""

import math
import time
import types
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nanoquant.reconstruct import reconstruct_weight


@dataclass
class Phase3KDConfig:
    strategy: str = "connected_scales"
    kd_samples: int = 2
    lr: float = 1e-3
    min_improvement_ratio: float = 0.02
    eval_interval: int = 25
    plateau_intervals: int = 2
    max_cache_seconds: float = 120.0
    max_sample_seconds: float = 60.0
    max_phase3_seconds: float = 600.0
    max_layers: int = 64
    layer_subset: str = "experts_only"


@dataclass
class Phase3KDResult:
    applied: bool
    reason: str
    strategy: str
    baseline_kl: float | None = None
    final_kl: float | None = None
    improvement_ratio: float | None = None
    n_iters: int | None = None
    n_scales: int | None = None
    cached_samples: int | None = None
    cache_seconds: float | None = None
    sample_seconds: float | None = None
    layers_considered: int | None = None
    layers_updated: int | None = None
    updated_tensor_kind: str | None = None
    elapsed_seconds: float | None = None
    aborted_by_budget: bool = False

    @classmethod
    def from_dict(cls, strategy: str, data: Dict[str, float | bool | str | int]):
        return cls(
            applied=bool(data.get("applied", False)),
            reason=str(data.get("reason", "unknown")),
            strategy=strategy,
            baseline_kl=(float(data["baseline_kl"]) if data.get("baseline_kl") is not None else None),
            final_kl=(float(data["final_kl"]) if data.get("final_kl") is not None else None),
            improvement_ratio=(float(data["improvement_ratio"]) if data.get("improvement_ratio") is not None else None),
            n_iters=(int(data["n_iters"]) if data.get("n_iters") is not None else None),
            n_scales=(int(data["n_scales"]) if data.get("n_scales") is not None else None),
            cached_samples=(int(data["cached_samples"]) if data.get("cached_samples") is not None else None),
            cache_seconds=(float(data["cache_seconds"]) if data.get("cache_seconds") is not None else None),
            sample_seconds=(float(data["sample_seconds"]) if data.get("sample_seconds") is not None else None),
            layers_considered=(int(data["layers_considered"]) if data.get("layers_considered") is not None else None),
            layers_updated=(int(data["layers_updated"]) if data.get("layers_updated") is not None else None),
            updated_tensor_kind=(str(data["updated_tensor_kind"]) if data.get("updated_tensor_kind") is not None else None),
            elapsed_seconds=(float(data["elapsed_seconds"]) if data.get("elapsed_seconds") is not None else None),
            aborted_by_budget=bool(data.get("aborted_by_budget", False)),
        )


class _LinearOutputScaler:
    """Attach learnable output scales to linear modules via forward hooks."""

    def __init__(self, model_quant: nn.Module, device: str):
        self.model_quant = model_quant
        self.device = torch.device(device)
        self.entries: List[Tuple[str, nn.Linear, nn.Parameter, torch.utils.hooks.RemovableHandle]] = []
        self._attach()

    def _attach(self) -> None:
        for name, module in self.model_quant.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(token in name for token in ("layers.", "mlp", "self_attn", "experts", "lm_head")):
                continue
            scale = nn.Parameter(torch.ones(1, device=self.device, dtype=torch.float32))

            def _hook(mod, inp, out, scale_ref=scale):
                scale_t = scale_ref.to(out[0].device if isinstance(out, tuple) else out.device)
                scale_t = scale_t.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
                if isinstance(out, tuple):
                    return (out[0] * scale_t, *out[1:])
                return out * scale_t

            handle = module.register_forward_hook(_hook)
            self.entries.append((name, module, scale, handle))

    @property
    def params(self) -> List[nn.Parameter]:
        return [entry[2] for entry in self.entries]

    def remove_hooks(self) -> None:
        for _, _, _, handle in self.entries:
            handle.remove()

    def fold_into_weights(self) -> None:
        with torch.no_grad():
            for _, module, scale, _ in self.entries:
                factor = scale.detach().to(module.weight.device, dtype=module.weight.dtype)
                module.weight.mul_(factor)
                if module.bias is not None:
                    module.bias.mul_(factor)


@dataclass
class _SlimMoEOverrideEntry:
    layer_name: str
    module: nn.Linear
    original_forward: object
    bias: torch.Tensor | None
    U_bin: torch.Tensor
    V_bin: torch.Tensor
    d_in: torch.Tensor
    d_out: torch.Tensor
    s1_param: nn.Parameter
    s2_param: nn.Parameter
    original_s1: torch.Tensor
    original_s2: torch.Tensor
    U_lat: nn.Parameter | None = None
    V_lat: nn.Parameter | None = None

    def current_weight(self) -> torch.Tensor:
        if self.U_lat is None or self.V_lat is None:
            U = self.U_bin
            V = self.V_bin
        else:
            U = self.U_lat + (torch.sign(self.U_lat) - self.U_lat).detach()
            V = self.V_lat + (torch.sign(self.V_lat) - self.V_lat).detach()
        return reconstruct_weight(U, V, self.s1_param, self.s2_param, self.d_in, self.d_out)


class _SlimMoEFactorOverride:
    """Connected override for SlimMoE expert w1/w2/w3 modules."""

    def __init__(
        self,
        model_quant: nn.Module,
        all_quantized: Dict[str, Dict[str, torch.Tensor]],
        device: str,
        config: Phase3KDConfig,
        use_latents: bool,
    ):
        self.model_quant = model_quant
        self.all_quantized = all_quantized
        self.device = torch.device(device)
        self.config = config
        self.use_latents = use_latents
        self.entries: List[_SlimMoEOverrideEntry] = []
        self._attach()

    def _is_selected_layer(self, layer_name: str) -> bool:
        if self.config.layer_subset == "experts_only":
            return ".block_sparse_moe.experts." in layer_name
        if self.config.layer_subset == "experts_and_mlp":
            return ".block_sparse_moe." in layer_name
        return True

    def _attach(self) -> None:
        module_map = dict(self.model_quant.named_modules())
        selected = 0
        for layer_name, factors in self.all_quantized.items():
            if not self._is_selected_layer(layer_name):
                continue
            module = module_map.get(layer_name)
            if not isinstance(module, nn.Linear):
                continue
            if not all(k in factors for k in ("U_bin", "V_bin", "s1", "s2", "d_in", "d_out")):
                continue
            s1 = factors["s1"].float().to(self.device)
            s2 = factors["s2"].float().to(self.device)
            entry = _SlimMoEOverrideEntry(
                layer_name=layer_name,
                module=module,
                original_forward=module.forward,
                bias=(module.bias.detach().to(self.device) if module.bias is not None else None),
                U_bin=factors["U_bin"].float().to(self.device),
                V_bin=factors["V_bin"].float().to(self.device),
                d_in=factors["d_in"].float().to(self.device),
                d_out=factors["d_out"].float().to(self.device),
                s1_param=nn.Parameter(s1.clone()),
                s2_param=nn.Parameter(s2.clone()),
                original_s1=s1.clone().cpu(),
                original_s2=s2.clone().cpu(),
                U_lat=(nn.Parameter(factors["U_bin"].float().to(self.device).clone()) if self.use_latents else None),
                V_lat=(nn.Parameter(factors["V_bin"].float().to(self.device).clone()) if self.use_latents else None),
            )

            def _forward(mod, input, entry_ref=entry):
                W = entry_ref.current_weight()
                W = W.to(device=input.device, dtype=input.dtype)
                bias = entry_ref.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input.dtype)
                return F.linear(input, W, bias)

            module.forward = types.MethodType(_forward, module)
            self.entries.append(entry)
            selected += 1
            if selected >= self.config.max_layers:
                break

    @property
    def params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for entry in self.entries:
            params.extend([entry.s1_param, entry.s2_param])
            if entry.U_lat is not None and entry.V_lat is not None:
                params.extend([entry.U_lat, entry.V_lat])
        return params

    def restore(self) -> None:
        for entry in self.entries:
            entry.module.forward = entry.original_forward

    def commit(self) -> None:
        with torch.no_grad():
            for entry in self.entries:
                factors = self.all_quantized[entry.layer_name]
                factors["s1"] = entry.s1_param.detach().cpu()
                factors["s2"] = entry.s2_param.detach().cpu()
                if entry.U_lat is not None and entry.V_lat is not None:
                    U = torch.sign(entry.U_lat.detach())
                    V = torch.sign(entry.V_lat.detach())
                    U[U == 0] = 1.0
                    V[V == 0] = 1.0
                    factors["U_bin"] = U.cpu()
                    factors["V_bin"] = V.cpu()

    def rollback(self) -> None:
        with torch.no_grad():
            for entry in self.entries:
                entry.s1_param.copy_(entry.original_s1.to(self.device))
                entry.s2_param.copy_(entry.original_s2.to(self.device))


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _cache_kd_data(model_fp: nn.Module, dataloader, max_cache_seconds: float | None = None, max_sample_seconds: float | None = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, float | bool | str | int]]:
    fp_device = _module_device(model_fp)
    fp_logits_cache: List[torch.Tensor] = []
    input_ids_cache: List[torch.Tensor] = []
    cache_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching logits for KD")):
            sample_start = time.time()
            if isinstance(batch, dict):
                ids = batch["input_ids"]
            elif isinstance(batch, (list, tuple)):
                ids = batch[0]
            else:
                ids = batch
            input_ids_cache.append(ids.cpu())
            fp_out = model_fp(input_ids=ids.to(fp_device)).logits
            fp_logits_cache.append(fp_out.cpu())
            elapsed = time.time() - cache_start
            sample_elapsed = time.time() - sample_start
            if max_sample_seconds is not None and sample_elapsed > max_sample_seconds:
                return input_ids_cache, fp_logits_cache, {
                    "aborted": True,
                    "reason": "sample_too_slow",
                    "cached_samples": batch_idx + 1,
                    "cache_seconds": float(elapsed),
                    "sample_seconds": float(sample_elapsed),
                }
            if max_cache_seconds is not None and elapsed > max_cache_seconds:
                return input_ids_cache, fp_logits_cache, {
                    "aborted": True,
                    "reason": "cache_too_slow",
                    "cached_samples": batch_idx + 1,
                    "cache_seconds": float(elapsed),
                }

    return input_ids_cache, fp_logits_cache, {
        "aborted": False,
        "reason": "ok",
        "cached_samples": len(input_ids_cache),
        "cache_seconds": float(time.time() - cache_start),
    }


def _compute_avg_kl(model_quant: nn.Module, input_ids_cache: List[torch.Tensor], fp_logits_cache: List[torch.Tensor], quant_device: torch.device) -> float:
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in range(len(input_ids_cache)):
            ids = input_ids_cache[idx].to(quant_device)
            fp_logits = fp_logits_cache[idx].to(quant_device)
            quant_logits = model_quant(input_ids=ids).logits
            fp_probs = F.log_softmax(fp_logits, dim=-1)
            quant_log_probs = F.log_softmax(quant_logits, dim=-1)
            loss = F.kl_div(quant_log_probs, fp_probs.exp(), reduction="batchmean", log_target=False)
            if torch.isfinite(loss):
                total_loss += loss.item()
                count += 1
    if count == 0:
        return math.inf
    return total_loss / count


def tune_scales_kd(
    model_quant: nn.Module,
    model_fp: nn.Module,
    dataloader,
    n_iters: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda",
    min_improvement_ratio: float = 0.02,
    eval_interval: int = 100,
    plateau_intervals: int = 3,
    max_cache_seconds: float = 240.0,
    max_sample_seconds: float = 60.0,
) -> Dict[str, float | bool | str | int]:
    model_fp.eval()
    model_quant.eval()
    quant_device = torch.device(device)
    scaler = _LinearOutputScaler(model_quant, device=device)
    if not scaler.params:
        return {"applied": False, "reason": "no_linear_modules"}

    input_ids_cache, fp_logits_cache, cache_stats = _cache_kd_data(
        model_fp,
        dataloader,
        max_cache_seconds=max_cache_seconds,
        max_sample_seconds=max_sample_seconds,
    )
    if cache_stats.get("aborted"):
        scaler.remove_hooks()
        return {"applied": False, "reason": cache_stats.get("reason"), **cache_stats}

    baseline_kl = _compute_avg_kl(model_quant, input_ids_cache, fp_logits_cache, quant_device)
    optimizer = torch.optim.Adam(scaler.params, lr=lr)
    best_kl = baseline_kl
    best_scales = [p.detach().clone() for p in scaler.params]
    stale_intervals = 0
    completed_iters = 0

    for iteration in range(n_iters):
        finite_steps = 0
        for idx in range(len(input_ids_cache)):
            ids = input_ids_cache[idx].to(quant_device)
            fp_logits = fp_logits_cache[idx].to(quant_device)
            optimizer.zero_grad()
            quant_logits = model_quant(input_ids=ids).logits
            fp_probs = F.log_softmax(fp_logits, dim=-1)
            quant_log_probs = F.log_softmax(quant_logits, dim=-1)
            loss = F.kl_div(quant_log_probs, fp_probs.exp(), reduction="batchmean", log_target=False)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            finite_steps += 1

        completed_iters = iteration + 1
        if finite_steps == 0:
            break
        if completed_iters % eval_interval == 0 or completed_iters == n_iters:
            avg_kl = _compute_avg_kl(model_quant, input_ids_cache, fp_logits_cache, quant_device)
            print(f"  KD iter {completed_iters}/{n_iters}, avg KL loss: {avg_kl:.6f}")
            if math.isfinite(avg_kl) and avg_kl < best_kl:
                best_kl = avg_kl
                best_scales = [p.detach().clone() for p in scaler.params]
                stale_intervals = 0
            else:
                stale_intervals += 1
                if stale_intervals >= plateau_intervals:
                    break

    with torch.no_grad():
        for param, best in zip(scaler.params, best_scales):
            param.copy_(best)
    final_kl = _compute_avg_kl(model_quant, input_ids_cache, fp_logits_cache, quant_device)
    if not math.isfinite(final_kl):
        final_kl = best_kl
    improvement = baseline_kl - final_kl if math.isfinite(baseline_kl) and math.isfinite(final_kl) else float("-inf")
    improvement_ratio = (improvement / baseline_kl) if math.isfinite(baseline_kl) and baseline_kl > 0 else 0.0
    applied = math.isfinite(final_kl) and improvement_ratio >= min_improvement_ratio
    if applied:
        scaler.fold_into_weights()
    scaler.remove_hooks()
    return {
        "applied": applied,
        "reason": "ok" if applied else "insufficient_improvement",
        "baseline_kl": float(baseline_kl),
        "final_kl": float(final_kl),
        "improvement_ratio": float(improvement_ratio),
        "n_iters": completed_iters,
        "n_scales": len(scaler.params),
        "cached_samples": cache_stats.get("cached_samples"),
        "cache_seconds": cache_stats.get("cache_seconds"),
        "sample_seconds": cache_stats.get("sample_seconds"),
    }


def _tune_slimmoe_factor_strategy(
    model_quant: nn.Module,
    model_fp: nn.Module,
    dataloader,
    all_quantized: Dict[str, Dict[str, torch.Tensor]],
    config: Phase3KDConfig,
    n_iters: int,
    device: str,
    use_latents: bool,
    updated_tensor_kind: str,
) -> Dict[str, float | bool | str | int]:
    model_fp.eval()
    model_quant.eval()
    quant_device = torch.device(device)
    start_time = time.time()
    override = _SlimMoEFactorOverride(model_quant, all_quantized, device=device, config=config, use_latents=use_latents)
    if not override.entries:
        return {
            "applied": False,
            "reason": "no_slimmoe_expert_layers",
            "layers_considered": 0,
            "layers_updated": 0,
            "updated_tensor_kind": updated_tensor_kind,
            "elapsed_seconds": 0.0,
        }

    input_ids_cache, fp_logits_cache, cache_stats = _cache_kd_data(
        model_fp,
        dataloader,
        max_cache_seconds=config.max_cache_seconds,
        max_sample_seconds=config.max_sample_seconds,
    )
    if cache_stats.get("aborted"):
        override.restore()
        return {
            "applied": False,
            "reason": str(cache_stats.get("reason")),
            "layers_considered": len(override.entries),
            "layers_updated": 0,
            "updated_tensor_kind": updated_tensor_kind,
            "elapsed_seconds": float(time.time() - start_time),
            "aborted_by_budget": True,
            **cache_stats,
        }

    baseline_kl = _compute_avg_kl(model_quant, input_ids_cache, fp_logits_cache, quant_device)
    optimizer = torch.optim.Adam(override.params, lr=config.lr)
    best_kl = baseline_kl
    best_state = []
    for entry in override.entries:
        state = [entry.s1_param.detach().clone(), entry.s2_param.detach().clone()]
        if entry.U_lat is not None and entry.V_lat is not None:
            state.extend([entry.U_lat.detach().clone(), entry.V_lat.detach().clone()])
        best_state.append(state)
    stale_intervals = 0
    completed_iters = 0
    aborted_by_budget = False

    for iteration in range(n_iters):
        if time.time() - start_time > config.max_phase3_seconds:
            aborted_by_budget = True
            break
        finite_steps = 0
        for idx in range(len(input_ids_cache)):
            ids = input_ids_cache[idx].to(quant_device)
            fp_logits = fp_logits_cache[idx].to(quant_device)
            optimizer.zero_grad()
            quant_logits = model_quant(input_ids=ids).logits
            fp_probs = F.log_softmax(fp_logits, dim=-1)
            quant_log_probs = F.log_softmax(quant_logits, dim=-1)
            loss = F.kl_div(quant_log_probs, fp_probs.exp(), reduction="batchmean", log_target=False)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            finite_steps += 1

        completed_iters = iteration + 1
        if finite_steps == 0:
            break
        if completed_iters % config.eval_interval == 0 or completed_iters == n_iters:
            avg_kl = _compute_avg_kl(model_quant, input_ids_cache, fp_logits_cache, quant_device)
            label = "Latent factor KD" if use_latents else "Factor-scale KD"
            print(f"  {label} iter {completed_iters}/{n_iters}, avg KL loss: {avg_kl:.6f}")
            if math.isfinite(avg_kl) and avg_kl < best_kl:
                best_kl = avg_kl
                best_state = []
                for entry in override.entries:
                    state = [entry.s1_param.detach().clone(), entry.s2_param.detach().clone()]
                    if entry.U_lat is not None and entry.V_lat is not None:
                        state.extend([entry.U_lat.detach().clone(), entry.V_lat.detach().clone()])
                    best_state.append(state)
                stale_intervals = 0
            else:
                stale_intervals += 1
                if stale_intervals >= config.plateau_intervals:
                    break

    with torch.no_grad():
        for entry, state in zip(override.entries, best_state):
            entry.s1_param.copy_(state[0])
            entry.s2_param.copy_(state[1])
            if entry.U_lat is not None and entry.V_lat is not None and len(state) == 4:
                entry.U_lat.copy_(state[2])
                entry.V_lat.copy_(state[3])

    final_kl = _compute_avg_kl(model_quant, input_ids_cache, fp_logits_cache, quant_device)
    if not math.isfinite(final_kl):
        final_kl = best_kl
    improvement = baseline_kl - final_kl if math.isfinite(baseline_kl) and math.isfinite(final_kl) else float("-inf")
    improvement_ratio = (improvement / baseline_kl) if math.isfinite(baseline_kl) and baseline_kl > 0 else 0.0
    applied = math.isfinite(final_kl) and improvement_ratio >= config.min_improvement_ratio and not aborted_by_budget

    if applied:
        override.commit()
        reason = "ok"
        layers_updated = len(override.entries)
    else:
        override.rollback()
        reason = "budget_exceeded" if aborted_by_budget else "insufficient_improvement"
        layers_updated = 0
    override.restore()

    return {
        "applied": applied,
        "reason": reason,
        "baseline_kl": float(baseline_kl),
        "final_kl": float(final_kl),
        "improvement_ratio": float(improvement_ratio),
        "n_iters": completed_iters,
        "cached_samples": cache_stats.get("cached_samples"),
        "cache_seconds": cache_stats.get("cache_seconds"),
        "sample_seconds": cache_stats.get("sample_seconds"),
        "layers_considered": len(override.entries),
        "layers_updated": layers_updated,
        "updated_tensor_kind": updated_tensor_kind,
        "elapsed_seconds": float(time.time() - start_time),
        "aborted_by_budget": aborted_by_budget,
    }


def run_phase3_kd(
    model_quant: nn.Module,
    model_name_or_path: str,
    cal_data: torch.Tensor,
    n_iters: int,
    device: str,
    config: Phase3KDConfig,
    all_quantized: Dict[str, Dict[str, torch.Tensor]] | None = None,
) -> Phase3KDResult:
    from transformers import AutoModelForCausalLM

    if config.strategy not in ("connected_scales", "factor_scales", "factor_latents"):
        return Phase3KDResult(applied=False, reason="unsupported_strategy", strategy=config.strategy)

    model_fp = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model_fp.eval()
    for p in model_fp.parameters():
        p.requires_grad_(False)

    if device == "cuda":
        model_quant.to(device)

    try:
        kd_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(cal_data[: config.kd_samples]),
            batch_size=1,
            shuffle=False,
        )
        if config.strategy == "connected_scales":
            stats = tune_scales_kd(
                model_quant,
                model_fp,
                kd_loader,
                n_iters=n_iters,
                lr=config.lr,
                device=device,
                min_improvement_ratio=config.min_improvement_ratio,
                eval_interval=config.eval_interval,
                plateau_intervals=config.plateau_intervals,
                max_cache_seconds=config.max_cache_seconds,
                max_sample_seconds=config.max_sample_seconds,
            )
        else:
            if all_quantized is None:
                return Phase3KDResult(applied=False, reason="missing_factor_store", strategy=config.strategy)
            stats = _tune_slimmoe_factor_strategy(
                model_quant,
                model_fp,
                kd_loader,
                all_quantized=all_quantized,
                config=config,
                n_iters=n_iters,
                device=device,
                use_latents=(config.strategy == "factor_latents"),
                updated_tensor_kind=("u_v_s1_s2" if config.strategy == "factor_latents" else "s1_s2"),
            )
        return Phase3KDResult.from_dict(config.strategy, stats)
    finally:
        if device == "cuda":
            model_quant.cpu()
        model_fp.cpu()
        del model_fp
        torch.cuda.empty_cache()
