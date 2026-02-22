# Codebase Architecture

## High-Level Overview
NanoQuant is a specialized quantization framework designed for large-scale Mixture-of-Experts (MoE) models, specifically optimized for the Qwen3.5-397B-A17B architecture. The system employs a multi-phase quantization pipeline to achieve high compression with minimal perplexity degradation.

## Pipeline Phases
1. **Hessian Capture:** Uses PyTorch forward hooks to collect second-order information (Hessian diagonals) during a calibration pass. This data informs the importance of specific weights during quantization.
2. **Block-wise Reconstruction (LB-ADMM):** Implements Low-Bit Alternating Direction Method of Multipliers. This phase performs local optimization on model blocks to minimize reconstruction error.
3. **Refinement (STE):** Uses Straight-Through Estimators for fine-tuning quantized weights, allowing gradient flow through non-differentiable quantization functions.
4. **Global Scale Refinement:** Knowledge Distillation (KD) is used in later stages to align the quantized model's outputs with the original FP16/BF16 teacher model.

## Design Patterns
- **Orchestrator Pattern:** The quantization pipeline is managed by central controllers that sequence data collection, optimization, and refinement.
- **Hook-based Data Collection:** Extensive use of PyTorch hooks for non-intrusive metadata collection during model execution.
- **Proxy/View Pattern:** The `WeightView` abstraction allows the system to treat standard linear layers and fused MoE expert tensors uniformly, simplifying the optimization logic.
