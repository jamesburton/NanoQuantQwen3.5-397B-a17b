# Codebase Structure

## Directory Map
```
.
├── .planning/               # Project management and codebase mapping
│   └── codebase/            # Structured codebase documentation
├── nanoquant/               # Core library implementation
│   ├── admm.py              # LB-ADMM optimization logic
│   ├── hessian.py           # Hessian information capture and hooks
│   ├── moe.py               # MoE-specific layer handling and weight views
│   ├── quantize.py          # Quantization primitives and kernels
│   ├── reconstruct.py       # Block-wise reconstruction logic
│   └── refine.py            # STE-based refinement and KD
├── scripts/                 # Execution and utility scripts
│   ├── eval_ppl.py          # Perplexity evaluation tools
│   └── run_stage1.py        # Entry point for Stage 1 (Hessian + Reconstruction)
├── checkpoints/             # Storage for intermediate model states
├── logs/                    # Execution logs
├── outputs/                 # Quantized model outputs
├── requirements.txt         # Project dependencies
└── README.md                # Project overview and instructions
```

## Key Components
- **`nanoquant/`**: Contains the mathematical core of the quantization process. It is decoupled from specific training loops to allow for modular reuse.
- **`scripts/`**: Orchestrates the core library to perform end-to-end quantization tasks.
- **`WeightView` (in `moe.py`)**: A critical abstraction that provides a unified interface for accessing weights across different architecture types.
