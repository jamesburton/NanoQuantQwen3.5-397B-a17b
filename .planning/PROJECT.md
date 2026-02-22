# NanoQuant Expansion Project

## Project Goal
Implement and evaluate NanoQuant quantization across various Mixture-of-Experts (MoE) model sizes, specifically optimized for consumer-grade GPU environments (e.g., 12GB VRAM with aggressive CPU/NVMe offloading).

## Core Strategy
- **Accessibility:** Focus on quantization pipelines that can run on consumer hardware via aggressive memory offloading and block-wise processing.
- **Model Range:** Test against smaller MoE (e.g., Qwen2-MoE-57B) and larger MoE architectures to validate scaling laws.
- **Integration:** Target the most widely used and stable downstream integration (e.g., Hugging Face Transformers/PEFT or vLLM) to ensure immediate usability.

## Success Criteria
- [ ] Functional Stage 1 quantization for Qwen2-MoE models on a 12GB GPU.
- [ ] Perplexity (PPL) evaluation baseline for at least three MoE model sizes.
- [ ] Documented integration guide for at least one major inference engine.
- [ ] Refined ADMM parameters optimized for lower-bit (e.g., sub-bit) quantization on consumer hardware.
