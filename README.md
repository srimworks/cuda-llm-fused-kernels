# Custom C++/CUDA Fused Kernels for LLM Serving & Vector Retrieval (`sm_75` / `sm_80+`)

**Author:** Alugunulla Vigneswara Kireeti (Vignesh) — *Level 5 Principal AI Systems Architect*  
**Hugging Face Model Hub:** [`kireetialugunulla/VEN-Hermes-14B-AWQ`](https://huggingface.co/kireetialugunulla/VEN-Hermes-14B-AWQ)

---

## Overview

In high-throughput open-weight LLM inference (`Qwen3-14B` / `Llama-3`) and dense vector retrieval (`pgvector` 1024-dim embeddings), PyTorch Eager execution suffers from severe **High-Bandwidth Memory (HBM) round-trip bottlenecks** due to un-fused point-wise and reduction kernels.

This repository provides **two production C++/CUDA kernels written from scratch** with PyBind11 C++ extensions (`torch.utils.cpp_extension`):

1. **`fused_rmsnorm_swiglu_kernel` (`15.87x` Speedup)**:
   - Fuses `RMSNorm(x, weight, eps)` and `SwiGLU(gate, up)` modulation into a **single CUDA kernel launch** with zero intermediate global memory writes.
   - Uses warp-level `__shfl_down_sync(0xffffffff, val, offset)` register reductions and shared SRAM (`__shared__ float shared_warp_sums[32]`) to compute root-mean-square variance across hidden dimensions ($D=4096, 5120$) in a single register pass.
2. **`fused_l2_cosine_similarity_kernel` (`6.73x` Speedup)**:
   - Fuses L2 vector normalization and all-pairs cosine similarity matrix computation $\text{sim}[m, n] = \frac{\langle q_m, k_n \rangle}{\|q_m\|_2 \|k_n\|_2 + \epsilon}$ in a single pass.
   - Utilizes **128-bit vectorized `float4` memory loads** (`reinterpret_cast<const float4*>`) and simultaneous 3-way warp shuffle reductions (`dot_sum`, `q_norm_sq`, `k_norm_sq`).

---

## Empirical Hardware Benchmark (Dual NVIDIA Tesla T4 `sm_75`, CUDA 12.8)

Measured across **300 warmup iterations + 2,000 hardware-timed `torch.cuda.Event` iterations** per tensor shape (`benchmarks/cuda_benchmark_results.json`):

| Custom C++/CUDA Kernel | Tensor Shape & Precision | PyTorch Eager Latency ($\mu\text{s}$) | Custom CUDA Kernel Latency ($\mu\text{s}$) | Speedup Multiplier | Max Absolute Error | Numerical Parity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`fused_rmsnorm_swiglu_kernel`** | $B=32, D=4096$ (`fp16`) | `163.96 us` | **`10.92 us`** | **`15.01x`** | `0.000122` | Verified (`True`) |
| **`fused_rmsnorm_swiglu_kernel`** | $B=64, D=5120$ (`fp16`, 14B Hidden) | `155.75 us` | **`9.82 us`** | **`15.87x`** | `0.000488` | Verified (`True`) |
| **`fused_rmsnorm_swiglu_kernel`** | $B=128, D=5120$ (`fp16`, 14B Hidden) | `213.64 us` | **`27.87 us`** | **`7.67x`** | `0.001953` | Verified (`True`) |
| **`fused_rmsnorm_swiglu_kernel`** | $B=64, D=5120$ (`fp32`) | `109.30 us` | **`20.32 us`** | **`5.38x`** | `0.000002` | Verified (`True`) |
| **`fused_l2_cosine_similarity_kernel`** | $M=32, N=128, D=1024$ (`fp32`) | `122.75 us` | **`18.23 us`** | **`6.73x`** | `0.000000` | Verified (`True`) |

---

## Build & Installation

```bash
pip install torch setuptools ninja
python setup.py build_ext --inplace
```

## License
Apache-2.0
