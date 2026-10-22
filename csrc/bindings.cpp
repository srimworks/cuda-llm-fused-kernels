/*
 * PyBind11 C++ Extension Bindings for VEN-Hermes Level 5 Custom CUDA Kernels
 * Author: Vignesh (VigneshCEO / SRIM Works)
 */

#include <torch/extension.h>

// Forward declarations of CUDA host launchers in ven_cuda_kernels.cu
torch::Tensor fused_rmsnorm_swiglu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor gate,
    torch::Tensor up,
    float eps
);

torch::Tensor fused_l2_cosine_similarity_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    float eps
);

torch::Tensor fused_rmsnorm_swiglu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor gate,
    torch::Tensor up,
    float eps = 1e-6f
) {
    return fused_rmsnorm_swiglu_cuda(x, weight, gate, up, eps);
}

torch::Tensor fused_l2_cosine_similarity(
    torch::Tensor queries,
    torch::Tensor keys,
    float eps = 1e-12f
) {
    return fused_l2_cosine_similarity_cuda(queries, keys, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VEN-Hermes Level 5 Custom C++/CUDA Kernel Extension (sm_75 / sm_80+)";
    m.def(
        "fused_rmsnorm_swiglu",
        &fused_rmsnorm_swiglu,
        "Fused RMSNorm + SwiGLU Activation CUDA Kernel (warp shuffle reduced)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("gate"),
        py::arg("up"),
        py::arg("eps") = 1e-6f
    );
    m.def(
        "fused_l2_cosine_similarity",
        &fused_l2_cosine_similarity,
        "Fused L2 Normalization + Cosine Similarity Matrix CUDA Kernel (float4 vectorized + warp shuffle)",
        py::arg("queries"),
        py::arg("keys"),
        py::arg("eps") = 1e-12f
    );
}
