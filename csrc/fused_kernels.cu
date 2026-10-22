/*
 * VEN-Hermes Level 5 Custom C++ / CUDA Kernels (sm_75 Tesla T4 & sm_80+)
 * Author: Vignesh (VigneshCEO / SRIM Works)
 *
 * Implements two production CUDA kernels from scratch:
 *   1. fused_rmsnorm_swiglu_kernel:
 *      Fuses RMSNorm(x, weight, eps) and SwiGLU(gate, up) + residual modulation
 *      using warp-level __shfl_down_sync reductions to eliminate global HBM round-trips.
 *   2. fused_l2_cosine_similarity_kernel:
 *      Fuses L2 vector normalization and batched cosine similarity matrix computation
 *      for VEN's 1024-dim Company & Evidence pgvector embeddings.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256

// -----------------------------------------------------------------------------
// Device Helper: Warp-level & Block-level Sum Reduction using __shfl_down_sync
// -----------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* shared_warp_sums) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared_warp_sums[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared_warp_sums[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// -----------------------------------------------------------------------------
// Kernel 1: Fused RMSNorm + SwiGLU Activation Kernel
// Given:
//   x      : [B, D] input hidden state
//   weight : [D]    RMSNorm scale parameter
//   gate   : [B, D] SwiGLU gate projection
//   up     : [B, D] SwiGLU up projection
// Computes in a single kernel launch without intermediate global memory writes:
//   rms_x_i = (x_i * rsqrt(mean(x^2) + eps)) * weight_i
//   swiglu_i = (gate_i / (1.0f + expf(-gate_i))) * up_i
//   out_i = rms_x_i * swiglu_i
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_rmsnorm_swiglu_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    const int D,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t* x_row = x + row * D;
    const scalar_t* gate_row = gate + row * D;
    const scalar_t* up_row = up + row * D;
    scalar_t* out_row = out + row * D;

    __shared__ float shared_warp_sums[WARP_SIZE];
    __shared__ float s_inv_rms;

    // Pass 1: Accumulate sum of squares across row in registers
    float sum_sq = 0.0f;
    for (int col = tid; col < D; col += stride) {
        float v = static_cast<float>(x_row[col]);
        sum_sq += v * v;
    }

    // Block-wide reduction via __shfl_down_sync
    sum_sq = block_reduce_sum(sum_sq, shared_warp_sums);

    if (tid == 0) {
        s_inv_rms = rsqrtf((sum_sq / static_cast<float>(D)) + eps);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;

    // Pass 2: Fused RMSNorm scaling + SiLU(gate) * up in register pipeline
    for (int col = tid; col < D; col += stride) {
        float x_val = static_cast<float>(x_row[col]);
        float w_val = static_cast<float>(weight[col]);
        float g_val = static_cast<float>(gate_row[col]);
        float u_val = static_cast<float>(up_row[col]);

        float normed = (x_val * inv_rms) * w_val;
        float silu_g = g_val / (1.0f + __expf(-g_val));
        float swiglu = silu_g * u_val;

        out_row[col] = static_cast<scalar_t>(normed * swiglu);
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Fused L2 Normalization + Cosine Similarity Matrix Kernel
// For VEN Marketing Agent's 1024-dim Company & Evidence embeddings:
// Given:
//   queries : [M, D] unnormalized query embeddings
//   keys    : [N, D] unnormalized key/evidence embeddings
// Computes:
//   sim[m, n] = dot(q_m, k_n) / (||q_m||_2 * ||k_n||_2 + eps)
// Each warp computes one (m, n) cosine similarity cell using warp shuffle
// reductions over D dimensions, fusing L2 norm + dot product in a single pass!
// -----------------------------------------------------------------------------
__global__ void fused_l2_cosine_similarity_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float* __restrict__ sim_out,
    const int M,
    const int N,
    const int D,
    const float eps
) {
    // Each block has (blockDim.x / 32) warps; each warp handles one pair (m, n)
    const int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int total_pairs = M * N;

    if (global_warp_id >= total_pairs) {
        return;
    }

    const int m = global_warp_id / N;
    const int n = global_warp_id % N;

    const float* q_ptr = queries + m * D;
    const float* k_ptr = keys + n * D;

    float dot_sum = 0.0f;
    float q_norm_sq = 0.0f;
    float k_norm_sq = 0.0f;

    // Vectorized float4 load when D is divisible by 4 (e.g. D=1024)
    if (D % 4 == 0) {
        const int D4 = D / 4;
        const float4* q4_ptr = reinterpret_cast<const float4*>(q_ptr);
        const float4* k4_ptr = reinterpret_cast<const float4*>(k_ptr);

        for (int idx = lane; idx < D4; idx += WARP_SIZE) {
            float4 q4 = q4_ptr[idx];
            float4 k4 = k4_ptr[idx];

            dot_sum += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            q_norm_sq += q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
            k_norm_sq += k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
        }
    } else {
        for (int col = lane; col < D; col += WARP_SIZE) {
            float q_val = q_ptr[col];
            float k_val = k_ptr[col];
            dot_sum += q_val * k_val;
            q_norm_sq += q_val * q_val;
            k_norm_sq += k_val * k_val;
        }
    }

    // Warp-level reductions for dot, ||q||^2, and ||k||^2 simultaneously
    dot_sum   = warp_reduce_sum(dot_sum);
    q_norm_sq = warp_reduce_sum(q_norm_sq);
    k_norm_sq = warp_reduce_sum(k_norm_sq);

    if (lane == 0) {
        float inv_q = rsqrtf(q_norm_sq + eps);
        float inv_k = rsqrtf(k_norm_sq + eps);
        sim_out[m * N + n] = dot_sum * inv_q * inv_k;
    }
}

// -----------------------------------------------------------------------------
// Host C++ Launchers Called by PyBind11
// -----------------------------------------------------------------------------
torch::Tensor fused_rmsnorm_swiglu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor gate,
    torch::Tensor up,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");

    x = x.contiguous();
    weight = weight.contiguous();
    gate = gate.contiguous();
    up = up.contiguous();

    const int D = x.size(-1);
    const int B = x.numel() / D;
    auto out = torch::empty_like(x);

    const int threads = MAX_THREADS_PER_BLOCK;
    const dim3 blocks(B);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_rmsnorm_swiglu_cuda", ([&] {
        fused_rmsnorm_swiglu_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            up.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            D,
            eps
        );
    }));

    return out;
}

torch::Tensor fused_l2_cosine_similarity_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    float eps
) {
    TORCH_CHECK(queries.is_cuda(), "queries must be a CUDA tensor");
    TORCH_CHECK(keys.is_cuda(), "keys must be a CUDA tensor");
    TORCH_CHECK(queries.dim() == 2 && keys.dim() == 2, "queries and keys must be 2D [M, D] and [N, D]");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Embedding dimension D must match");

    queries = queries.to(torch::kFloat32).contiguous();
    keys = keys.to(torch::kFloat32).contiguous();

    const int M = queries.size(0);
    const int D = queries.size(1);
    const int N = keys.size(0);

    auto sim_out = torch::empty({M, N}, queries.options());

    const int total_pairs = M * N;
    const int warps_per_block = MAX_THREADS_PER_BLOCK / WARP_SIZE; // 8 warps per block
    const int num_blocks = (total_pairs + warps_per_block - 1) / warps_per_block;

    fused_l2_cosine_similarity_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        sim_out.data_ptr<float>(),
        M,
        N,
        D,
        eps
    );

    return sim_out;
}
