from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_cuda_ops",
    version="1.0.0",
    author="Alugunulla Vigneswara Kireeti (Vignesh)",
    description="Level 5 Fused RMSNorm+SwiGLU and L2 Cosine Similarity CUDA Kernels",
    ext_modules=[
        CUDAExtension(
            name="fused_cuda_ops",
            sources=["csrc/bindings.cpp", "csrc/fused_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
