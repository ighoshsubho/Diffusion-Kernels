from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='linear_attention_cuda',
    ext_modules=[
        CUDAExtension('linear_attention_cuda', [
            'linear_attention_cuda.cpp',
            'linear_attention_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3', 
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '--expt-relaxed-constexpr'
            ]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })