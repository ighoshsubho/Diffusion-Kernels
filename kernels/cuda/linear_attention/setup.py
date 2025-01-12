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
        include_dirs=[
            torch.utils.cpp_extension.include_paths()[0],
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })