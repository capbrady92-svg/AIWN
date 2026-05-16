"""
Build script for IndexedLinear CUDA extension.
Handles Ubuntu 24.04 + CUDA 12.8 glibc math header conflict.

Usage:
    export CXX=g++-13 CC=gcc-13
    python setup_cuda.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


def get_arch_flags():
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    print(f"Detected GPU: SM{arch}")
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


setup(
    name="indexed_linear_cuda",
    ext_modules=[
        CUDAExtension(
            name="indexed_linear_cuda",
            sources=["indexed_linear_cuda.cu"],
            extra_compile_args={
                "cxx":  ["-O3", "-w"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    # Fix Ubuntu 24.04 glibc vs CUDA 12.8 math header conflict
                    "-Xcudafe=--diag_suppress=2977",
                    "-Xcudafe=--diag_suppress=20014",
                ] + get_arch_flags(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)