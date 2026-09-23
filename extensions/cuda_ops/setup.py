"""
Setup script for compiling the pbc_graph_outer CUDA extension.

Usage:
    python setup_outer.py install
    # or
    python setup_outer.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get the directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name='pbc_graph_outer_cuda',
    version='0.1.0',
    description='CUDA-accelerated PBC graph operations for batchASE',
    python_requires='>=3.10',
    py_modules=['pbc_graph_outer_cuda'],
    ext_modules=[
        CUDAExtension(
            name='pbc_graph_outer_cuda',
            sources=[
                'pbc_graph_outer_wrapper.cpp',
                'pbc_graph_outer.cu',
            ],
            include_dirs=[current_dir],
            library_dirs=[torch_lib_dir],
            runtime_library_dirs=[torch_lib_dir],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '-gencode=arch=compute_90,code=sm_90',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)

