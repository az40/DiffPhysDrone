from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# Get PyTorch installation paths
torch_path = os.path.dirname(torch.__file__)
include_dirs = [os.path.join(torch_path, 'include')]
library_dirs = [os.path.join(torch_path, 'lib')]

# Get CUDA paths
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
if not os.path.exists(cuda_home):
    # Try common CUDA installation paths
    for path in ['/usr/local/cuda']:
        if os.path.exists(path):
            cuda_home = path
            break

cuda_include_dirs = [os.path.join(cuda_home, 'include')]
cuda_library_dirs = [os.path.join(cuda_home, 'lib64')]

# Combine all include and library directories
all_include_dirs = include_dirs + cuda_include_dirs
all_library_dirs = library_dirs + cuda_library_dirs

setup(
    name='quadsim_cuda',
    ext_modules=[
        CUDAExtension(
            'quadsim_cuda', 
            [
                'quadsim.cpp',
                'quadsim_kernel.cu',
                'dynamics_kernel.cu',
            ],
            include_dirs=all_include_dirs,
            library_dirs=all_library_dirs,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
