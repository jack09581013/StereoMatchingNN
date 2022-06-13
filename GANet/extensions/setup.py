from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ganet_lib',
      ext_modules=[cpp_extension.CUDAExtension('ganet_lib', ['cuda_lib.cpp', 'cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})