

# utils/preprocess.py
from ctypes import c_void_p, byref
from cuda.bindings import runtime as cudart
from cuda.bindings.driver import (
    cuInit, cuModuleLoad, cuModuleGetFunction, cuLaunchKernel, cuCtxGetCurrent
)
from trthelpers import cuda_call
import numpy as np
import os

class GpuPreprocessor:
    def __init__(self, ptx_path="/home/tpv/TPV/repos/beast-emb/full/preprocess.ptx", width=640, height=640):
        cuInit(0)
        ctx = cuCtxGetCurrent()
        self.width, self.height = width, height
        self.block = (32, 32, 1)
        self.grid = ((width + 31) // 32, (height + 31) // 32, 1)
        self.stream = cudart.cudaStreamCreate()[1]

        self.module = c_void_p()
        cuda_call(cuModuleLoad(byref(self.module), ptx_path.encode('utf-8')))

        self.kernel = c_void_p()
        cuda_call(cuModuleGetFunction(byref(self.kernel), self.module, b"preprocess_kernel"))

    def __call__(self, raw_ptr, blob_ptr):
        args = (
            c_void_p(raw_ptr),
            c_void_p(blob_ptr),
            np.int32(self.width),
            np.int32(self.height)
        )
        args_ptrs = (c_void_p * len(args))(*[
            c_void_p(id(arg)) if isinstance(arg, c_void_p) else byref(arg) for arg in args
        ])

        cuLaunchKernel(
            self.kernel,
            self.grid[0], self.grid[1], self.grid[2],
            self.block[0], self.block[1], self.block[2],
            0, self.stream,
            args_ptrs,
            None
        )
        cudart.cudaStreamSynchronize(self.stream)

