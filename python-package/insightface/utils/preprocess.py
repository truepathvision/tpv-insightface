from ctypes import c_void_p, POINTER, cast, byref
from cuda.bindings.driver import (
    cuInit, cuModuleLoad, cuModuleGetFunction, cuLaunchKernel, cuCtxGetCurrent
)
from cuda.bindings.runtime import cudaStreamCreate, cudaStreamSynchronize
from .trthelpers import cuda_call
import numpy as np

class GpuPreprocessor:
    def __init__(self, ptx_path="/home/tpv/TPV/repos/beast-emb/full/preprocess.ptx", width=640, height=640):
        cuInit(0)
        cuCtxGetCurrent()
        self.width, self.height = width, height
        self.block = (32, 32, 1)
        self.grid = ((width + 31) // 32, (height + 31) // 32, 1)
        self.stream = cudaStreamCreate()[1]

        # FIX: cuModuleLoad returns a CUmodule directly — do NOT use byref()
        self.module = cuModuleLoad(ptx_path.encode("utf-8"))

        # This one does need byref() to populate `self.kernel`
        self.kernel = c_void_p()
        cuda_call(cuModuleGetFunction(byref(self.kernel), self.module, b"preprocess_kernel"))

    def __call__(self, raw_ptr, blob_ptr):
        args = (
            c_void_p(raw_ptr),
            c_void_p(blob_ptr),
            np.int32(self.width),
            np.int32(self.height)
        )
        arg_ptrs = (c_void_p * len(args))(*args)
        arg_ptrs_ptr = cast(arg_ptrs, POINTER(c_void_p))

        cuLaunchKernel(
            self.kernel,
            self.grid[0], self.grid[1], self.grid[2],
            self.block[0], self.block[1], self.block[2],
            0, self.stream,
            arg_ptrs_ptr,
            None
        )
        cudaStreamSynchronize(self.stream)


