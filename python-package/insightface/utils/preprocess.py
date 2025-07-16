import numpy as np
from cuda.bindings import runtime as cudart
from cuda.bindings.driver import cuInit, cuModuleLoad, cuModuleGetFunction, cuLaunchKernel
from cuda.bindings.driver import cuCtxGetCurrent
from ctypes import c_void_p, byref

# Step 1: Initialize
cuInit(0)
ctx = cuCtxGetCurrent()

# Step 2: Load compiled kernel
module = c_void_p()
cudart.check_call(cuModuleLoad(byref(module), b"preprocess.ptx"))

# Step 3: Get function handle
kernel = c_void_p()
cudart.check_call(cuModuleGetFunction(byref(kernel), module, b"preprocess_kernel"))

# Step 4: Allocate input/output
width, height = 640, 640
img_size = width * height * 3
blob_size = 3 * width * height

raw_img_gpu = cudart.cudaMalloc(img_size)[1]
blob_out_gpu = cudart.cudaMalloc(blob_size * 4)[1]  # float32

# Step 5: Launch kernel
block = (32, 32, 1)
grid = ((width + 31) // 32, (height + 31) // 32, 1)

args = (
    c_void_p(raw_img_gpu),
    c_void_p(blob_out_gpu),
    np.int32(width),
    np.int32(height)
)
args_ptrs = (c_void_p * len(args))(*[c_void_p(id(arg)) if isinstance(arg, c_void_p) else byref(arg) for arg in args])

stream = cudart.cudaStreamCreate()[1]
cuLaunchKernel(
    kernel,
    grid[0], grid[1], grid[2],
    block[0], block[1], block[2],
    0, stream,
    args_ptrs,
    None
)
cudart.cudaStreamSynchronize(stream)

