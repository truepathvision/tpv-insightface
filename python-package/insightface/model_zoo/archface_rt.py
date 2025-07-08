import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically manages CUDA context
from ..utils import face_align

class ArcFaceTensorRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine_path = engine_path

        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Bindings
        self.input_idx = self.engine.get_binding_index('input.1')
        self.output_idx = self.engine.get_binding_index(self.engine.get_binding_name(1))

        # Input/Output shapes
        self.input_shape = self.engine.get_binding_shape(self.input_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_idx)

        self.input_size = (self.input_shape[3], self.input_shape[2])  # width, height

        self.input_mean = 127.5
        self.input_std = 127.5

        self.batch_size = self.input_shape[0]
        self.input_nbytes = np.prod(self.input_shape) * np.float32().nbytes
        self.output_nbytes = np.prod(self.output_shape) * np.float32().nbytes

        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.input_nbytes)
        self.d_output = cuda.mem_alloc(self.output_nbytes)

        # Allocate host memory
        self.h_output = np.empty(self.output_shape, dtype=np.float32)

        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        return np.dot(feat1, feat2) / (np.norm(feat1) * np.norm(feat2))

    def get_feat(self, imgs):
        assert isinstance(imgs, list), "imgs should be a list of images"
        assert len(imgs) == self.batch_size, f"Expected batch of {self.batch_size} images"

    # Convert to NCHW tensor
        blob = cv2.dnn.blobFromImages(
            imgs,
            scalefactor=1.0 / self.input_std,
            size=self.input_size,
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        ).astype(np.float32)  # Shape: [B, 3, 112, 112]

        assert blob.shape == tuple(self.input_shape), f"Blob shape {blob.shape} doesn't match engine input {self.input_shape}"

        cuda.memcpy_htod_async(self.d_input, blob, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.copy()  # Shape: [B, 512] 

