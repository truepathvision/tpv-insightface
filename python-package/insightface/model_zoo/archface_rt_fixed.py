import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from ..utils.trthelpers import HostDeviceMem, cuda_call
from ..app.common import Face
class ArcFaceRT:
    def __init__(self, engine_path, input_size=(112,112), mean=127.5, std=127.5, profile_idx=0):
        self.engine_path = engine_path
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.profile_idx = profile_idx
        
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        _, self.stream = cudart.cudaStreamCreate()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.graph_cache = {}
        self._fixed_blobs = {}
        self.tensor_names = [self.input_name, self.output_name]
        self._closed = False

    def _load_engine(self):
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    

    def _setup_batch(self, batch_size):
        self._fixed_blobs[batch_size] = np.empty((batch_size, 3, *self.input_size), dtype=np.float32)

        inputs, outputs, bindings = self._allocate_buffers(batch_size)
        fixed_blob = self._fixed_blobs[batch_size]
        np.copyto(inputs[0].host.reshape(fixed_blob.shape), fixed_blob)

        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        cudart.cudaMemcpyAsync(inputs[0].device, inputs[0].host, inputs[0].nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream)
        cudart.cudaMemcpyAsync(outputs[0].host, outputs[0].device, outputs[0].nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        graph = cuda_call(cudart.cudaStreamEndCapture(self.stream))
        graph_exec = cuda_call(cudart.cudaGraphInstantiate(graph, 0))

        self.graph_cache[batch_size] = {
            "inputs": inputs,
            "outputs": outputs,
            "bindings": bindings,
            "graph": graph,
            "graph_exec": graph_exec
        }


    
    def get(self, blob):
        batch_size = blob.shape[0]

        if batch_size not in self.graph_cache:
            self._setup_batch(batch_size)

        entry = self.graph_cache[batch_size]
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        graph_exec = entry["graph_exec"]
        fixed_blob = self._fixed_blobs[batch_size]

        np.copyto(fixed_blob, blob)
        np.copyto(inputs[0].host.reshape(blob.shape), fixed_blob)

        cudart.cudaMemcpyAsync(inputs[0].device, inputs[0].host, inputs[0].nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        cudart.cudaGraphLaunch(graph_exec, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        return outputs[0].host.reshape(batch_size, -1)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            for batch_size, entry in self.graph_cache.items():
                if entry.get('graph_exec') is not None:
                    cudart.cudaGraphExecDestroy(entry['graph_exec'])
                if entry.get('graph') is not None:
                    cudart.cudaGraphDestroy(entry['graph'])
                for mem in entry['inputs'] + entry['outputs']:
                    mem.free()
        
        except Exception as e:
            print(f'[Error]: Failed to close: {str(e)}')

        self.graph_cache.clear()
        try:
            if self.stream is not None:
                cudart.cudaStreamSynchronize(self.stream)
                cudart.cudaStreamDestroy(self.stream)
        
        except Exception as e:
            print(f'Failed to destroy stream: {str(e)}')
        self.context = None
        self.engine = None

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"[WARN] ArcFaceTRT destructor: {e}")
 
