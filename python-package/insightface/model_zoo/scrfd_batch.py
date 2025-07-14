import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from ..utils.trthelpers import HostDeviceMem, cuda_call
from ..app.common import Face
from scrfd_graph import postprocess_trt_outputs

class SCRFD_TRT_G_Batched:
    def __init__(self, engine_path, input_size=(640, 640), threshold=0.5, nms_thresh=0.4, profile_idx=0):
        self.engine_path = engine_path
        self.input_size = input_size
        self.threshold = threshold
        self.nms_thresh = nms_thresh
        self.profile_idx = profile_idx

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        _, self.stream = cudart.cudaStreamCreate()

        self.input_name = self.engine.get_tensor_name(0)
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.graph_cache = {}
        self._closed = False

    def _load_engine(self):
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self, batch_size):
        self.context.set_optimization_profile_async(self.profile_idx, self.stream)
        self.context.set_input_shape(self.input_name, (batch_size, 3, *self.input_size))
        assert self.context.all_binding_shapes_specified

        inputs, outputs, bindings = [], [], []

        for name in self.tensor_names:
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            mem = HostDeviceMem(size, dtype)
            bindings.append(int(mem.device))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(mem)
            else:
                outputs.append(mem)

        for i, name in enumerate(self.tensor_names):
            self.context.set_tensor_address(name, bindings[i])

        return inputs, outputs, bindings

    def detect(self, blob_batch, scales):
        """
        blob_batch: np.ndarray of shape [B, 3, H, W]
        scales: list of scale factors for each image
        """
        batch_size = blob_batch.shape[0]

        if batch_size not in self.graph_cache:
            inputs, outputs, bindings = self._allocate_buffers(batch_size)
            np.copyto(inputs[0].host.reshape(blob_batch.shape), blob_batch)

            cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            cudart.cudaMemcpyAsync(inputs[0].device, inputs[0].host, inputs[0].nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            self.context.execute_async_v3(stream_handle=self.stream)
            for out in outputs:
                cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
            graph = cudart.cudaStreamEndCapture(self.stream)
            graph_exec = cudart.cudaGraphInstantiate(graph, 0)
            self.graph_cache[batch_size] = {
                "inputs": inputs,
                "outputs": outputs,
                "bindings": bindings,
                "graph": graph,
                "graph_exec": graph_exec
            }

        entry = self.graph_cache[batch_size]
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        graph_exec = entry["graph_exec"]

        np.copyto(inputs[0].host.reshape(blob_batch.shape), blob_batch)
        cudart.cudaMemcpyAsync(inputs[0].device, inputs[0].host, inputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        cudart.cudaGraphLaunch(graph_exec, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        results = [out.host.copy() for out in outputs]
        input_shape = self.input_size

        # Per-image postprocessing
        stride_groups = len([8, 16, 32])
        batch_results = []
        for b in range(batch_size):
            offset = b * (results[0].size // batch_size)
            result_per_img = [
                r.reshape((batch_size, -1, *r.shape[1:]))[b] for r in results
            ]
            dets, kpss = postprocess_trt_outputs(result_per_img, input_shape, threshold=self.threshold)
            dets[:, :4] /= scales[b]
            if kpss is not None:
                kpss /= scales[b]
            image_results = []
            for i in range(dets.shape[0]):
                image_results.append((dets[i, :4], kpss[i] if kpss is not None else None, dets[i, 4]))
            batch_results.append(image_results)

        return batch_results

    def close(self):
        if self._closed:
            return
        self._closed = True
        for batch_size, entry in self.graph_cache.items():
            if entry.get('graph_exec'):
                cudart.cudaGraphExecDestroy(entry['graph_exec'])
            if entry.get('graph'):
                cudart.cudaGraphDestroy(entry['graph'])
            for mem in entry['inputs'] + entry['outputs']:
                mem.free()

        self.graph_cache.clear()
        try:
            if self.stream:
                cudart.cudaStreamSynchronize(self.stream)
                cudart.cudaStreamDestroy(self.stream)
        except Exception as e:
            print(f"Failed to destroy stream: {e}")
        self.context = None
        self.engine = None

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"[WARN] destructor: {e}")

