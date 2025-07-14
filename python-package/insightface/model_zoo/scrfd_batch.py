import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from ..utils.trthelpers import HostDeviceMem, cuda_call
from ..app.common import Face

def distance2bbox(points, distances):
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distances):
    landmarks = []
    for i in range(0, distances.shape[1], 2):
        px = points[:, 0] + distances[:, i]
        py = points[:, 1] + distances[:, i + 1]
        landmarks.append(np.stack([px, py], axis=-1))
    return np.stack(landmarks, axis=1)

def generate_anchors(h, w, stride, num_anchors=1):
    shift_x = np.arange(w) * stride
    shift_y = np.arange(h) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    anchor_centers = np.stack((shift_x, shift_y), axis=-1).reshape(-1, 2)

    if num_anchors > 1:
        anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)

    return anchor_centers

def postprocess_trt_outputs(results, input_shape, threshold=0.5):
    strides = [8, 16, 32]
    fmc = len(strides)
    
    input_h, input_w = input_shape
    all_scores, all_bboxes, all_kps = [], [], []

    for i, stride in enumerate(strides):
        score = results[i].reshape(-1)
        bbox = results[i + fmc].reshape(-1, 4) * stride
        landmark = results[i + fmc * 2].reshape(-1, 10) * stride

        h, w = input_h // stride, input_w // stride
        anchors = generate_anchors(h, w, stride, num_anchors=2)

        boxes = distance2bbox(anchors, bbox)
        kps = distance2kps(anchors, landmark)

        inds = np.where(score > threshold)[0]
        if len(inds) == 0:
            continue

        all_scores.append(score[inds])
        all_bboxes.append(boxes[inds])
        all_kps.append(kps[inds])

    if not all_bboxes:
        return np.zeros((0, 5)), np.zeros((0, 5, 2))

    scores = np.concatenate(all_scores)
    bboxes = np.concatenate(all_bboxes)
    kpss = np.concatenate(all_kps)

    dets = np.hstack((bboxes, scores[:, None]))
    keep = nms(dets, iou_threshold=0.4)
    return dets[keep], kpss[keep]

def nms(dets, iou_threshold=0.4):
    x1, y1, x2, y2, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def split_batched_results(results, batch_size, input_shape):
    fmc = 3  # number of feature map strides
    h, w = input_shape
    per_image_results = [[] for _ in range(batch_size)]

    # Output ordering: [score_8, score_16, score_32, bbox_8, ..., kps_32]
    for i in range(len(results)):
        r = results[i]
        if i < fmc:
            # score, shape = (B * N,)
            per_image_size = r.shape[0] // batch_size
            r = r.reshape(batch_size, per_image_size)
        else:
            feat_dim = 4 if fmc <= i < 2*fmc else 10
            per_image_size = r.shape[0] // (batch_size * feat_dim)
            r = r.reshape(batch_size, per_image_size, feat_dim)

        for b in range(batch_size):
            per_image_results[b].append(r[b])

    return per_image_results



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
        print("Output tensor order:")
        for i, name in enumerate(self.tensor_names):
            print(f"[{i}] {name}")
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
        batch_size = blob_batch.shape[0]
        if batch_size not in self.graph_cache:
            inputs, outputs, bindings = self._allocate_buffers(batch_size)
            np.copyto(inputs[0].host.reshape(blob_batch.shape), blob_batch)

            cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            cudart.cudaMemcpyAsync(inputs[0].device, inputs[0].host, inputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            self.context.execute_async_v3(stream_handle=self.stream)
            for out in outputs:
                cudart.cudaMemcpyAsync(out.device, out.host, out.nbytes,
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

        entry = self.graph_cache[batch_size]
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        graph_exec = entry["graph_exec"]

        np.copyto(inputs[0].host.reshape(blob_batch.shape), blob_batch)
        cudart.cudaMemcpyAsync(inputs[0].device, inputs[0].host, inputs[0].nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        cudart.cudaGraphLaunch(graph_exec, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        all_results = [out.host.copy() for out in outputs]
        print(all_results)
        per_image_results = split_batched_results(all_results, batch_size, self.input_size)
        batch_results = []
        for i in range(batch_size):
            dets, kpss = postprocess_trt_outputs(per_image_results[i], self.input_size, threshold=self.threshold)
            dets[:, :4] /= scales[i]
            if kpss is not None:
                kpss /= scales[i]
            image_results = []
            for j in range(dets.shape[0]):
                image_results.append((dets[j, :4], kpss[j] if kpss is not None else None, dets[j, 4]))
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

