import numpy as np
import cv2
import tensorrt as trt
from cuda.bindings import runtime as cudart
import torch
from torchvision.ops import nms

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

def generate_anchors(h, w, stride, num_anchors=2):
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
    keep = nms(torch.tensor(dets[:, :4], device='cuda'), torch.tensor(dets[:, 4], device='cuda'),0.4)

    return dets[keep.cpu().numpy()], kpss[keep.cpu().numpy()]
"""
def nms_old(dets, iou_threshold=0.4):
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
    keep = nms(torch.tensor(dets[:, :4], device='cuda'), torch.tensor(dets[:, 4], device='cuda'), iou_threshold)
    return keep
"""
"""
def nms_bad(dets, iou_threshold=0.4):
    boxes = torch.tensor(dets[:, :4], dtype=torch.float32, device="cuda")
    scores = torch.tensor(dets[:, 4], dtype=torch.float32, device="cuda")
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    return keep.cpu().numpy()
"""
class SCRFD_TRT_G:
    def __init__(self, engine_path, input_size=(640, 640), threshold=0.5, nms_thresh=0.4):
        self.engine_path = engine_path
        self.input_size = input_size
        self.threshold = threshold
        self.nms_thresh = nms_thresh
        self.prev_shape = None 
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self._closed = False
        _, self.stream = cudart.cudaStreamCreate()
        self.input_name = "input.1"
        self.graph_created = False
        self.graph_exec = None
        self.graph = None
        self.profile_idx = 0
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.context.set_optimization_profile_async(self.profile_idx, self.stream)
        self.context.set_input_shape(self.input_name, (1, 3, 640, 640))
        self._fixed_blob = np.empty((1, 3, *self.input_size), dtype=np.float32)

        assert self.context.all_binding_shapes_specified

        self.inputs, self.outputs, self.bindings = [], [], []
        for name in self.tensor_names:
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            mem = HostDeviceMem(size, dtype)
            self.bindings.append(int(mem.device))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append(mem)
            else:
                self.outputs.append(mem)

        for i, name in enumerate(self.tensor_names):
            self.context.set_tensor_address(name, self.bindings[i]) 
    
    def _load_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
 
    def _resize_pad(self, img):
        h, w = img.shape[:2]
        target_w, target_h = self.input_size
        img_ratio = h / w
        model_ratio = target_h / target_w

        if img_ratio > model_ratio:
            new_h = target_h
            new_w = int(new_h / img_ratio)
        else:
            new_w = target_w
            new_h = int(new_w * img_ratio)

        scale = new_h / h  # same as InsightFace's `det_scale`

        resized = cv2.resize(img, (new_w, new_h))
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized
        return padded, scale

    def _preprocess(self, img):
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1 / 128.0, size=self.input_size,
            mean=(127.5, 127.5, 127.5), swapRB=True
        )
        np.copyto(self._fixed_blob, blob)
        self.inputs[0].host = self._fixed_blob
 
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        
        if self.graph_exec:
            cudart.cudaGraphExecDestroy(self.graph_exec)
        if self.graph:
            cudart.cudaGraphDestroy(self.graph)

        for mem in self.inputs + self.outputs:
            try:
                mem.free()
            except Exception as e:
                print(f"[WARN] mem.free failed: {e}")

        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        try:
            if self.stream is not None:
                cudart.cudaStreamSynchronize(self.stream)
                cudart.cudaStreamDestroy(self.stream)
                self.stream = None
        except Exception as e:
            print(f"[WARN] Failed to destroy stream: {e}")

        self.context = None
        self.engine = None

    def detect(self, img, scale):
         
        self._preprocess(img)
        #np.copyto(self._fixed_blob, blob)
        self.inputs[0].host = self._fixed_blob
        if not self.graph_created:
            
            cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            cudart.cudaMemcpyAsync(
                self.inputs[0].device, self.inputs[0].host,
                self.inputs[0].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
            )
            self.context.execute_async_v3(stream_handle=self.stream)

            for out in self.outputs:
                cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

            graph = cuda_call(cudart.cudaStreamEndCapture(self.stream))
            graph_exec = cuda_call(cudart.cudaGraphInstantiate(graph, 0))


            self.graph = graph
            self.graph_exec = graph_exec
            self.graph_created = True

            cudart.cudaGraphLaunch(self.graph_exec, self.stream)
            #cudart.cudaStreamSynchronize(self.stream)

        else:
            cudart.cudaGraphLaunch(self.graph_exec, self.stream)
            #cudart.cudaStreamSynchronize(self.stream)

        results = [out.host for out in self.outputs]
        input_shape = (self._fixed_blob.shape[2], self._fixed_blob.shape[3])
        bboxes, kpss = postprocess_trt_outputs(results, input_shape, threshold=self.threshold)
        bboxes[:, :4] /= scale
        if kpss is not None:
            kpss /= scale
        
        if bboxes.shape[0] == 0:
            return [] 
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i,4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            #face = Face(bbox=bbox, kps=kps,det_score=det_score)
            ret.append((bbox,kps,det_score))
        if len(ret) == 0:
            return None
        return ret

    def draw(self, img, dets, kpss, color=(0, 255, 0), landmark_color=(0, 0, 255)):
        img_drawn = img.copy()
        for box, kps in zip(dets.astype(int), kpss.astype(int)):
            x1, y1, x2, y2, score = box
            cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 2)
            for kp in kps:
                cv2.circle(img_drawn, tuple(kp), 2, landmark_color, -1)
        return img_drawn

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f'[WARN] Error during SCRFD_TRT destructor: {e}')

