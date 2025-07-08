import numpy as np
import cv2
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import datetime

from insightface.model_zoo.scrfd import softmax, distance2bbox, distance2kps



class SCRFD_TRT:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._load_engine()
        self._allocate_buffers()
        self._init_vars()

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
                self.inputs.append((name, host_mem, device_mem))
            else:
                self.outputs.append((name, host_mem, device_mem))
                self.outputs.sort(key=lambda x: x[0])
            print(f"[ALLOC] {name}: shape={shape} size={size} dtype={dtype}")


    def _init_vars(self):
        self.input_mean = 127.5
        self.input_std = 128.0
        self.input_size = (self.input_shape[3], self.input_shape[2])
        self.fmc = 5
        self._feat_stride_fpn = [8, 16, 32, 64, 128]
        self._num_anchors = 1
        self.use_kps = (len(self.outputs) == self.fmc * 3)  # True if 15 outputs, else False
 
    def _resize_input(self, img, input_size):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        return det_img, det_scale

    def _limit_max(self, det, kpss, img_shape, max_num):
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        img_center = img_shape[0] // 2, img_shape[1] // 2
        offsets = np.vstack([
            (det[:, 0] + det[:, 2]) / 2 - img_center[1],
            (det[:, 1] + det[:, 3]) / 2 - img_center[0]
        ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        values = area - offset_dist_squared * 2.0
        bindex = np.argsort(values)[::-1][:max_num]
        det = det[bindex, :]
        kpss = kpss[bindex, :] if kpss is not None else None
        return det, kpss

    def prepare(self, ctx_id, **kwargs):
        self.nms_thresh = kwargs.get("nms_thresh", self.nms_thresh)
        self.det_thresh = kwargs.get("det_thresh", self.det_thresh)
        if "input_size" in kwargs:
            self.input_size = kwargs["input_size"]

    def preprocess(self, img):
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, self.input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        print(f"[Preprocess] blob.shape: {blob.shape}, total elements: {blob.size}")
        return blob

    def forward(self, img, threshold):
        scores_list, bboxes_list, kpss_list = [], [], []
        blob = self.preprocess(img)
        net_outs = self.infer(blob)

        input_height, input_width = blob.shape[2], blob.shape[3]

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx].reshape(-1)
            bbox_preds = net_outs[idx + self.fmc] * stride
            
            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape(-1, 2)
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape(-1, 2)
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            assert anchor_centers.shape[0] == bbox_preds.shape[0], \
                f"[ERROR] anchor_centers {anchor_centers.shape} vs bbox_preds {bbox_preds.shape}"

            pos_inds = np.where(scores >= threshold)[0]
            pos_scores = scores[pos_inds]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_bboxes = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds).reshape(kps_preds.shape[0], -1, 2)
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list


    def infer(self, blob):
        input_name, host_mem, device_mem = self.inputs[0]
        print(f"[Infer] host_mem.shape: {host_mem.shape}, blob.shape: {blob.shape}")
        np.copyto(host_mem, blob.ravel())
        cuda.memcpy_htod_async(device_mem, host_mem, self.stream)
        self.context.set_tensor_address(input_name, int(device_mem))

        output_list = []
        for name, host_mem, device_mem in self.outputs:
            self.context.set_tensor_address(name, int(device_mem))
            output_list.append((name, host_mem))

        self.context.execute_async_v3(self.stream.handle)

        for name, host_mem, device_mem in self.outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
        self.stream.synchronize()

        outputs = []
        for name, host_mem in output_list:
            shape = self.engine.get_tensor_shape(name)
            outputs.append(host_mem.reshape(shape))
        return outputs

    def detect(self, img, input_size=None, max_num=0):
        input_size = self.input_size if input_size is None else input_size
        det_img, det_scale = self._resize_input(img, input_size)
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list).ravel()
        order = scores.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        kpss = np.vstack(kpss_list)[order][keep] / det_scale if self.use_kps else None

        if max_num > 0 and det.shape[0] > max_num:
            det, kpss = self._limit_max(det, kpss, img.shape[:2], max_num)

        results = []
        for i in range(det.shape[0]):
            face_dict = {
                'bbox': det[i, 0:4],
                'det_score': det[i, 4],
            }
            if self.use_kps and kpss is not None:
                face_dict['kps'] = kpss[i]
            results.append(face_dict)

        return results
 

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

