import numpy as np
import cv2
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import datetime

class SCRFD_TRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self._load_engine()
        self._allocate_buffers()
        self._init_vars()


    def _load_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
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
                self.inputs.append((name, host_mem, device_mem))
                self.input_shape = shape  # Save it here!
            else:
                self.outputs.append((name, host_mem, device_mem))

 
 

    def _init_vars(self):
        self.input_mean = 127.5
        self.input_std = 128.0
        self.input_size = (self.input_shape[2], self.input_shape[1])
        self.fmc = 5
        self._feat_stride_fpn = [8, 16, 32, 64, 128]
        self._num_anchors = 1
        self.use_kps = True if len(self.outputs) == 15 else False
        self.det_thresh = 0.5
        self.nms_thresh = 0.4
        self.center_cache = {}

    def preprocess(self, img):
        input_size = (640, 640)
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        return blob
    
    def infer(self, blob):
        _, input_host, input_device = self.inputs[0]

        np.copyto(input_host, blob.ravel())
        cuda.memcpy_htod_async(input_device, input_host, self.stream)

    # Set input address
        input_name = self.engine.get_tensor_name(0)
        self.context.set_tensor_address(input_name, int(input_device))

    # Set output addresses
        output_list = []
        for i, (_, host_mem, device_mem) in enumerate(self.outputs):
            output_index = self.engine.num_input_tensors + i
            output_name = self.engine.get_tensor_name(output_index)
            self.context.set_tensor_address(output_name, int(device_mem))
            output_list.append((output_name, host_mem))  # Save for reshaping

    # Run inference
        self.context.execute_async_v3(self.stream.handle)

    # Copy outputs back
        for _,host_mem, device_mem in self.outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)

        self.stream.synchronize()

    # Reshape and return outputs
        outputs = []
        for name, host_mem in output_list:
            shape = self.engine.get_tensor_shape(name)
            outputs.append(host_mem.reshape(shape))
        return outputs
 
 

    def forward(self, img, threshold):
        scores_list, bboxes_list, kpss_list = [], [], []
        blob = self.preprocess(img)
        net_outs = self.infer(blob)

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx][0]
            bbox_preds = net_outs[idx + self.fmc][0] * stride
            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2][0] * stride

            height, width = input_height // stride, input_width // stride
            key = (height, width, stride)
            if key not in self.center_cache:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape(-1, 2)
                self.center_cache[key] = anchor_centers

            anchor_centers = self.center_cache[key]
            pos_inds = np.where(scores >= threshold)[0]
            pos_scores = scores[pos_inds]
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            pos_bboxes = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kpss = self._distance2kps(anchor_centers, kps_preds).reshape(kps_preds.shape[0], -1, 2)
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list if self.use_kps else None

    def detect(self, img, input_size=None, max_num=0):
        input_size = self.input_size if input_size is None else input_size
        det_img, det_scale = self._resize_input(img, input_size)
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list).ravel()
        order = scores.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores[:, None]))[order]

        keep = self._nms(pre_det)
        det = pre_det[keep]
        kpss = None
        if self.use_kps:
            kpss = np.vstack(kpss_list)[order][keep] / det_scale

        if max_num > 0 and det.shape[0] > max_num:
            det, kpss = self._limit_max(det, kpss, img.shape[:2], max_num)

        return det, kpss

    def _resize_input(self, img, input_size):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width] = resized_img
        scale = float(new_height) / img.shape[0]
        return det_img, scale

    def _distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _nms(self, dets):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
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
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def _limit_max(self, det, kpss, img_shape, max_num):
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        center = np.array([img_shape[1] // 2, img_shape[0] // 2])
        offsets = np.stack([
            (det[:, 0] + det[:, 2]) / 2 - center[0],
            (det[:, 1] + det[:, 3]) / 2 - center[1]
        ])
        dist = np.sum(np.power(offsets, 2.0), axis=0)
        values = area - dist * 2.0
        idx = np.argsort(values)[::-1][:max_num]
        return det[idx], kpss[idx] if kpss is not None else None

