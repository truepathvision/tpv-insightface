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
        self.input_size = None
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

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
                self.inputs.append((name, None, None))  # Delay alloc
            else:
                self.outputs.append((name, None, None))  # Delay alloc

        self.outputs.sort(key=lambda x: x[0])



    def _init_vars(self):
        self.input_mean = 127.5
        self.input_std = 128.0
       # self.input_size = (self.input_shape[3], self.input_shape[2])
    
        output_count = len(self.outputs)
    
        if output_count == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        else:
            raise ValueError(f"Unexpected number of outputs: {output_count}")

 
    def _resize_input(self, img, input_size):
        h, w = img.shape[:2]
        target_w, target_h = input_size

        im_ratio = h / w
        model_ratio = target_h / target_w

        if im_ratio > model_ratio:
            new_h = target_h
            new_w = int(w * target_h / h)
        else:
            new_w = target_w
            new_h = int(h * target_w / w)

        resized_img = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        det_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized_img

        scale_x = new_w / w
        scale_y = new_h / h
        return det_img, (scale_x, scale_y)

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
        assert self.input_size is not None, "Must set input_size for dynamic shape engines."
    
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
        input_name = self.input_name
        input_shape = blob.shape
        size = blob.size
        dtype = blob.dtype

    # Allocate input memory
        host_input = cuda.pagelocked_empty(size, dtype)
        device_input = cuda.mem_alloc(host_input.nbytes)
        np.copyto(host_input, blob.ravel())
        cuda.memcpy_htod_async(device_input, host_input, self.stream)

        self.context.set_input_shape(input_name, input_shape)
        self.context.set_tensor_address(input_name, int(device_input))

        output_list = []
        reshaped_outputs = []

        for name, _, _ in self.outputs:
            shape = self.context.get_tensor_shape(name)  # dynamic shape
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = np.prod(shape)
            host_output = cuda.pagelocked_empty((size,), dtype)

            device_output = cuda.mem_alloc(host_output.nbytes)
            self.context.set_tensor_address(name, int(device_output))

            output_list.append((name, host_output, device_output, shape))

        self.context.execute_async_v3(self.stream.handle)

        for name, host_output, device_output, _ in output_list:
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)

        self.stream.synchronize()

        for name, host_output, device_output, shape in output_list:
            reshaped_outputs.append(host_output.reshape(shape))

        return reshaped_outputs
 

    def detect(self, img, input_size=None, max_num=0):
        h, w = img.shape[:2]

    # Set the input size from image if not already set
        if input_size is None:
            if w < 240 or h < 240 or w > 1280 or h > 1280:
                raise ValueError(f"Input image size ({w}, {h}) is outside dynamic shape range [240–1280]")
        # Must be divisible by 32 to align with SCRFD FPN
            aligned_w = (w // 32) * 32
            aligned_h = (h // 32) * 32
            self.input_size = (aligned_w, aligned_h)
            print(f"[Auto-set] input_size set to: {self.input_size}")
        else:
            self.input_size = input_size

        det_img, det_scale = self._resize_input(img, self.input_size)
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

