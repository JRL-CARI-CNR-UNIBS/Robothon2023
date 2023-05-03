#! /usr/bin/env python3
from yolov6.core.inferer import Inferer
from yolov6.utils.nms import non_max_suppression
from yolov6.data.data_augment import letterbox

import torch
import numpy as np
import time


class OnlineInference(Inferer):
    def __init__(self, weights, device, yaml, img_size, half, source="", webcam="", webcam_addr=""):
        super().__init__(source, webcam, webcam_addr, weights, device, yaml, img_size, half)

    @staticmethod
    def process_online_image(frame, img_size, stride, half,):
        """Process image before image inference."""

        image = letterbox(frame, img_size, stride=stride)[0]

        # Convert
        image = image.transpose((2, 0, 1)) # [::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, frame
    def realtime_inference(self, img, conf_thres, iou_thres, agnostic_nms, max_det, view_img=False):
        inference_result = []
        img, img_src = self.process_online_image(img, self.img_size, self.stride, self.half)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
        t1 = time.time()
        pred_results = self.model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)[0]
        t2 = time.time()
        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src.copy()

        if len(det):
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = [int((int(xyxy[0].cpu().data.numpy()) + int(xyxy[2].cpu().data.numpy())) / 2),
                        int((int(xyxy[1].cpu().data.numpy()) + int(xyxy[3].cpu().data.numpy())) / 2),
                        int(xyxy[2] - xyxy[0]),
                        int(xyxy[3] - xyxy[1])]

                class_num = int(cls)
                class_name = self.class_names[class_num]

                if view_img:
                    label = (f'{self.class_names[class_num]} {conf:.2f}')
                    self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label,
                                            color=self.generate_colors(class_num, True))

                inference_result.append({"class_name": class_name, "xywh": xywh, "conf": conf.data.cpu().item()})
            img_src = np.asarray(img_ori)
        img_result = img_src.copy()
        inference_time = t2 - t1

        return inference_result, img_result, inference_time
