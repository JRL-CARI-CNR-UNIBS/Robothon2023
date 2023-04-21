#! /usr/bin/env python3
# import rospy

import torch
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
import numpy as np
import os, requests, math

# from PIL import Image
import PIL
import cv2

def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor

def process_image(img_path, img_size, stride, half):
  '''Process image before image inference.'''

  try:
    from PIL import Image
    img_src = np.asarray(Image.open(img_path))
    assert img_src is not None, f'Invalid image: {img_path}'
  except Exception as e:
    LOGGER.Warning(e)
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))  # HWC to CHW
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src


def check_img_size(img_size, s=32, floor=0):
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

def process_online_image(frame, img_size, stride, half):
  '''Process image before image inference.'''

  # try:
  #
  #   img_src = frame
  #   assert img_src is not None, f'Invalid image: {img_path}'
  # except Exception as e:
  #   LOGGER.Warning(e)
  image = letterbox(frame, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))  # HWC to CHW
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, frame

def main():
    # rospy.init_node("prova")
    # img = cv2.imread()
    device = "gpu"
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    test = torch.load(f"/home/galois/projects/third_parties_ws/src/software_modules/YOLOv6/runs/train/exp3/weights/best_ckpt.pt")
    print(test)
    model = DetectBackend(f"/home/galois/projects/third_parties_ws/src/software_modules/YOLOv6/runs/train/exp3/weights/best_ckpt.pt", device=device)
    stride = model.stride

    class_names = load_yaml("/home/galois/projects/third_parties_ws/src/software_modules/YOLOv6/data/robothon.yaml")['names']
    img_size = [1280,720]
    half = False
    img_size = check_img_size(img_size, stride)  # check image size

    img_path = "/home/galois/projects/third_parties_ws/src/software_modules/YOLOv6/data/custom_dataset/images/train/frame_5.png"
    half = False
    hide_labels=False
    hide_conf=False
    if half & (device.type != 'cpu'):
      model.model.half()
    else:
      model.model.float()
      half = False

    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

    # model(torch.zeros(1, 3, 1280, 720).to(device).type_as(next(model.model.parameters())))
    # img_size = check_img_size(img_size, s=stride)
    #
    frame = cv2.imread("/home/galois/projects/third_parties_ws/src/software_modules/YOLOv6/data/custom_dataset/images/train/frame_5.png")
    # img, img_src = process_image(img_path, img_size, stride, half)
    img, img_src = process_online_image(frame, img_size, stride, half)
    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    classes:Optional[List[int]] = None # the classes to keep

    conf_thres: float =.25 #@param {type:"number"}
    iou_thres: float =.45 #@param {type:"number"}
    max_det:int =  1000#@param {type:"integer"}
    agnostic_nms: bool = False #@param {type:"boolean"}
    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_src.copy()
    if len(det):
      det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
      print(det)
      for *xyxy, conf, cls in reversed(det):
          print(cls)
          print(conf)
          print(xyxy)
          class_num = int(cls)
          label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
          Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
    # im = PIL.Image.fromarray(img_ori)
    cv2.imshow("Inference",img_ori)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    # im.show()

if __name__ == "__main__":
    main()
