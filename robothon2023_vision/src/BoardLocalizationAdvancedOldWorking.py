#! /usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, Transform, Vector3, TransformStamped
from RealSense import RealSense

import tf2_ros
import tf
import geometry_msgs.msg

import numpy as np
import cv2

from utils_detection import *
import datetime
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# import torch

# from utils.general import check_requirements, set_logging
# from utils.google_utils import attempt_download
# from utils.torch_utils import select_device
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
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'

SERVICE_CALLBACK = GREEN + "Service call {} received" + END
PARAM_NOT_DEFINED_ERROR = "Parameter: {} not defined"
SUCCESSFUL = "Successfully executed"
NOT_SUCCESSFUL = "Not Successfully executed"

SERVICE_NAME = "/robothon2022/board_localization"

@dataclass
class BoardLocalization:
    test: bool
    folder_path: str
    weight_path: str
    realsense: RealSense = field(init=False)
    broadcaster: tf2_ros.StaticTransformBroadcaster = field(init=False)

    def __post_init__(self):
        self.realsense = RealSense()

        # Retrieve camera parameters
        rospy.loginfo(YELLOW + "Waiting camera parameters ..." + END)

        self.realsense.getCameraParam()
        self.realsense.waitCameraInfo()

        rospy.loginfo(GREEN + "Camera parameters retrived correctly" + END)

        # Estimated parameters
        # self.depth = 300  # Estimated distance

        # Tf Broadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        rospy.loginfo(GREEN + "Service alive ...." + END)

    def callback(self, request):
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


        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME))
        # Acquire the rgb-frame
        self.realsense.acquireOnceBoth()
        rgb_frame = self.realsense.getColorFrame()
        #
        # # Save images if is test case
        # if self.test:
        #     now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        #     self.realsense.saveAquiredImage(f"{self.folder_path}frame_{now}.png")
        #
        # rospy.loginfo(RED + "Starting Identification..." + END)
        #
        # # Roi
        # crop_img = getRoi(rgb_frame, 144, 669, 360, 1150)
        # crop_img = rgb_frame
        #
        # # Color spaces
        # hsv, lab, bw = getAllColorSpaces(crop_img)
        # hue, saturation, value = cv2.split(hsv)
        # l_col, a_col, b_col = cv2.split(lab)
        #
        # # Threshold
        # ret, value_th = cv2.threshold(value, 90, 255, 0)
        #
        # # Get board contour
        # board_cnt, contours_limited, contours = getBoardContour(value_th)
        #
        # # Screen identification
        # ScreenPos, idx_screen = getScreen(a_col, contours_limited)
        #
        # contours_limited.pop(idx_screen)  # Remove screen contour from list
        #
        # rospy.loginfo(GREEN + "Screen position: {}".format(ScreenPos) + END)
        # new_img = getRoi(rgb_frame, 144, 669, 360, 1150)
        # new_img = cv2.circle(crop_img, (ScreenPos[0][0], ScreenPos[0][1]), 5, color=(255, 0, 0), thickness=2)
        #
        # rospy.loginfo(GREEN + "Identifying red button..." + END)
        # new_img = getRoi(rgb_frame, 144, 669, 360, 1150)
        # crop_img2 = getRoi(rgb_frame, 144, 669, 360, 1150)
        # RedBlueButPos, BlueButtonPos, id_red_blue_contour = getRedBlueButtonsNewVersion2(saturation, b_col,
        #                                                                                  contours_limited, crop_img2,
        #                                                                                  ScreenPos, rgb_frame)
        # rospy.loginfo(GREEN + "Identified red button" + END)
        # print(RedBlueButPos)
        # print(BlueButtonPos)
        # contours_limited.pop(id_red_blue_contour)
        # new_img = cv2.circle(crop_img, (RedBlueButPos[0], RedBlueButPos[1]), 5, color=(0, 0, 255), thickness=2)
        # new_img = cv2.circle(crop_img, (BlueButtonPos[0], BlueButtonPos[1]), 5, color=(255, 0, 0), thickness=2)
        # #
        # cv2.imshow("Identificato", new_img)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()
        #
        # crop_img = getRoi(rgb_frame, 144, 669, 360, 1150)
        #
        # depth_frame = self.realsense.getDistanceFrame()
        # print(f"Shape depth: {depth_frame.shape}")
        # print(f"Distanza: {depth_frame[RedBlueButPos[1], RedBlueButPos[0]]}")
        # print(f"Distanza: {depth_frame[BlueButtonPos[1], BlueButtonPos[0]]}")
        #
        # red_button_camera = np.array(self.realsense.deproject(RedBlueButPos[0], RedBlueButPos[1],
        #                                                       depth_frame[RedBlueButPos[1], RedBlueButPos[0]])) / 1000.0
        # blue_button_camera = np.array(self.realsense.deproject(BlueButtonPos[0], BlueButtonPos[1], depth_frame[
        #     BlueButtonPos[1], BlueButtonPos[0]])) / 1000.0
        # screen_camera = np.array(self.realsense.deproject(ScreenPos[0][0], ScreenPos[0][1],
        #                                                   depth_frame[ScreenPos[0][1], ScreenPos[0][0]])) / 1000.0
        #
        # listener = tf.TransformListener()
        # rospy.sleep(1.0)
        # found = False
        # while not found:
        #     try:
        #         (trans, rot) = listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
        #         found = True
        #         print("Retrieved camera_color_optical_frame -> base_link")
        #     except (tf.LookupException, tf.ConnectivityException):
        #         rospy.loginfo(YELLOW + "Unable to retrieve tf-t between: camera_color_optical_frame -> base_link" + END)
        #         rospy.sleep(0.5)
        #
        # rospy.loginfo(YELLOW + "Trasformata camera_link -> base_link \n :{}".format(trans) + RED)
        # rospy.loginfo(YELLOW + "Trasformata camera_link -> base_link \n :{}".format(rot) + RED)
        #
        # trans_world_camera = tf.transformations.translation_matrix(trans)
        # rot_world_camera = tf.transformations.quaternion_matrix(rot)
        # M_world_camera = np.dot(trans_world_camera, rot_world_camera)
        # # print(M_world_camera)
        #
        # # self.pubTF(red_button_camera,"red_prima","camera_color_optical_frame")
        # # self.pubTF(key_lock_camera,"keuy_prima","camera_color_optical_frame")
        # # self.pubTF(screen_camera,"screen_prima","camera_color_optical_frame")
        #
        # red_button_world = np.dot(M_world_camera, self.get4Vector(red_button_camera))
        # blue_button_world = np.dot(M_world_camera, self.get4Vector(blue_button_camera))
        # screen_world = np.dot(M_world_camera, self.get4Vector(screen_camera))
        # print(f"Red button (in base_link) before set z: {red_button_world}")
        # red_button_world_backup = red_button_world[0:-1]
        # red_button_world = red_button_world[0:-1]
        # red_button_world[-1] = 1.05591877  # 1.5
        # blue_button_world = blue_button_world[0:-1]
        # blue_button_world[-1] = 1.05591877  # red_button_world[-1]
        # screen_world = screen_world[0:-1]
        # screen_world[-1] = 1.05591877
        # z_axis = np.array([0.0, 0.0, -1.0])
        #
        # print(f"Red button (in base_link) after set z: {red_button_world}")
        # print(f"Blue button (in base_link) after set z: {blue_button_world}")
        # print(f"Screen (in base_link) after set z: {screen_world}")
        #
        # x_axis = (blue_button_world - red_button_world) / np.linalg.norm(blue_button_world - red_button_world)
        # y_axis_first_approach = np.cross(z_axis, x_axis)
        # y_axis_norm = y_axis_first_approach / np.linalg.norm(y_axis_first_approach)
        #
        # rot_mat_camera_board = np.array([x_axis, y_axis_norm, z_axis]).T
        # M_camera_board_only_rot = tf.transformations.identity_matrix()
        # M_camera_board_only_rot[0:-1, 0:-1] = rot_mat_camera_board
        #
        # M_camera_board_only_tra = tf.transformations.identity_matrix()
        # M_camera_board_only_tra[0:3, -1] = np.array(
        #     [red_button_world_backup[0], red_button_world_backup[1], red_button_world_backup[2]])
        #
        # M_camera_board = np.dot(M_camera_board_only_tra, M_camera_board_only_rot)
        #
        # rotation_quat = tf.transformations.quaternion_from_matrix(M_camera_board)
        #
        # # Boardcast board tf
        # rospy.loginfo(GREEN + "Publishing tf" + END)
        # static_transformStamped_board = self.getStaticTrasformStamped("base_link", "board",
        #                                                               M_camera_board[0:3, -1],
        #                                                               rotation_quat)
        #
        # static_transform_red = self.getStaticTrasformStamped("camera_color_optical_frame", "red",
        #                                                      red_button_camera,
        #                                                      [0, 0, 0, 1])
        # static_transform_blue = self.getStaticTrasformStamped("camera_color_optical_frame", "blue",
        #                                                       blue_button_camera,
        #                                                       [0, 0, 0, 1])
        # static_transform_screen = self.getStaticTrasformStamped("camera_color_optical_frame", "screen",
        #                                                         screen_camera,
        #                                                         [0, 0, 0, 1])
        #
        # rospy.loginfo(GREEN + "Published tf" + END)
        #
        # static_transformStamped_reference = self.getStaticTrasformStamped("board", "reference",
        #                                                                   [0.137, 0.094, -0.155],
        #                                                                   [0.0, 0.0, 0.959, -0.284])
        #
        # self.broadcaster.sendTransform(
        #     [static_transformStamped_board, static_transformStamped_reference, static_transform_red,
        #      static_transform_blue, static_transform_screen])
        # # self.broadcastTF(Quaternion(0,0,0.959,-0.284), Vector3(0.137,0.094,-0.155), "reference","board")

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
        print(img_size)
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
        frame = rgb_frame #cv2.imread("/home/galois/projects/third_parties_ws/src/software_modules/YOLOv6/data/custom_dataset/images/train/frame_5.png")
        # img, img_src = process_image(img_path, img_size, stride, half)
        img, img_src = process_online_image(frame, img_size, stride, half)
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        pred_results = model(img)
        print("Detection:")
        print(pred_results.shape)

        classes:Optional[List[int]] = None # the classes to keep

        conf_thres: float =.70 #@param {type:"number"}
        iou_thres: float =.45 #@param {type:"number"}
        max_det:int =  1000#@param {type:"integer"}
        agnostic_nms: bool = False #@param {type:"boolean"}
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        print("Filtrati:_")
        print(det)
        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src.copy()
        if len(det):
          det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()

          print("Stima :")
          print(det)
          for *xyxy, conf, cls in reversed(det):
              print(cls)
              print(conf)
              print(xyxy)
              class_num = int(cls)
              label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')


              center_x = int((int(xyxy[0].cpu().data.numpy()) + int(xyxy[2].cpu().data.numpy())) / 2)
              center_y = int((int(xyxy[1].cpu().data.numpy()) + int(xyxy[3].cpu().data.numpy())) / 2)
              img_ori = cv2.circle(img_ori, (center_x, center_y), 2, color=(255, 0, 0), thickness=2)
              # img_ori = cv2.circle(img_ori, (int(xyxy[0].cpu().data.numpy()), int(xyxy[1].cpu().data.numpy())), 5, color=(255, 0, 0), thickness=2)
              # img_ori = cv2.circle(img_ori, (int(xyxy[2].cpu().data.numpy()), int(xyxy[3].cpu().data.numpy())), 5, color=(255, 0, 0), thickness=2)

              # Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))

        # im = PIL.Image.fromarray(img_ori)
        cv2.imshow("Inference",img_ori)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()

        return TriggerResponse(True, SUCCESSFUL)

    def get4Vector(self, vect):
        vet = np.array([0.0, 0.0, 0.0, 1.0])
        vet[:-1] = vect
        return vet

    def getStaticTrasformStamped(self, header_frame_id_name, child_frame_id_name, tra, quat):
        static_transformStamped_board = geometry_msgs.msg.TransformStamped()
        static_transformStamped_board.header.stamp = rospy.Time.now()
        static_transformStamped_board.header.frame_id = header_frame_id_name
        static_transformStamped_board.child_frame_id = child_frame_id_name

        static_transformStamped_board.transform.translation.x = tra[0]
        static_transformStamped_board.transform.translation.y = tra[1]
        static_transformStamped_board.transform.translation.z = tra[2]

        static_transformStamped_board.transform.rotation.x = quat[0]
        static_transformStamped_board.transform.rotation.y = quat[1]
        static_transformStamped_board.transform.rotation.z = quat[2]
        static_transformStamped_board.transform.rotation.w = quat[3]
        return static_transformStamped_board


def main():
    rospy.init_node("board_localization")
    try:
        test = rospy.get_param("~test")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("test") + END)
        return 0
    try:
        images_path = rospy.get_param("~images_path")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("images_path") + END)
        return 0
    valid_images_path = os.path.exists(images_path)
    if not valid_images_path:
        rospy.logerr(RED + "Images_path does not exist" + END)
        return 0
    try:
        weight_path = rospy.get_param("~weight_path")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("weight_path") + END)
        return 0

    board_localization = BoardLocalization(test, images_path, weight_path)

    rospy.Service(SERVICE_NAME, Trigger, board_localization.callback)
    rospy.spin()


if __name__ == "__main__":
    main()
