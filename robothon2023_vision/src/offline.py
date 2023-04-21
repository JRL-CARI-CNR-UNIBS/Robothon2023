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

from PIL import Image
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
    labels_path: str
    realsense: RealSense = field(init=False)
    broadcaster: tf2_ros.StaticTransformBroadcaster = field(init=False)
    model: DetectBackend = field(init=False)
    # class_names: Dict(str) = field(init=False)
    stride: int = field(init=False)
    device: torch.device = field(init=False)
    img_size: List[int] = field(init=False)
    listener: tf.TransformListener = field(init=False)

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
        self.listener = tf.TransformListener()

        self.img_size = [1280, 720]

        # YOLO
        device = "gpu"
        cuda = device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        print(self.weight_path)
        self.model = DetectBackend(self.weight_path, device=self.device)
        self.stride = self.model.stride
        print(load_yaml(self.labels_path)['names'])
        self.class_names = load_yaml(self.labels_path)['names']
        # print(self.class_names)
        self.img_size = self.check_img_size(self.img_size, self.stride)  # check image size
        print(self.img_size)
        self.model.model.float()
        #
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, *self.img_size).to(self.device).type_as(
                    next(self.model.model.parameters())))  # warmup

        rospy.loginfo(GREEN + "Service alive ...." + END)

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size, list) else [new_size] * 2

    def process_online_image(self, frame, img_size, stride, half):
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

    def make_inference(self, frame, confidence):
        conf_thres= confidence # float = .70  # @param {type:"number"}
        iou_thres: float = .45  # @param {type:"number"}
        max_det: int = 1000  # @param {type:"integer"}
        half = False
        agnostic_nms: bool = False  # @param {type:"boolean"}

        img, img_src = self.process_online_image(frame, self.img_size, self.stride, half)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        pred_results = self.model(img)
        print("Detection:")
        print(pred_results.shape)

        classes: Optional[List[int]] = None  # the classes to keep

        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src.copy()

        features = {"red_button": [],
                    "door_handle": []}
        if len(det):
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            return det
        else:
            return None

    def callback(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME))

        # Acquire the rgb-frame
        self.realsense.acquireOnceBoth()
        rgb_frame = self.realsense.getColorFrame()

        # Save images if is test case
        if self.test:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.realsense.saveAquiredImage(f"{self.folder_path}frame_{now}.png")

        rospy.loginfo(RED + "Starting Identification..." + END)

        frame = rgb_frame

        hide_labels = False
        hide_conf = False

        features = {"red_button": [],
                    "door_handle": []}

        det = self.make_inference(frame, 0.7)
        img_ori = rgb_frame.copy()
        if det is not None:
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)

                print(f"Category: {self.class_names[class_num]}")
                label = None if hide_labels else (
                    self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')

                print((xyxy[0].cpu().data.numpy(), xyxy[1].cpu().data.numpy()))

                center_x = int((int(xyxy[0].cpu().data.numpy()) + int(xyxy[2].cpu().data.numpy())) / 2)
                center_y = int((int(xyxy[1].cpu().data.numpy()) + int(xyxy[3].cpu().data.numpy())) / 2)
                if self.class_names[class_num] in features:
                    features[self.class_names[class_num]] = [center_x, center_y]

                img_ori = cv2.circle(img_ori, (center_x, center_y), 2, color=(255, 0, 0), thickness=2)

        depth_frame = self.realsense.getDistanceFrame()

        print(f"Shape depth: {depth_frame.shape}")
        print(f"Distanza: {depth_frame[features['red_button'][1], features['red_button'][1]]}")

        red_button_camera = np.array(self.realsense.deproject(features['red_button'][0], features['red_button'][1],
                                                              depth_frame[
                                                                  features['red_button'][1], features['red_button'][
                                                                      0]])) / 1000.0
        door_handle_camera = np.array(self.realsense.deproject(features['door_handle'][0], features['door_handle'][1],
                                                               depth_frame[
                                                                   features['door_handle'][1], features['door_handle'][
                                                                       0]])) / 1000.0
        found = False
        while not found:
            try:
                (trans, rot) = self.listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
                found = True
                print("Retrieved camera_color_optical_frame -> base_link")
            except (tf.LookupException, tf.ConnectivityException):
                rospy.loginfo(YELLOW + "Unable to retrieve tf-t between: camera_color_optical_frame -> base_link" + END)
                rospy.sleep(0.5)

        rospy.loginfo(YELLOW + "Trasformata camera_link -> base_link \n :{}".format(trans) + RED)
        rospy.loginfo(YELLOW + "Trasformata camera_link -> base_link \n :{}".format(rot) + RED)

        trans_world_camera = tf.transformations.translation_matrix(trans)
        rot_world_camera = tf.transformations.quaternion_matrix(rot)
        M_world_camera = np.dot(trans_world_camera, rot_world_camera)

        red_button_world = np.dot(M_world_camera, self.get4Vector(red_button_camera))
        door_handle_world = np.dot(M_world_camera, self.get4Vector(door_handle_camera))
        # screen_world = np.dot(M_world_camera, self.get4Vector(screen_camera))
        print(f"Red button (in base_link) before set z: {red_button_world}")
        red_button_world_backup = red_button_world[0:-1]
        red_button_world = red_button_world[0:-1]
        red_button_world[-1] = 0.923  # 1.5
        door_handle_world = door_handle_world[0:-1]
        door_handle_world[-1] = 0.923  # red_button_world[-1]
        z_axis = np.array([0.0, 0.0, -1.0])

        print(f"Red button (in base_link) after set z: {red_button_world}")
        print(f"Blue button (in base_link) after set z: {door_handle_world}")
        # print(f"Screen (in base_link) after set z: {screen_world}")

        x_axis = (door_handle_world - red_button_world) / np.linalg.norm(door_handle_world - red_button_world)
        y_axis_first_approach = np.cross(z_axis, x_axis)
        y_axis_norm = y_axis_first_approach / np.linalg.norm(y_axis_first_approach)

        rot_mat_camera_board = np.array([x_axis, y_axis_norm, z_axis]).T
        M_camera_board_only_rot = tf.transformations.identity_matrix()
        M_camera_board_only_rot[0:-1, 0:-1] = rot_mat_camera_board

        M_camera_board_only_tra = tf.transformations.identity_matrix()
        M_camera_board_only_tra[0:3, -1] = np.array(
            [red_button_world_backup[0], red_button_world_backup[1], red_button_world_backup[2]])

        M_camera_board = np.dot(M_camera_board_only_tra, M_camera_board_only_rot)

        rotation_quat = tf.transformations.quaternion_from_matrix(M_camera_board)

        # Boardcast board tf
        rospy.loginfo(GREEN + "Publishing tf" + END)
        static_transformStamped_board = self.getStaticTrasformStamped("base_link", "board",
                                                                      M_camera_board[0:3, -1],
                                                                      rotation_quat)

        static_transform_red = self.getStaticTrasformStamped("camera_color_optical_frame", "red",
                                                             red_button_camera,
                                                             [0, 0, 0, 1])
        static_transform_door = self.getStaticTrasformStamped("camera_color_optical_frame", "blue",
                                                              door_handle_camera,
                                                              [0, 0, 0, 1])

        rospy.loginfo(GREEN + "Published tf" + END)

        # static_transformStamped_reference = self.getStaticTrasformStamped("board", "reference",
        #                                                                   [0.137, 0.094, -0.155],
        #                                                                   [0.0, 0.0, 0.959, -0.284])

        roi_red_plug = rgb_frame.copy()
        red = np.array([features['red_button'][0], features['red_button'][1]])
        door = np.array([features['door_handle'][0], features['door_handle'][1]])
        test = (door - red) / (np.linalg.norm(door - red))
        orto = np.array([-test[1] / test[0], 1])
        shift_start = 100
        shift_end = 160
        y_roi_plug_start_lin = int(features['red_button'][0] + test[0] * shift_start)
        x_roi_plug_star_lin = int(features['red_button'][1] + test[1] * shift_start)
        y_roi_plug_end_lin = int(features['red_button'][0] + test[0] * shift_end)
        x_roi_plug_end_lin = int(features['red_button'][1] + test[1] * shift_end)
        shift_start_orto = -70
        shift_end_orto = 70
        y_roi_plug_start_orto = int(int((y_roi_plug_start_lin + y_roi_plug_end_lin) / 2) + orto[0] * shift_start_orto)
        y_roi_plug_end_orto = int(int((y_roi_plug_start_lin + y_roi_plug_end_lin) / 2) + orto[0] * shift_end_orto)
        x_roi_plug_star_orto = int(int((x_roi_plug_star_lin + x_roi_plug_end_lin) / 2) + orto[1] * shift_start_orto)
        x_roi_plug_end_orto = int(int((x_roi_plug_star_lin + x_roi_plug_end_lin) / 2) + orto[1] * shift_end_orto)

        x_min = min([x_roi_plug_star_lin, x_roi_plug_end_lin, x_roi_plug_star_orto, x_roi_plug_end_orto])
        x_max = max([x_roi_plug_star_lin, x_roi_plug_end_lin, x_roi_plug_star_orto, x_roi_plug_end_orto])

        y_min = min([y_roi_plug_start_lin, y_roi_plug_end_lin, y_roi_plug_start_orto, y_roi_plug_end_orto])
        y_max = max([y_roi_plug_start_lin, y_roi_plug_end_lin, y_roi_plug_start_orto, y_roi_plug_end_orto])
        roi_red_plug[:, :, :] = 0

        roi_red_plug[x_min:x_max, y_min:y_max] = rgb_frame[x_min:x_max, y_min:y_max]

        det = self.make_inference(roi_red_plug, 0.2)
        cv2.imshow("imgpart", roi_red_plug)

        img_ori = rgb_frame.copy()
        if det is not None:
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)

                print(f"Category: {self.class_names[class_num]}")
                label = None if hide_labels else (
                    self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')

                print((xyxy[0].cpu().data.numpy(), xyxy[1].cpu().data.numpy()))

                center_x = int((int(xyxy[0].cpu().data.numpy()) + int(xyxy[2].cpu().data.numpy())) / 2)
                center_y = int((int(xyxy[1].cpu().data.numpy()) + int(xyxy[3].cpu().data.numpy())) / 2)
                # if self.class_names[class_num] in features:
                #     features[self.class_names[class_num]] = [center_x, center_y]

                img_ori = cv2.circle(img_ori, (center_x, center_y), 2, color=(255, 0, 0), thickness=2)
                # img_ori = cv2.circle(img_ori, (int(xyxy[0].cpu().data.numpy()), int(xyxy[1].cpu().data.numpy())), 5, color=(255, 0, 0), thickness=2)
                # img_ori = cv2.circle(img_ori, (int(xyxy[2].cpu().data.numpy()), int(xyxy[3].cpu().data.numpy())), 5, color=(255, 0, 0), thickness=2)

                Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label,
                                           color=Inferer.generate_colors(class_num, True))

        # for t in range(-70, 70):
        #
        #     y_roi_plug = int(y_roi_plug_start + orto[0] * t)
        #     x_roi_plug = int(x_roi_plug_star + orto[1] * t)
        #     rgb_frame = cv2.circle(rgb_frame, (y_roi_plug, x_roi_plug), 5, color=(255, 0, 0), thickness=2)
        #
        #     print(y_roi_plug)
        #     print(x_roi_plug)
        #
        # rgb_frame = cv2.circle(rgb_frame, (y_min,x_min), 5, color=(255, 0, 0), thickness=2)
        # rgb_frame = cv2.circle(rgb_frame, (y_max, x_max), 5, color=(255, 0, 0), thickness=2)

        # rgb_frame = cv2.circle(rgb_frame, (features['red_button'][0], features['red_button'][1]), 5, color=(255, 0, 0), thickness=2)

        cv2.imshow("img", img_ori)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        print(x_axis)

        self.broadcaster.sendTransform(
            [static_transformStamped_board, static_transform_red,
             static_transform_door])
        # self.broadcastTF(Quaternion(0,0,0.959,-0.284), Vector3(0.137,0.094,-0.155), "reference","board")
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
    try:
        labels_path = rospy.get_param("~labels_path")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("labels_path") + END)
        return 0
    print(labels_path)
    board_localization = BoardLocalization(test, images_path, weight_path, labels_path)

    rospy.Service(SERVICE_NAME, Trigger, board_localization.callback)
    rospy.spin()


if __name__ == "__main__":
    main()
