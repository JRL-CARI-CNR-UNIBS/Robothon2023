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
from typing import List, Dict, Optional, Tuple
import multiprocessing
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
# from sensor_msgs.msg import CameraInfo, Image
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
# import PIL
import cv2
from slider_detect_triangles import *
from identification_function import *

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import supervision as sv

from Inference import OnlineInference

COLOR_FRAME_TOPIC = '/camera/color/image_raw'

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'

SERVICE_CALLBACK = GREEN + "Service call {} received" + END
PARAM_NOT_DEFINED_ERROR = "Parameter: {} not defined"
SUCCESSFUL = "Successfully executed"
NOT_SUCCESSFUL = "Not Successfully executed"

SERVICE_NAME_BOARD = "/robothon2023/board_localization"
SERVICE_NAME_SCREEN_TARGET_1 = "/robothon2023/screen_target_init"
SERVICE_NAME_SCREEN_TARGET_2 = "/robothon2023/screen_target"
ALIGNMENT_SCREEN_TOPIC = "/robothon2023/align_screen"

SLIDER_TRAVEL_MM = 9.1  # 8.5 #29.0
SLIDER_TRAVEL_PX = 56  # 56  # 49 #256.0

MODEL_TYPE = "vit_b"


@dataclass
class BoardLocalization:
    test: bool
    folder_path: str
    weight_path: str
    weight_path_ssm: str
    labels_path: str
    realsense: RealSense = field(init=False)
    broadcaster: tf2_ros.StaticTransformBroadcaster = field(init=False)
    model: DetectBackend = field(init=False)
    # class_names: Dict(str) = field(init=False)
    stride: int = field(init=False)
    device: torch.device = field(init=False)
    img_size: List[int] = field(init=False)
    listener: tf.TransformListener = field(init=False)
    red_button_camera: np.array = field(default=None, init=False)
    door_handle_camera: np.array = field(default=None, init=False)
    n_frame: int = field(default=0, init=False)
    start_triangle_pos: Tuple[int, int] = field(default=None, init=False)

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

        self.direction = 1
        self.k = 0
        self.image_publisher = rospy.Publisher("detected_image", sensor_msgs.msg.Image, queue_size=50)
        self.bridge = CvBridge()
        self.first_identification = True
        #
        # self.sam = sam_model_registry[MODEL_TYPE](checkpoint=self.weight_path_ssm).to(device=self.device)
        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        self.online_board_inference = OnlineInference(weights=self.weight_path,
                                                      device=0,
                                                      yaml=self.labels_path,
                                                      img_size=[1280, 720],
                                                      half=False)
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
        conf_thres = confidence  # float = .70  # @param {type:"number"}
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

    def generate_colors(self, i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color

    # def plot_box_and_label(self, image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),font=cv2.FONT_HERSHEY_COMPLEX):
    def plot_box_and_label(self, image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),
                           font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # p1, p2 = pp
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    def board_localization(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME_BOARD))
        rospy.sleep(1)
        self.red_button_camera = None
        self.door_handle_camera = None
        n_red = 0
        n_door = 0
        max_trial = 5
        for k in range(0, max_trial):
            # Acquire the rgb-frame
            self.realsense.acquireOnceBoth()
            rgb_frame = self.realsense.getColorFrame()

            self.n_frame += 1

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

            inference_result, _, _ = self.online_board_inference.realtime_inference(frame, conf_thres=0.6,
                                                                                    iou_thres=.45, agnostic_nms=False,
                                                                                    max_det=1000, view_img=False)

            print("---------------- INFERENCE RESULT -----------")
            print(inference_result)
            det = self.make_inference(frame, 0.6)
            img_ori = rgb_frame.copy()
            for recognize_object in inference_result:
                class_name = recognize_object["class_name"]
                xywh = recognize_object["xywh"]
                conf = recognize_object["conf"]
                if class_name in features:
                    if features[class_name]:  # If already exist take the one with biggest conf
                        if features[class_name][2] > conf:
                            continue
                    features[class_name] = [xywh[0], xywh[1], conf]
            print(features)
            #
            # features = {"red_button": [],
            #             "door_handle": []}
            # if det is not None:
            #     for *xyxy, conf, cls in reversed(det):
            #         print("-----")
            #         print(xyxy)
            #         class_num = int(cls)
            #
            #         print(f"Category: {self.class_names[class_num]}")
            #         print(f"Confidence: {conf}")
            #         label = None if hide_labels else (
            #             self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
            #
            #         center_x = int((int(xyxy[0].cpu().data.numpy()) + int(xyxy[2].cpu().data.numpy())) / 2)
            #         center_y = int((int(xyxy[1].cpu().data.numpy()) + int(xyxy[3].cpu().data.numpy())) / 2)
            #         print(f"Centri: {center_x, center_y}")
            #         if self.class_names[class_num] in features:
            #             features[self.class_names[class_num]] = [center_x, center_y]
            #
            #         img_ori = cv2.circle(img_ori, (center_x, center_y), 2, color=(255, 0, 0), thickness=2)
            #         box_xyxy = (xyxy[0].cpu().data.numpy(), xyxy[1].cpu().data.numpy(), xyxy[2].cpu().data.numpy(),
            #                     xyxy[3].cpu().data.numpy())
            #         # print(box_xyxy)
            #         self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), box_xyxy, label,
            #                                 color=self.generate_colors(class_num, True))
            #         cv2.imwrite("/home/galois/init_file.png", img_ori)
            # print(features)
            depth_frame = self.realsense.getDistanceFrame()

            print(f"Shape depth: {depth_frame.shape}")

            red_button_camera = None
            door_handle_camera = None
            if features['red_button']:
                n_red += 1
                red_button_camera = np.array(
                    self.realsense.deproject(features['red_button'][0], features['red_button'][1],
                                             depth_frame[
                                                 features['red_button'][1], features['red_button'][
                                                     0]])) / 1000.0
                if self.red_button_camera is not None:
                    self.red_button_camera = self.red_button_camera + 1.0 / n_red * (
                            red_button_camera - self.red_button_camera)
                else:
                    self.red_button_camera = red_button_camera
                print(f"Diff red: {red_button_camera - self.red_button_camera}")

            if features['door_handle']:
                n_door += 1
                door_handle_camera = np.array(
                    self.realsense.deproject(features['door_handle'][0], features['door_handle'][1],
                                             depth_frame[
                                                 features['door_handle'][1], features['door_handle'][
                                                     0]])) / 1000.0
                if self.door_handle_camera is not None:
                    self.door_handle_camera = self.door_handle_camera + 1.0 / n_door * (
                            door_handle_camera - self.door_handle_camera)
                else:
                    self.door_handle_camera = door_handle_camera
                print(f"Diff door: {door_handle_camera - self.door_handle_camera}")
            if self.first_identification is True and n_red >= 1 and n_door >= 1:
                self.first_identification = False
                break
            if k >= max_trial - 1:
                self.first_identification = True

        print(self.red_button_camera)
        print(self.door_handle_camera)

        if (self.red_button_camera is None) or (self.door_handle_camera is None):
            print(RED + "Something not identified" + END)
            return TriggerResponse(False, NOT_SUCCESSFUL)

        print(YELLOW + f"Red button found: {n_red} " + END)
        print(YELLOW + f"Door handle found: {n_door} " + END)

        found = False
        while not found:
            try:
                (trans, rot) = self.listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
                found = True
                print("Retrieved camera_color_optical_frame -> base_link")
            except (tf.LookupException, tf.ConnectivityException):
                rospy.loginfo(YELLOW + "Unable to retrieve tf-t between: camera_color_optical_frame -> base_link" + END)
                rospy.sleep(0.1)

        rospy.loginfo(YELLOW + "Trasformata camera_link -> base_link \n :{}".format(trans) + RED)
        rospy.loginfo(YELLOW + "Trasformata camera_link -> base_link \n :{}".format(rot) + RED)

        trans_world_camera = tf.transformations.translation_matrix(trans)
        rot_world_camera = tf.transformations.quaternion_matrix(rot)
        M_world_camera = np.dot(trans_world_camera, rot_world_camera)

        red_button_world = np.dot(M_world_camera, self.get4Vector(self.red_button_camera))
        door_handle_world = np.dot(M_world_camera, self.get4Vector(self.door_handle_camera))
        # screen_world = np.dot(M_world_camera, self.get4Vector(screen_camera))
        print(f"Red button (in base_link) before set z: {red_button_world}")
        red_button_world_backup = red_button_world[0:-1]
        red_button_world = red_button_world[0:-1]
        red_button_world[-1] = 0.933  # 0.923
        door_handle_world = door_handle_world[0:-1]
        door_handle_world[-1] = 0.933  # 0.923
        z_axis = np.array([0.0, 0.0, -1.0])

        print(f"Red button (in base_link) after set z: {red_button_world}")
        print(f"Door handle (in base_link) after set z: {door_handle_world}")

        # print(f"Screen (in base_link) after set z: {screen_world}")

        x_axis = (door_handle_world - red_button_world) / np.linalg.norm(door_handle_world - red_button_world)
        y_axis_first_approach = np.cross(z_axis, x_axis)
        y_axis_norm = y_axis_first_approach / np.linalg.norm(y_axis_first_approach)

        rot_mat_camera_board = np.array([x_axis, y_axis_norm, z_axis]).T
        M_camera_board_only_rot = tf.transformations.identity_matrix()
        M_camera_board_only_rot[0:-1, 0:-1] = rot_mat_camera_board

        M_camera_board_only_tra = tf.transformations.identity_matrix()
        M_camera_board_only_tra[0:3, -1] = np.array(
            [red_button_world_backup[0], red_button_world_backup[1], 0.933])

        M_camera_board = np.dot(M_camera_board_only_tra, M_camera_board_only_rot)

        rotation_quat = tf.transformations.quaternion_from_matrix(M_camera_board)

        # Boardcast board tf
        rospy.loginfo(GREEN + "Publishing tf" + END)
        static_transformStamped_board = self.getStaticTrasformStamped("base_link", "board",
                                                                      M_camera_board[0:3, -1],
                                                                      rotation_quat)

        static_transform_red = self.getStaticTrasformStamped("camera_color_optical_frame", "red",
                                                             self.red_button_camera,
                                                             [0, 0, 0, 1])
        static_transform_door = self.getStaticTrasformStamped("camera_color_optical_frame", "door_handle",
                                                              self.door_handle_camera,
                                                              [0, 0, 0, 1])

        world_to_board_euler_rot_z = math.atan2(rot_mat_camera_board[1][0], rot_mat_camera_board[0][0])

        tf_params = rospy.get_param('tf_params')

        if ((world_to_board_euler_rot_z > math.radians(90)) or (world_to_board_euler_rot_z < math.radians(-160))):
            print('Rotate position: ' + str(math.degrees(world_to_board_euler_rot_z)))
            rospy.set_param("/RL_params/cable_winding/release_to_rotation/traslation", [0.0, 0.0, -0.03])
            rospy.set_param("/RL_params/cable_winding/release_to_rotation/rotation", [0.0, 0.0, 0.0, 1.0])
            rospy.set_param("/RL_params/cable_winding/rotation/traslation", [0.0, 0.0, 0.0])
            rospy.set_param("/RL_params/cable_winding/rotation/rotation", [0.0, 0.0, 0.707, 0.707])
            rospy.set_param("/RL_params/cable_winding/push_to_pick/MOVEY", 1.0)
            rospy.set_param("/RL_params/cable_winding/push_2/MOVEY", 1.0)
            for id, single_tf in enumerate(tf_params):
                if single_tf["name"] == "probe_insertion":
                    tf_params[id]["position"] = [0.2275, 0.097, -0.012]
                    tf_params[id]["quaternion"] = [0.000, 0.000, 0.000, 1.000]
        else:
            print('Normal position: ' + str(math.degrees(world_to_board_euler_rot_z)))
            rospy.set_param("/RL_params/cable_winding/release_to_rotation/traslation", [0.0, 0.0, 0.0])
            rospy.set_param("/RL_params/cable_winding/release_to_rotation/rotation", [0.0, 0.0, 0.0, 1.0])
            rospy.set_param("/RL_params/cable_winding/rotation/traslation", [0.0, 0.0, 0.0])
            rospy.set_param("/RL_params/cable_winding/rotation/rotation", [0.0, 0.0, 0.0, 1.0])
            rospy.set_param("/RL_params/cable_winding/push_to_pick/MOVEY", -1.0)
            rospy.set_param("/RL_params/cable_winding/push_2/MOVEY", -1.0)
            for id, single_tf in enumerate(tf_params):
                if single_tf["name"] == "probe_insertion":
                    tf_params[id]["position"] = [0.2275, 0.097, -0.014]
                    tf_params[id]["quaternion"] = [0.000, 0.000, 1.000, 0.000]

        rospy.set_param('tf_params', tf_params)

        rospy.loginfo(GREEN + "Published tf" + END)

        self.broadcaster.sendTransform(
            [static_transformStamped_board, static_transform_red,
             static_transform_door])

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

    def screen_target_1(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME_SCREEN_TARGET_1))
        self.realsense.acquireOnceBoth()
        rgb_frame = self.realsense.getColorFrame()
        if self.test:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}.png")
        screen = rgb_frame[360:425, 660:810]
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        cv2.imshow("rgb_frame", rgb_frame)
        cv2.imshow("screen", screen)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        _, saturation, _ = cv2.split(hsv)
        # cv2.imshow('saturation', saturation)
        _, thresh = cv2.threshold(saturation, 55, 255, 0)
        # Set image contour white
        thresh[-1, :] = 255
        thresh[:, -1] = 255
        thresh[30:, :] = 255
        # cv2.imshow('thresh', thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centri = []
        for idx, cnt in enumerate(contours):
            # epsilon = 0.01 * cv2.arcLength(cnt, True)
            # cnt = cv2.approxPolyDP(cnt, epsilon, True)
            area = cv2.contourArea(cnt)
            print(area)
            if (area > 1000) or (area < 110):
                continue
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Centro triangolo sopra: (cx,cy) {cX, cY}")
            cv2.circle(screen, (cX, cY), 3, (0, 0, 255), -1)
            centri.append([cX, cY])

        self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}_init.png")
        if len(centri) == 1:
            self.start_triangle_pos = centri[0]
            self.triangle_mask = thresh
        else:
            pass
            # TODO:

        print(GREEN + f"Identified triangle in: {self.start_triangle_pos}" + END)
        return TriggerResponse(True, SUCCESSFUL)

    def screen_target_2(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME_SCREEN_TARGET_2))
        print(f"Previous identification: {self.start_triangle_pos}")
        rospy.sleep(2.5)
        self.realsense.acquireOnceBoth()
        rgb_frame = self.realsense.getColorFrame()
        # rgb_frame = cv2.imread("/home/galois/projects/robothon23/src/Robothon2023/robothon2023_vision/to_test/frame_1.png")
        if self.test:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}.png")

        screen = rgb_frame[360:425, 660:810]

        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        _, saturation, _ = cv2.split(hsv)
        # cv2.imshow('saturation', saturation)
        _, thresh = cv2.threshold(saturation, 100, 255, 0)
        # Set image contour white
        thresh[0, :] = 255
        thresh[-1, :] = 255
        thresh[:, -1] = 255
        thresh[:, 0] = 255

        thresh[30:, :] = 255
        cv2.imshow('thresh', thresh)
        cv2.imshow("saturation", saturation)
        cv2.imshow("thresh", thresh)
        cv2.imshow("img acq", rgb_frame)

        cv2.waitKey(-1)
        cv2.destroyAllWindows()

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centri = []
        screen_with_centers = screen.copy()
        right_area = 0
        for idx, cnt in enumerate(contours):
            # epsilon = 0.01 * cv2.arcLength(cnt, True)
            # cnt = cv2.approxPolyDP(cnt, epsilon, True)
            area = cv2.contourArea(cnt)
            print(area)
            if (area > 1000) or (area < 130):
                continue
            right_area = area
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Centro triangolo sopra: (cx,cy) {cX, cY}")
            # print((cX,cY))
            cv2.circle(screen_with_centers, (cX, cY), 3, (0, 0, 255), -1)
            centri.append([cX, cY])

        print(GREEN + f"Identified triangles in: {centri}" + END)
        target_mm = 0
        if len(centri) == 2:
            diff = np.zeros(2)
            for id, centro in enumerate(centri):
                # diff[id] = np.linalg.norm(np.array(centro) - np.array(self.start_triangle_pos))
                diff[id] = np.linalg.norm(centro[0] - 76)
            print(diff)
            id_target = np.argmax(diff)
            target_center = centri[id_target]
            centri.pop(id_target)
            print(f"Old center: {centri[0]}")
            print(f"Target to reach: {target_center}")
            diff = centri[0][0] - target_center[0]  # last [0] is cx
            print(f"Centers Distance:  {diff}")
            self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}_after.png")
            target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff
            print(type(target_mm))
            print(target_mm)
            target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff  # + 0.1 * float(np.sign(diff))
            print(type(target_mm))
            print(target_mm)
        elif len(centri) == 1 and centri[0][0] > 30 and centri[0][0] < 110:
            rospy.loginfo(GREEN + "Only one triangle identified" + END)
            rospy.loginfo(GREEN + f"Area triangle: {right_area}" + END)
            if right_area < 600:
                target_mm = 0.0
            else:
                # target_mm = 0.01
                middle = int(thresh.shape[1] / 2.0)
                n_div = 40
                k_to_test = list(range(middle - int(middle / 2.0), middle + int(middle / 2.0)))
                # thresh_test = thresh.copy()
                # print(list(range(middle - int(middle/2.0) , middle + int(middle/2.0))))
                # print(thresh.shape)
                diff_areas = np.zeros(len(k_to_test))

                for id_div, k in enumerate(k_to_test):
                    thresh_test = thresh.copy()
                    thresh_test[:, k] = 255
                    contours, hierarchy = cv2.findContours(thresh_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    areas = []
                    centri = []

                    for idx, cnt in enumerate(contours):
                        area = cv2.contourArea(cnt)
                        print(area)
                        if (area > 1000) or (area < 70):
                            continue
                        M = cv2.moments(cnt)
                        # cX = int(M["m10"] / M["m00"])
                        # cY = int(M["m01"] / M["m00"])
                        # print(f"Centro triangolo sopra: (cx,cy) {cX, cY}")
                        # print((cX,cY))
                        cv2.circle(screen_with_centers, (cX, cY), 3, (0, 0, 255), -1)
                        centri.append([cX, cY])
                        areas.append(area)
                    print(centri)
                    if len(centri) == 2:
                        print(len(areas))
                        print(areas)
                        diff_areas[id_div] = np.linalg.norm(areas[0] - areas[1])
                    else:
                        diff_areas[id_div] = np.Inf
                id_best = np.argmin(diff_areas)
                k_best = k_to_test[id_best]
                print(f"Best id: {id_best}")
                print(f"k: {k_best}")

                thresh_test[:, k_best] = 255
                cv2.imshow("k_best", thresh_test)
                cv2.waitKey(-1)
                cv2.destroyAllWindows()
                contours, hierarchy = cv2.findContours(thresh_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                centri = []
                for idx, cnt in enumerate(contours):
                    # epsilon = 0.01 * cv2.arcLength(cnt, True)
                    # cnt = cv2.approxPolyDP(cnt, epsilon, True)
                    area = cv2.contourArea(cnt)
                    print(area)
                    if (area > 1000) or (area < 130):
                        continue
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print(f"Centro triangolo: (cx,cy) {cX, cY}")
                    # print((cX,cY))
                    cv2.circle(screen, (cX, cY), 3, (0, 0, 255), -1)
                    centri.append([cX, cY])
                cv2.imshow("screen", screen)
                cv2.waitKey(-1)
                cv2.destroyAllWindows()

                if len(centri) == 2:

                    diff = np.zeros(2)
                    for id, centro in enumerate(centri):
                        # diff[id] = np.linalg.norm(np.array(centro) - np.array(self.start_triangle_pos))
                        diff[id] = np.linalg.norm(centro[0] - 76)
                    print(diff)
                    id_target = np.argmax(diff)
                    target_center = centri[id_target]
                    centri.pop(id_target)
                    print(f"Old center: {centri[0]}")
                    print(f"Target to reach: {target_center}")
                    diff = centri[0][0] - target_center[0]  # last [0] is cx
                    print(f"Centers Distance:  {diff}")
                    self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}_after.png")
                    target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff
                    print(type(target_mm))
                    print(target_mm)
                    target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff  # + 0.1 * float(np.sign(diff))
                    print(type(target_mm))
                    print(target_mm)
                else:
                    rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, 0.0, 0.0])
                    rospy.set_param("/RL_params/slider/match_second_triangle/traslation", [0.0, 0.0, 0.0])
                    return TriggerResponse(True, NOT_SUCCESSFUL)
        else:
            target_mm = 0.0
            rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, 0.0, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle/traslation", [0.0, 0.0, 0.0])

            rospy.loginfo(YELLOW + "Target reached automatically. No triangles detected" + END)
            return TriggerResponse(True, str(target_mm))

        rospy.loginfo(GREEN + f"Target relative position {target_mm} " + END)
        target_m = target_mm / 1000.0
        if np.abs(target_mm - 0.0) < 1e-5:
            target_m_min = target_m - 1e-3
            target_m_max = target_m + 2e-3
        else:
            target_m_min = target_m - float(np.sign(target_m)) * 1e-3
            target_m_max = target_m + 2 * float(np.sign(target_m)) * 1e-3

        target_m_max = float(np.sign(target_m_max) * min([np.abs(target_m_max), 8.5 * 1e-3]))
        target_m_min = float(np.sign(target_m_min) * min([np.abs(target_m_min), 8.5 * 1e-3]))

        rospy.loginfo(GREEN + f"Target 1: {target_m_min}" + END)
        rospy.loginfo(GREEN + f"Target 2: {target_m_max}" + END)

        rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, target_m_min, 0.0])
        rospy.set_param("/RL_params/slider/match_second_triangle/traslation", [0.0, target_m_max, 0.0])

        cv2.imshow("screen", screen)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        return TriggerResponse(True, str(target_mm))

    def screen_target_1_new(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME_SCREEN_TARGET_1))
        rospy.loginfo(GREEN + f"Init display pos " + END)

        self.realsense.acquireOnceBoth()
        rgb_frame = self.realsense.getColorFrame()
        if self.test:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}.png")
        screen = rgb_frame[360:425, 660:848]
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        _, saturation, _ = cv2.split(hsv)
        # cv2.imshow('saturation', saturation)
        _, thresh = cv2.threshold(saturation, 55, 255, 0)
        # Set image contour white
        thresh[-1, :] = 255
        thresh[:, -1] = 255
        thresh[30:, :] = 255
        # cv2.imshow('thresh', thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centri = []
        for idx, cnt in enumerate(contours):
            # epsilon = 0.01 * cv2.arcLength(cnt, True)
            # cnt = cv2.approxPolyDP(cnt, epsilon, True)
            area = cv2.contourArea(cnt)
            print(area)
            if (area > 1000) or (area < 110):
                continue
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Centro triangolo sopra: (cx,cy) {cX, cY}")
            cv2.circle(screen, (cX, cY), 3, (0, 0, 255), -1)
            centri.append([cX, cY])
        # upper = None
        # lower = None
        # if len(centri) == 2:
        #     if centri[0][1]> centri[1][1]:
        #         upper = centri[1]
        #         lower = centri[0]
        #     else:
        #         upper = centri[0]
        #         lower = centri[1]
        self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}_init.png")
        if len(centri) == 1:
            self.start_triangle_pos = centri[0]
            self.triangle_mask = thresh
        else:
            pass
            # TODO:
        # cv2.imshow("screen", screen)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()

        print(GREEN + f"Identified triangle in: {self.start_triangle_pos}" + END)
        return TriggerResponse(True, SUCCESSFUL)

    def screen_target_2_new(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME_SCREEN_TARGET_2))

        if self.start_triangle_pos is None:
            rospy.loginfo(RED + f"start_triangle_pos is None" + END)
        print(f"Previous identification: {self.start_triangle_pos}")
        rospy.sleep(1)
        self.realsense.acquireOnceBoth()
        rgb_frame = self.realsense.getColorFrame()
        # rgb_frame = cv2.imread("/home/galois/projects/robothon23/src/Robothon2023/robothon2023_vision/to_test/frame_1.png")
        if self.test:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.realsense.saveAquiredImage(f"{self.folder_path}screen2_{now}.png")

        screen = rgb_frame[360:425, 660:848]
        # cv2.imshow("rgb_frame",rgb_frame)
        # cv2.imshow("screen", screen)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()

        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        _, saturation, _ = cv2.split(hsv)
        # cv2.imshow('saturation', saturation)
        _, thresh = cv2.threshold(saturation, 110, 255, 0)
        # Set image contour white
        thresh[0, :] = 255
        thresh[-1, :] = 255
        thresh[:, -1] = 255
        thresh[:, 0] = 255

        thresh[28:, :] = 255
        cv2.imwrite(f"{self.folder_path}screen2thresh_{now}.png", thresh)
        cv2.imwrite(f"{self.folder_path}screen2sat_{now}.png", saturation)
        cv2.imwrite(f"{self.folder_path}screen2rgb_frame_{now}.png", rgb_frame)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centri = []
        screen_with_centers = screen.copy()
        right_area = 0
        for idx, cnt in enumerate(contours):
            # epsilon = 0.01 * cv2.arcLength(cnt, True)
            # cnt = cv2.approxPolyDP(cnt, epsilon, True)
            area = cv2.contourArea(cnt)
            print(area)
            if (area > 1000) or (area < 130):
                continue
            right_area = area
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Centro triangolo sopra: (cx,cy) {cX, cY}")
            # print((cX,cY))
            cv2.circle(screen_with_centers, (cX, cY), 3, (0, 0, 255), -1)
            centri.append([cX, cY])
        # self.start_triangle_pos = (92, 25)
        # if (self.start_triangle_pos is None) and len(centri) != 2:
        #     rospy.loginfo(RED + "start_triangle_pos is none and len centers diff to 2" + END)
        #     return TriggerResponse(False, NOT_SUCCESSFUL)

        print(GREEN + f"Identified triangles in: {centri}" + END)
        target_mm = 0
        if len(centri) == 2:
            rospy.loginfo(GREEN + f"Identified 2 triangles " + END)

            diff = np.zeros(2)
            for id, centro in enumerate(centri):
                # diff[id] = np.linalg.norm(np.array(centro) - np.array(self.start_triangle_pos))
                diff[id] = np.linalg.norm(centro[0] - self.start_triangle_pos[0])
            print(diff)
            id_target = np.argmax(diff)
            target_center = centri[id_target]
            centri.pop(id_target)
            print(f"Old center: {centri[0]}")
            print(f"Target to reach: {target_center}")
            # diff = centri[0][0] - target_center[0]   #last [0] is cx
            diff = self.start_triangle_pos[0] - target_center[0]
            print(f"Centers Distance:  {diff}")
            self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}_after.png")
            target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff
            print(type(target_mm))
            print(target_mm)
            target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff  # + 0.1 * float(np.sign(diff))
            print(type(target_mm))
            print(target_mm)
        elif len(centri) == 1:
            rospy.loginfo(GREEN + "Only one triangle identified" + END)
            print(f"area: {right_area} ")
            if right_area < 450:
                target_mm = 0.0
            else:
                # target_mm = 0.01
                middle = int(thresh.shape[1] / 2.0)
                n_div = 40
                k_to_test = list(range(middle - int(middle / 2.0), middle + int(middle / 2.0)))
                # thresh_test = thresh.copy()
                # print(list(range(middle - int(middle/2.0) , middle + int(middle/2.0))))
                # print(thresh.shape)
                diff_areas = np.zeros(len(k_to_test))

                for id_div, k in enumerate(k_to_test):
                    thresh_test = thresh.copy()
                    thresh_test[:, k] = 255
                    contours, hierarchy = cv2.findContours(thresh_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    areas = []
                    centri = []

                    for idx, cnt in enumerate(contours):
                        area = cv2.contourArea(cnt)
                        print(area)
                        if (area > 1000) or (area < 70):
                            continue
                        M = cv2.moments(cnt)
                        # cX = int(M["m10"] / M["m00"])
                        # cY = int(M["m01"] / M["m00"])
                        # print(f"Centro triangolo sopra: (cx,cy) {cX, cY}")
                        # print((cX,cY))
                        cv2.circle(screen_with_centers, (cX, cY), 3, (0, 0, 255), -1)
                        centri.append([cX, cY])
                        areas.append(area)
                    print(centri)
                    if len(centri) == 2:
                        print(len(areas))
                        print(areas)
                        diff_areas[id_div] = np.linalg.norm(areas[0] - areas[1])
                    else:
                        diff_areas[id_div] = np.Inf
                id_best = np.argmin(diff_areas)
                k_best = k_to_test[id_best]
                print(f"Best id: {id_best}")
                print(f"k: {k_best}")

                thresh_test[:, k_best] = 255
                cv2.imwrite(f"{self.folder_path}screen2_kbest{now}.png", thresh_test)

                # cv2.imshow("k_best",thresh_test)
                # cv2.waitKey(-1)
                # cv2.destroyAllWindows()
                contours, hierarchy = cv2.findContours(thresh_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                centri = []
                for idx, cnt in enumerate(contours):
                    # epsilon = 0.01 * cv2.arcLength(cnt, True)
                    # cnt = cv2.approxPolyDP(cnt, epsilon, True)
                    area = cv2.contourArea(cnt)
                    print(area)
                    if (area > 1000) or (area < 130):
                        continue
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print(f"Centro triangolo: (cx,cy) {cX, cY}")
                    # print((cX,cY))
                    cv2.circle(screen, (cX, cY), 3, (0, 0, 255), -1)
                    centri.append([cX, cY])
                cv2.imwrite(f"{self.folder_path}screen2_detected{now}.png", thresh_test)

                # cv2.imshow("screen",screen)
                # cv2.waitKey(-1)
                # cv2.destroyAllWindows()

                if len(centri) == 2:

                    diff = np.zeros(2)
                    for id, centro in enumerate(centri):
                        # diff[id] = np.linalg.norm(np.array(centro) - np.array(self.start_triangle_pos))
                        # diff[id] = np.linalg.norm(centro[0] - 76)
                        diff[id] = np.linalg.norm(centro[0] - self.start_triangle_pos[0])
                    print(diff)
                    id_target = np.argmax(diff)
                    target_center = centri[id_target]
                    centri.pop(id_target)
                    print(f"Old center: {centri[0]}")
                    print(f"Target to reach: {target_center}")
                    # diff = centri[0][0] - target_center[0]  # last [0] is cx
                    diff = self.start_triangle_pos[0] - target_center[0]
                    print(f"Centers Distance:  {diff}")
                    self.realsense.saveAquiredImage(f"{self.folder_path}screen_{now}_after.png")
                    target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff
                    print(type(target_mm))
                    print(target_mm)
                    target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * diff  # + 0.1 * float(np.sign(diff))
                    print(type(target_mm))
                    print(target_mm)
                else:
                    return TriggerResponse(False, NOT_SUCCESSFUL)
        else:
            target_mm = 0.0
            rospy.loginfo(YELLOW + "Target reached automatically. No triangles detected" + END)
            return TriggerResponse(True, SUCCESSFUL)

        if np.abs(target_mm - 0.0) < 1e-5:
            rospy.loginfo(GREEN + f"Target relative position equal to 0{target_mm} " + END)

            rospy.loginfo(GREEN + f"Target relative position {target_mm} " + END)
            target_m = target_mm / 1000.0
            rospy.loginfo(GREEN + f"Target 1: {- 2 * 1e-3}" + END)
            rospy.loginfo(GREEN + f"Target 2: {+ 4 * 1e-3}" + END)

            rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation",
                            [0.0, - 2 * 1e-3, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle/traslation",
                            [0.0, 4 * 1e-3, 0.0])
        else:
            rospy.loginfo(GREEN + f"Target relative position {target_mm} " + END)
            target_m = target_mm / 1000.0
            rospy.loginfo(GREEN + f"Target 1: {target_m - float(np.sign(target_m)) * 1e-3}" + END)
            rospy.loginfo(GREEN + f"Target 2: {target_m + 2 * float(np.sign(target_m)) * 1e-3}" + END)
            # target_min = max([target_m - float(np.sign(target_m)) * 1e-3, ])
            # target_max = min([target_m + 2 *float(np.sign(target_m)) * 1e-3, ])
            rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation",
                            [0.0, target_m - float(np.sign(target_m)) * 1e-3, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle/traslation",
                            [0.0, target_m + 2 * float(np.sign(target_m)) * 1e-3, 0.0])
        cv2.imwrite(f"{self.folder_path}screen2_last_det{now}.png", screen)
        # cv2.imshow("screen", screen)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()
        return TriggerResponse(False, NOT_SUCCESSFUL)

    def new_approach(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format("new_approach"))
        self.realsense.acquireOnce()
        frame = self.realsense.getColorFrame()
        screen = frame[360:425, 660:848]
        now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        cv2.imwrite(f"{self.folder_path}/fast_screen/screen_{now}.png", frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(self.mask_generator)
        print("prima")
        print(self.device)
        t0 = rospy.Time.now().to_sec()
        # sam_result = self.mask_generator.generate(image_rgb)
        rospy.loginfo(f"Elapsed time for estimation ssm: {rospy.Time.now().to_sec() - t0}")
        print("dopo")
        # mask_annotator = sv.MaskAnnotator()

        # detections = sv.Detections.from_sam(sam_result=sam_result)
        # annotated_image = mask_annotator.annotate(scene=frame.copy(), detections=detections)

        # sv.plot_images_grid(
        #     images=[frame, annotated_image],
        #     grid_size=(1, 2),
        #     titles=['source image', 'segmented image']
        # )

        # usefull_masks = [segmentation for segmentation in sam_result if
        #                  segmentation["area"] > 280 and segmentation["area"] < 400]
        # if len(usefull_masks) == 3:
        #     centers = []
        #     for single_mask in usefull_masks:
        #         mask = single_mask["segmentation"].astype(np.uint8)
        #         contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #         contour = contours[0]
        #         len(contours)
        #         if len(contours) == 1:
        #             area = cv2.contourArea(contour)
        #             print(area)
        #         fake_image = np.zeros((*mask.shape[:2], 3), np.uint8)
        #         cv2.drawContours(fake_image, [contour], 0, (255, 255, 255), 1)
        #         M = cv2.moments(contour)
        #         cX_opencv = int(M["m10"] / M["m00"])
        #         cY_opencv = int(M["m01"] / M["m00"])
        #         cv2.circle(fake_image, (cX_opencv, cY_opencv), 1, color=(0, 0, 255), thickness=1)
        #         center = (cX_opencv,cY_opencv)
        #         centers.append(center)
        #         # sv.plot_image(fake_image)
        #         self.image_publisher.publish(self.bridge.cv2_to_imgmsg(fake_image))
        #     lower_triangle_id = np.argmax([center(1) for center in centers])
        #     lower_triangle_coords = centers.pop(lower_triangle_id)
        # target_triangle_id = np.argmax()

        # self.image_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_image))
        return TriggerResponse(True, SUCCESSFUL)

    def get_q(self, M):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX, cY)
        mu_20 = M["m20"] / M["m00"] - cX ** 2
        mu_02 = M["m02"] / M["m00"] - cY ** 2
        mu_11 = M["m11"] / M["m00"] - cX * cY
        mat_cov = np.array([[mu_20, mu_11], [mu_11, mu_02]])
        eigv = np.linalg.eig(mat_cov)
        eigvalues = eigv[0]
        # print(eigvalues)
        eccentricity = np.sqrt(1 - min(eigvalues) / (max(eigvalues)))
        return center, eccentricity

    def last_trial(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format("----- arrivata la richiesta -----"))
        # rospy.sleep(5)
        for k in range(5):
            frame = rospy.wait_for_message(COLOR_FRAME_TOPIC, sensor_msgs.msg.Image, timeout=None)
            img = self.bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")
        rospy.sleep(0.5)
        for k in range(5):
            frame = rospy.wait_for_message(COLOR_FRAME_TOPIC, sensor_msgs.msg.Image, timeout=None)
            img = self.bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")

            # img = self.realsense.getColorFrame()
            img_time = frame.header.stamp.secs + 1e-9 * frame.header.stamp.nsecs
            time_now = rospy.Time.now().to_sec()

        img = img[330:460, 630:870]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hue, saturation, value = cv2.split(hsv)
        l_col, a_col, b_col = cv2.split(lab)
        b, g, r = cv2.split(img)

        _, r_thresh = cv2.threshold(r, 150, 255, cv2.THRESH_TOZERO)
        _, g_thresh_0 = cv2.threshold(g, 150, 255, cv2.THRESH_TOZERO)

        # cv2.imshow("r_thread", r_thresh)
        # cv2.imshow("g_thresh_0", g_thresh_0)

        img_sub = cv2.subtract(r_thresh, g_thresh_0)

        contours, hierarchy = cv2.findContours(img_sub, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        right_cnt = []
        right_cnt_centers = []
        right_cnt_distance_center = []
        right_centers_h = []
        img_center = list(reversed(np.array(img.shape[:2]) / 2.0))

        no_red_triangle = False
        # print(img_center)
        # Lower triangle identification
        for cnt in contours:
            print(f"Area: {cv2.contourArea(cnt)}")
            if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 350:  # 200
                # cv2.drawContours(r_thresh, [cnt], 0, (255, 0, 0), -1)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                M = cv2.moments(cnt)
                center, eccentricity = self.get_q(M)
                perimeter = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                normalized_distance = np.linalg.norm(np.array(center) - np.array(img_center)) / img.shape[1]
                print(f"fArea : {area}")
                # print(f"Perimeter : {perimeter}")
                # print(f"Ration: {perimeter / area}")
                print(f"Center: {center}")
                print(f"Aspect ratio: {aspect_ratio}")
                print(f"Normalized distance: {normalized_distance}")
                if aspect_ratio > 0.7 and aspect_ratio < 1.5 and normalized_distance < 0.45:
                    right_cnt.append(cnt)
                    right_cnt_centers.append(center)
                    right_cnt_distance_center.append(np.linalg.norm(center - np.array(center)))
                    right_centers_h.append(h)
        if len(right_cnt_distance_center) == 1:
            print("One red triangle detected")
            id_lower_triangle = 0
            lower_triangle_cnt = right_cnt[id_lower_triangle]
            lower_triangle_center = right_cnt_centers[id_lower_triangle]
            lower_triangle_h = right_centers_h[0]
            cv2.circle(img, lower_triangle_center, 3, (0, 0, 255), -1)
            red_triangle = lower_triangle_center
            # cv2.drawContours(img, [lower_triangle_cnt], 0, (255, 0, 0), 3)
        elif len(right_cnt_distance_center) == 0:
            print("No lower red triangle detected")
            no_red_triangle = True
            # return TriggerResponse(False, NOT_SUCCESSFUL)
        else:
            print("More than one red triangle detected")
            return TriggerResponse(False, NOT_SUCCESSFUL)

        #################################################
        # Upper triangle identification
        print("-----")
        print("Upper triangles identification")
        # cv2.imshow("r_prev", r)
        if no_red_triangle:
            print("Default crop for upper triangle identification")
            r_thresh[round(img_center[1]):, :] = 0
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(r_thresh))
        else:
            r_thresh[round(lower_triangle_center[1] - lower_triangle_h / 2.0 * 0.8):, :] = 0

        contours, hierarchy = cv2.findContours(r_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        right_cnt_upper = []
        right_cnt_centers_upper = []
        right_cnt_distance_center_upper = []
        right_cnt_upper_w = []

        for cnt in contours:
            if 100 < cv2.contourArea(cnt) < 700:  # 200
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                M = cv2.moments(cnt)
                center, eccentricity = self.get_q(M)
                perimeter = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                normalized_distance = np.linalg.norm(np.array(center) - np.array(img_center)) / img.shape[1]
                print(f"fArea : {area}")
                print(f"Center: {center}")
                print(f"Aspect ratio: {aspect_ratio}")
                print(f"Normalized distance: {normalized_distance}")
                if 0.7 < aspect_ratio < 2.5 and normalized_distance < 0.45:
                    right_cnt_upper.append(cnt)
                    right_cnt_centers_upper.append(center)
                    right_cnt_distance_center_upper.append(np.linalg.norm(center - np.array(center)))
                    right_cnt_upper_w.append(w)
                    # cv2.circle(img, center, 3, (0, 0, 255), -1)

        _, g_thresh = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
        g_thresh[:40, :] = 0

        g_thresh[52:, :] = 0
        screen_blob, hierarchy = cv2.findContours(g_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        center_blob = None
        for cnt in screen_blob:
            area_blob = cv2.contourArea(cnt)
            if area_blob < 2000:
                continue
            print(f"Blob: {area_blob}")
            M_blob = cv2.moments(cnt)
            center_blob, _ = self.get_q(M_blob)
            print(center_blob)
            cv2.circle(img, center_blob, 3, (0, 0, 255), -1)
        if center_blob is None:
            center_blob = img_center

        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(img))
        rospy.sleep(0.5)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(g))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(r))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(r_thresh))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(g_thresh_0))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(img_sub))
        rospy.sleep(0.5)

        if no_red_triangle and len(right_cnt_centers_upper) >= 1:
            print("Rosso non identificato, Giallo/Verde si. Errore")
            return TriggerResponse(False, NOT_SUCCESSFUL)

        goal = 0.0
        if len(right_cnt_centers_upper) == 1 and aspect_ratio < 1.3:
            center_1 = right_cnt_centers_upper[0]
            cv2.circle(img, center_1, 3, (255, 0, 0), -1)
            goal = red_triangle[0] - center_1[0]
            # TODO: Caso iniziale
        elif len(right_cnt_centers_upper) == 1 and aspect_ratio >= 1.3:
            fake_center = right_cnt_centers_upper[0]
            w = right_cnt_upper_w[0]
            center_1 = (round(fake_center[0] - w / 2 + 8), fake_center[1])
            center_2 = (round(fake_center[0] + w / 2 - 8), fake_center[1])
            centers = [center_1, center_2]
            reference_triangle_id = np.argmin(np.linalg.norm(np.array(centers) - center_blob, axis=1))
            reference_triangle_center = centers[reference_triangle_id]
            centers.pop(reference_triangle_id)
            target_center = centers[0]
            cv2.circle(img, reference_triangle_center, 3, (255, 0, 0), -1)
            cv2.circle(img, target_center, 3, (0, 0, 255), -1)
            goal = reference_triangle_center[0] - target_center[0]
            # pass
        elif len(right_cnt_centers_upper) == 2:
            reference_triangle_id = np.argmin(np.linalg.norm(np.array(right_cnt_centers_upper) - center_blob, axis=1))
            reference_triangle_center = right_cnt_centers_upper[reference_triangle_id]
            right_cnt_centers_upper.pop(reference_triangle_id)
            target_center = right_cnt_centers_upper[0]
            cv2.circle(img, reference_triangle_center, 3, (255, 0, 0), -1)
            cv2.circle(img, target_center, 3, (0, 0, 255), -1)
            goal = reference_triangle_center[0] - target_center[0]
            # TODO: Calcola target
        elif len(right_cnt_centers_upper) > 2:
            return TriggerResponse(False, NOT_SUCCESSFUL)
            print("Più di due triangoli sopra identificati")
            # TODO: più di due triangoli
        else:
            if no_red_triangle:
                print("Nessun triangolo identificato. Fine")
                rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, 0.0, 0.0])
                rospy.set_param("/RL_params/slider/match_second_triangle/traslation", [0.0, 0.0, 0.0])
                rospy.set_param("/RL_params/slider/match_second_triangle_return/traslation", [0.0, 0.0, 0.0])
                return TriggerResponse(True, SUCCESSFUL)
            else:
                print("Rosso identificato, Giallo/Verde no. Errore")
                return TriggerResponse(False, NOT_SUCCESSFUL)
            # TODO: Fine ì
        print(GREEN + f"Goal: {goal}" + END)
        target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * goal  # + 0.1 * float(np.sign(diff))
        target_m = target_mm / 1000
        rospy.loginfo(GREEN + f"Target 1: {target_m - float(np.sign(target_m)) * 1e-3}" + END)
        rospy.loginfo(GREEN + f"Target 2: {target_m + 2 * float(np.sign(target_m)) * 1e-3}" + END)
        # target_min = max([target_m - float(np.sign(target_m)) * 1e-3, ])
        # target_max = min([target_m + 2 *float(np.sign(target_m)) * 1e-3, ])
        rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, target_m, 0.0])
        rospy.set_param("/RL_params/slider/match_second_triangle/traslation",
                        [0.0, 2 * float(np.sign(target_m)) * 1e-3, 0.0])
        rospy.set_param("/RL_params/slider/match_second_triangle_return/traslation",
                        [0.0, -2 * float(np.sign(target_m)) * 1e-3, 0.0])

        return TriggerResponse(True, SUCCESSFUL)

    def old_triangle_identification(self, image):
        img = image.copy()
        rospy.loginfo(SERVICE_CALLBACK.format("----- arrivata la richiesta -----"))
        reference_triangle_center = None
        target_triangle_center = None
        red_triangle_center = None
        red_triangle_2_center = None

        # img = img[330:460, 630:870]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hue, saturation, value = cv2.split(hsv)
        l_col, a_col, b_col = cv2.split(lab)
        b, g, r = cv2.split(img)

        _, r_thresh = cv2.threshold(r, 150, 255, cv2.THRESH_TOZERO)
        _, g_thresh_0 = cv2.threshold(g, 150, 255, cv2.THRESH_TOZERO)

        # cv2.imshow("r_thread", r_thresh)
        # cv2.imshow("g_thresh_0", g_thresh_0)

        img_sub = cv2.subtract(r_thresh, g_thresh_0)

        contours, hierarchy = cv2.findContours(img_sub, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        right_cnt = []
        right_cnt_centers = []
        right_cnt_distance_center = []
        right_centers_h = []
        img_center = list(reversed(np.array(img.shape[:2]) / 2.0))

        no_red_triangle = False
        # print(img_center)
        # Lower triangle identification
        for cnt in contours:
            # print(f"Area: {cv2.contourArea(cnt)}")
            if cv2.contourArea(cnt) > 100 and cv2.contourArea(cnt) < 350:  # 200
                # cv2.drawContours(r_thresh, [cnt], 0, (255, 0, 0), -1)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                M = cv2.moments(cnt)
                center, eccentricity = self.get_q(M)
                perimeter = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                normalized_distance = np.linalg.norm(np.array(center) - np.array(img_center)) / img.shape[1]
                # print(f"fArea : {area}")
                # # print(f"Perimeter : {perimeter}")
                # # print(f"Ration: {perimeter / area}")
                # print(f"Center: {center}")
                # print(f"Aspect ratio: {aspect_ratio}")
                # print(f"Normalized distance: {normalized_distance}")
                if aspect_ratio > 0.7 and aspect_ratio < 1.5 and normalized_distance < 0.45:
                    right_cnt.append(cnt)
                    right_cnt_centers.append(center)
                    right_cnt_distance_center.append(np.linalg.norm(center - np.array(center)))
                    right_centers_h.append(h)
        if len(right_cnt_distance_center) == 1:
            # print("One red triangle detected")
            id_lower_triangle = 0
            lower_triangle_cnt = right_cnt[id_lower_triangle]
            lower_triangle_center = right_cnt_centers[id_lower_triangle]
            lower_triangle_h = right_centers_h[0]
            cv2.circle(img, lower_triangle_center, 3, (0, 0, 255), -1)
            red_triangle = lower_triangle_center
            reference_triangle_center = red_triangle
            # cv2.drawContours(img, [lower_triangle_cnt], 0, (255, 0, 0), 3)
        elif len(right_cnt_distance_center) == 0:
            # print("No lower red triangle detected")
            no_red_triangle = True
            # return TriggerResponse(False, NOT_SUCCESSFUL)
        else:
            print("More than one red triangle detected")
            red_triangle_center = -1
            red_triangle_2_center = -1
            return red_triangle_center, red_triangle_2_center, reference_triangle_center, target_triangle_center
            # return TriggerResponse(False, NOT_SUCCESSFUL)

        #################################################
        # Upper triangle identification
        print("-----")
        print("Upper triangles identification")
        # cv2.imshow("r_prev", r)
        if no_red_triangle:
            # print("Default crop for upper triangle identification")
            r_thresh[round(img_center[1]):, :] = 0
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(r_thresh))
        else:
            r_thresh[round(lower_triangle_center[1] - lower_triangle_h / 2.0 * 0.8):, :] = 0

        contours, hierarchy = cv2.findContours(r_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        right_cnt_upper = []
        right_cnt_centers_upper = []
        right_cnt_distance_center_upper = []
        right_cnt_upper_w = []

        for cnt in contours:
            if 100 < cv2.contourArea(cnt) < 700:  # 200
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                M = cv2.moments(cnt)
                center, eccentricity = self.get_q(M)
                perimeter = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                normalized_distance = np.linalg.norm(np.array(center) - np.array(img_center)) / img.shape[1]
                print(f"fArea : {area}")
                print(f"Center: {center}")
                print(f"Aspect ratio: {aspect_ratio}")
                print(f"Normalized distance: {normalized_distance}")
                if 0.7 < aspect_ratio < 2.5 and normalized_distance < 0.45:
                    right_cnt_upper.append(cnt)
                    right_cnt_centers_upper.append(center)
                    right_cnt_distance_center_upper.append(np.linalg.norm(center - np.array(center)))
                    right_cnt_upper_w.append(w)
                    # cv2.circle(img, center, 3, (0, 0, 255), -1)

        _, g_thresh = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
        g_thresh[:40, :] = 0

        g_thresh[52:, :] = 0
        screen_blob, hierarchy = cv2.findContours(g_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        center_blob = None
        for cnt in screen_blob:
            area_blob = cv2.contourArea(cnt)
            if area_blob < 2000:
                continue
            print(f"Blob: {area_blob}")
            M_blob = cv2.moments(cnt)
            center_blob, _ = self.get_q(M_blob)
            print(center_blob)
            cv2.circle(img, center_blob, 3, (0, 0, 255), -1)
        if center_blob is None:
            center_blob = img_center

        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(img))
        rospy.sleep(0.5)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(g))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(r))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(r_thresh))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(g_thresh_0))
        rospy.sleep(0.5)
        # input("Avanti ...")
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(img_sub))
        rospy.sleep(0.5)

        if no_red_triangle and len(right_cnt_centers_upper) >= 1:
            red_triangle = None
        else:
            goal = 0.0
            if len(right_cnt_centers_upper) == 1 and aspect_ratio < 1.3:
                center_1 = right_cnt_centers_upper[0]
                cv2.circle(img, center_1, 3, (255, 0, 0), -1)
                goal = red_triangle[0] - center_1[0]
                target_triangle_center = center_1
                reference_triangle_center = red_triangle
                # TODO: Caso iniziale
            elif len(right_cnt_centers_upper) == 1 and aspect_ratio >= 1.3:
                fake_center = right_cnt_centers_upper[0]
                w = right_cnt_upper_w[0]
                center_1 = (round(fake_center[0] - w / 2 + 8), fake_center[1])
                center_2 = (round(fake_center[0] + w / 2 - 8), fake_center[1])
                centers = [center_1, center_2]
                reference_triangle_id = np.argmin(np.linalg.norm(np.array(centers) - center_blob, axis=1))
                reference_triangle_center = centers[reference_triangle_id]
                centers.pop(reference_triangle_id)
                target_center = centers[0]
                cv2.circle(img, reference_triangle_center, 3, (255, 0, 0), -1)
                cv2.circle(img, target_center, 3, (0, 0, 255), -1)
                goal = reference_triangle_center[0] - target_center[0]
                target_triangle_center = target_center

                # pass
            elif len(right_cnt_centers_upper) == 2:
                reference_triangle_id = np.argmin(
                    np.linalg.norm(np.array(right_cnt_centers_upper) - center_blob, axis=1))
                reference_triangle_center = right_cnt_centers_upper[reference_triangle_id]
                right_cnt_centers_upper.pop(reference_triangle_id)
                target_center = right_cnt_centers_upper[0]
                cv2.circle(img, reference_triangle_center, 3, (255, 0, 0), -1)
                cv2.circle(img, target_center, 3, (0, 0, 255), -1)
                goal = reference_triangle_center[0] - target_center[0]
                target_triangle_center = target_center
                reference_triangle_center = reference_triangle_center
                # TODO: Calcola target
            elif len(right_cnt_centers_upper) > 2:
                print("Più di due triangoli sopra identificati")
                # TODO: più di due triangoli

        return red_triangle_center, red_triangle_2_center, reference_triangle_center, target_triangle_center

    def triangle_identification(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format("----- arrivata la richiesta -----"))

        time_now = rospy.Time.now().to_sec()
        # print(f"Now: {time_now}")
        img_time = 0.0
        while img_time < time_now + 1:
            frame = rospy.wait_for_message(COLOR_FRAME_TOPIC, sensor_msgs.msg.Image, timeout=None)
            img = self.bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")
            img_time = frame.header.stamp.secs + 1e-9 * frame.header.stamp.nsecs
            # print(f"Image time: {img_time}")
        if self.test:
            now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # cv2.imwrite(f"{self.folder_path}new_screen_{now}.png", img)

        # red_triangle_center, red_triangle_2_center, reference_triangle_center, target_triangle_center = self.old_triangle_identification(img)
        # print(f"Old identification, red 1: {red_triangle_center}")
        # print(f"Old identification, red 2: {red_triangle_2_center}")
        # print(f"Old identification, ref 1: {reference_triangle_center}")
        # print(f"Old identification, tar 1: {target_triangle_center}")

        print(type(img))
        mask = img.copy()
        mask[:, :, :] = 0
        mask[300:550, 600:1000, :] = img[300:550, 600:1000, :]
        # cv2.imwrite(f"{self.folder_path}new_screen_{now}.png", mask)
        # Preprocess image
        lb_color = [0, 230, 245]
        ub_color = [180, 255, 255]

        dist_percent = 20
        h_triangle = 20
        w_triangle = 18
        # print(type(mask))
        img = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # red = []
        # yellow = []
        # green = []
        # try:
        #     identification(img, red, yellow, green, self.folder_path)
        # except Exception:
        #     rospy.info(YELLOW + "Function identifiction failed" + END)
        # print(type(img))
        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(opencv_image))

        img = np.asarray(img)
        # img = PIL.Image.fromarray(img)
        screen_mask, edges, bounds, failed = preprocess_image_auto(img, lb_color, ub_color)
        # rospy.sleep(1)
        # cv2.imwrite(f"{self.folder_path}new_screen_mask_{now}.png", screen_mask)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(screen_mask))
        # rospy.sleep(1)
        if failed:
            print("No screen detected")
            # TODO: Check

        # Find centroids
        img_final, red_triangle_center, red_triangle_2_center, reference_triangle_center, target_triangle_center = find_centroids_independent(
            edges, dist_percent,
            h_triangle, w_triangle)
        print(f"Red triangle 1: {red_triangle_center}")
        print(f"Red triangle 2: {red_triangle_2_center}")
        print(f"Yellow triangle: {reference_triangle_center}")
        print(f"Green triangle: {target_triangle_center}")

        # Overlay centroids on original image
        if reference_triangle_center:
            h_target_1 = reference_triangle_center[1] + bounds[1]
            w_target_1 = reference_triangle_center[0] + bounds[0]
            cv.drawMarker(img, (h_target_1, w_target_1), (0, 200, 0), cv.MARKER_TRIANGLE_DOWN, 6, 3)

        if target_triangle_center:
            h_target_2 = target_triangle_center[1] + bounds[1]
            w_target_2 = target_triangle_center[0] + bounds[0]
            cv.drawMarker(img, (h_target_2, w_target_2), (100, 100, 0), cv.MARKER_TRIANGLE_DOWN, 6, 3)

        if red_triangle_center:
            h_actual_1 = red_triangle_center[1] + bounds[1]
            w_actual_1 = red_triangle_center[0] + bounds[0]
            cv.drawMarker(img, (h_actual_1, w_actual_1), (0, 0, 0), cv.MARKER_TRIANGLE_UP, 6, 3)

        if red_triangle_2_center:
            h_actual_2 = red_triangle_2_center[1] + bounds[1]
            w_actual_2 = red_triangle_2_center[0] + bounds[0]
            cv.drawMarker(img, (h_actual_2, w_actual_2), (0, 100, 100), cv.MARKER_TRIANGLE_UP, 6, 3)
        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(opencv_image))

        if red_triangle_center and red_triangle_2_center:
            print("More than one red triangle detected")
            rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, 0, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle/traslation", [0.0, 0, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle_return/traslation", [0.0, 0, 0.0])
            return TriggerResponse(False, NOT_SUCCESSFUL)
        elif red_triangle_center and target_triangle_center:
            print("Red triangle and target triangle detected")
            goal = red_triangle_center[1] - target_triangle_center[1]
        elif red_triangle_center and reference_triangle_center and (target_triangle_center is None):
            print("Red triangle and reference triangle detected")
            goal = red_triangle_center[1] - reference_triangle_center[1]
        elif red_triangle_center and (reference_triangle_center is None) and (target_triangle_center is None):
            print("Only red triangle detected")
            rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, 0, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle/traslation", [0.0, 0, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle_return/traslation", [0.0, 0, 0.0])
            return TriggerResponse(False, NOT_SUCCESSFUL)
        elif (red_triangle_center is None) and (reference_triangle_center is None) and (target_triangle_center is None):
            print("Triangle already matched")
            rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, 0, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle/traslation", [0.0, 0, 0.0])
            rospy.set_param("/RL_params/slider/match_second_triangle_return/traslation", [0.0, 0, 0.0])
            return TriggerResponse(True, SUCCESSFUL)

        elif red_triangle_center is None:
            print("Red triangle not detected")
            return TriggerResponse(False, NOT_SUCCESSFUL)

        print(GREEN + f"Goal: {goal}" + END)
        target_mm = SLIDER_TRAVEL_MM / SLIDER_TRAVEL_PX * goal  # + 0.1 * float(np.sign(diff))
        target_m = target_mm / 1000
        rospy.loginfo(GREEN + f"Target 1: {target_m - float(np.sign(target_m)) * 1e-3}" + END)
        rospy.loginfo(GREEN + f"Target 2: {target_m + 2 * float(np.sign(target_m)) * 1e-3}" + END)
        rospy.set_param("/RL_params/slider/match_second_triangle_approach/traslation", [0.0, target_m, 0.0])
        rospy.set_param("/RL_params/slider/match_second_triangle/traslation",
                        [0.0, 2 * float(np.sign(target_m)) * 1e-3, 0.0])
        rospy.set_param("/RL_params/slider/match_second_triangle_return/traslation",
                        [0.0, -2 * float(np.sign(target_m)) * 1e-3, 0.0])
        return TriggerResponse(False, NOT_SUCCESSFUL)

    def align_screen(self, req):
        found = False
        while not found:
            try:
                (trans, rot) = self.listener.lookupTransform('reference', 'open_tip', rospy.Time(0))
                found = True
                print("Retrieved open_tip -> base_link")
            except (tf.LookupException, tf.ConnectivityException):
                rospy.loginfo(YELLOW + "Unable to retrieve tf-t between: Retrieved open_tip -> base_link" + END)
                rospy.sleep(0.1)
        translation_b_op = tf.transformations.translation_matrix(trans)
        rotation_b_op = tf.transformations.quaternion_matrix(rot)
        rospy.loginfo(GREEN + f"Translation open_tip -> base_link: {trans}" + END)
        rospy.loginfo(GREEN + f"Orientation open_tip -> base_link: {rot}" + END)

        rospy.loginfo(GREEN + f"Translation open_tip -> base_link: {translation_b_op}" + END)
        rospy.loginfo(GREEN + f"Orientation open_tip -> base_link: {rotation_b_op}" + END)
        mismatched_angle = (np.arctan2(rotation_b_op[1, 0], rotation_b_op[0, 0]) - np.pi / 2)
        rospy.loginfo(GREEN + f"Mismatched angle: {mismatched_angle * 180.0 / np.pi}" + END)

        mismatched_angle_as_quat = [0.0, 0.0, sin(-mismatched_angle / 2.0), cos(-mismatched_angle / 2.0)]
        rospy.set_param("/RL_params/slider/align_slider/rotation",
                        mismatched_angle_as_quat)
        rospy.loginfo(GREEN + f"Mismatched angle quat: {mismatched_angle_as_quat}" + END)
        return TriggerResponse(True, NOT_SUCCESSFUL)


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
        weight_path_ssm = rospy.get_param("~weight_path_ssm")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("weight_path_ssm") + END)
        return 0
    try:
        labels_path = rospy.get_param("~labels_path")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("labels_path") + END)
        return 0

    vision_system = BoardLocalization(test, images_path, weight_path, weight_path_ssm, labels_path)

    rospy.Service(SERVICE_NAME_BOARD, Trigger, vision_system.board_localization)
    rospy.Service(SERVICE_NAME_SCREEN_TARGET_1, Trigger, vision_system.triangle_identification)
    rospy.Service(SERVICE_NAME_SCREEN_TARGET_2, Trigger, vision_system.triangle_identification)
    rospy.Service(ALIGNMENT_SCREEN_TOPIC, Trigger, vision_system.align_screen)

    rospy.spin()


if __name__ == "__main__":
    main()
