#! /usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, Transform, Vector3, TransformStamped
from RealSense import RealSense

import tf2_ros
import tf
import geometry_msgs.msg
from utils_detection import *
import datetime
from dataclasses import dataclass, field
from typing import List, Tuple
import torch
import numpy as np
import os, math
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
import cv2
from slider_detect_triangles import *
from identification_function import *

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

SERVICE_NAME_CIRCUIT = "/robothon2023/transferability/circuit_localization"
SERVICE_NAME_DIGITS = "/robothon2023/transferability/digits_detection"


@dataclass
class TransferabilityVisionSystem:
    test: bool
    folder_path: str
    weight_path_circuit: str
    labels_path_circuit: str
    weight_path_digits: str
    labels_path_digits: str
    realsense: RealSense = field(init=False)
    broadcaster: tf2_ros.StaticTransformBroadcaster = field(init=False)
    img_size: List[int] = field(init=False)
    listener: tf.TransformListener = field(init=False)
    n_frame: int = field(default=0, init=False)
    first_identification: bool = field(init=False)

    def __post_init__(self):
        self.realsense = RealSense()

        # Retrieve camera parameters
        rospy.loginfo(YELLOW + "Waiting camera parameters ..." + END)
        self.realsense.getCameraParam()
        self.realsense.waitCameraInfo()
        rospy.loginfo(GREEN + "Camera parameters retrived correctly" + END)

        # Tf Broadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.listener = tf.TransformListener()

        self.img_size = [1280, 720]

        self.image_publisher = rospy.Publisher("transferability_detected_image", sensor_msgs.msg.Image, queue_size=50)
        self.bridge = CvBridge()

        self.online_circuit_inference = OnlineInference(weights=self.weight_path_circuit,
                                                        device=0,
                                                        yaml=self.labels_path_circuit,
                                                        img_size=[1280, 720],
                                                        half=False)
        self.online_digits_inference = OnlineInference(weights=self.weight_path_digits,
                                                       device=0,
                                                       yaml=self.labels_path_digits,
                                                       img_size=[1280, 720],
                                                       half=False)
        self.first_identification = True

        rospy.loginfo(GREEN + "Service alive ...." + END)

    def circuit_localization(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME_CIRCUIT))
        rospy.sleep(1)
        self.black_coil_mean_camera = None
        self.yellow_coil_camera = None
        self.heat_sink_camera = None

        n_yellow_coil = 0
        n_black_coil = 0
        n_heat_sink = 0
        max_trial = 20
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

            features = {"yellow_coil": [],
                        "black_coil": [],
                        "heat_sink": []}

            inference_result, img_with_box, _ = self.online_circuit_inference.realtime_inference(frame, conf_thres=0.5,
                                                                                                 iou_thres=0.3,
                                                                                                 agnostic_nms=False,
                                                                                                 max_det=100,
                                                                                                 view_img=True)
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(img_with_box))
            print("---------------- INFERENCE RESULT -----------")
            print(inference_result)
            for recognize_object in inference_result:
                class_name = recognize_object["class_name"]
                xywh = recognize_object["xywh"]
                conf = recognize_object["conf"]
                if class_name in features:
                    if class_name != "black_coil":
                        if features[class_name]:  # If already exist take the one with biggest conf
                            if conf > features[class_name][2]:
                                features[class_name] = [xywh[0], xywh[1], conf]
                        else:
                            features[class_name] = [xywh[0], xywh[1], conf]
                    else:
                        if len(features[class_name]) > 2:
                            min_conf = 1
                            min_id = None
                            for id_hs, old_heat_sink in enumerate(features[class_name]):
                                if old_heat_sink[2] < min_conf:
                                    min_conf = old_heat_sink[2]
                                    min_id = id_hs
                            if conf > min_conf:
                                print(YELLOW + f"Removed: {features[class_name][min_id]}" + END)
                                features[class_name].pop(min_id)
                                features[class_name].append((xywh[0], xywh[1], conf))
                        else:
                            features[class_name].append((xywh[0], xywh[1], conf))
            print(features)
            depth_frame = self.realsense.getDistanceFrame()

            print(f"Shape depth: {depth_frame.shape}")

            yellow_coil_camera = None
            black_coil_camera = None
            heat_sink_camera = None
            if features['yellow_coil']:
                n_yellow_coil += 1
                yellow_coil_camera = np.array(
                    self.realsense.deproject(features['yellow_coil'][0], features['yellow_coil'][1],
                                             depth_frame[
                                                 features['yellow_coil'][1], features['yellow_coil'][
                                                     0]])) / 1000.0
                if self.yellow_coil_camera is not None:
                    self.yellow_coil_camera = self.yellow_coil_camera + 1.0 / n_yellow_coil * (
                            yellow_coil_camera - self.yellow_coil_camera)
                else:
                    self.yellow_coil_camera = yellow_coil_camera
                print(f"Diff yellow coil: {yellow_coil_camera - self.yellow_coil_camera}")

            if features['heat_sink']:
                n_heat_sink += 1
                heat_sink_camera = np.array(
                    self.realsense.deproject(features['heat_sink'][0], features['heat_sink'][1],
                                             depth_frame[
                                                 features['heat_sink'][1], features['heat_sink'][
                                                     0]])) / 1000.0
                if self.heat_sink_camera is not None:
                    self.heat_sink_camera = self.heat_sink_camera + 1.0 / n_heat_sink * (
                            heat_sink_camera - self.heat_sink_camera)
                else:
                    self.heat_sink_camera = heat_sink_camera
                print(f"Diff heat sink: {heat_sink_camera - self.heat_sink_camera}")

            if len(features['black_coil']) == 2:
                n_black_coil += 1
                black_coil_camera_mean = None
                for black_coil in features["black_coil"]:
                    black_coil_camera = np.array(
                        self.realsense.deproject(black_coil[0], black_coil[1],
                                                 depth_frame[
                                                     black_coil[1], black_coil[
                                                         0]])) / 1000.0
                    if black_coil_camera_mean is None:
                        black_coil_camera_mean = black_coil_camera
                    else:
                        black_coil_camera_mean = black_coil_camera_mean + 1.0 / 2 * (
                                black_coil_camera - black_coil_camera_mean)
                if self.black_coil_mean_camera is not None:
                    self.black_coil_mean_camera = self.black_coil_mean_camera + 1.0 / n_black_coil * (
                            black_coil_camera_mean - self.black_coil_mean_camera)
                else:
                    self.black_coil_mean_camera = black_coil_camera_mean
                print(f"Diff black coil: {black_coil_camera - self.black_coil_mean_camera}")

            if self.first_identification is True and n_black_coil >= 1 and n_yellow_coil >= 1:
                self.first_identification = False
                break
            if k >= max_trial - 1:
                self.first_identification = True

        print(self.yellow_coil_camera)
        # print(self.black_coil_camera)

        if (self.yellow_coil_camera is None) or (self.black_coil_mean_camera is None) or (
                self.heat_sink_camera is None):
            print(RED + "Something not identified" + END)
            return TriggerResponse(False, NOT_SUCCESSFUL)

        print(YELLOW + f"Yellow coil found: {n_yellow_coil} " + END)
        print(YELLOW + f"Black coil found: {n_black_coil} " + END)
        print(YELLOW + f"Heat sink found: {n_heat_sink} " + END)

        reference_green_circuit = np.mean([self.yellow_coil_camera, self.heat_sink_camera], axis=0)
        reference_yellow_circuit = self.black_coil_mean_camera
        print(reference_green_circuit)

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

        ref_yellow_world = np.dot(M_world_camera, self.get4Vector(reference_yellow_circuit))[0:-1]
        ref_green_world = np.dot(M_world_camera, self.get4Vector(reference_green_circuit))[0:-1]
        yellow_coil_world = np.dot(M_world_camera, self.get4Vector(self.yellow_coil_camera))[0:-1]
        heat_sink_world = np.dot(M_world_camera, self.get4Vector(self.heat_sink_camera))[0:-1]

        z_axis = np.array([0.0, 0.0, -1.0])

        x_axis = (ref_green_world - ref_yellow_world) / np.linalg.norm(ref_green_world - ref_yellow_world)
        y_axis_first_approach = np.cross(z_axis, x_axis)
        y_axis_norm = y_axis_first_approach / np.linalg.norm(y_axis_first_approach)

        rot_mat_world_circuit = np.array([x_axis, y_axis_norm, z_axis]).T
        M_world_circuit_only_rot = tf.transformations.identity_matrix()
        M_world_circuit_only_rot[0:-1, 0:-1] = rot_mat_world_circuit

        M_world_circuit_only_tra = tf.transformations.identity_matrix()
        M_world_circuit_only_tra[0:3, -1] = ref_yellow_world

        M_world_circuit = np.dot(M_world_circuit_only_tra, M_world_circuit_only_rot)
        rotation_quat = tf.transformations.quaternion_from_matrix(M_world_circuit)

        static_transform_circuit = self.getStaticTrasformStamped("base_link", "circuit",
                                                                 M_world_circuit[0:3, -1],
                                                                 rotation_quat)
        static_transform_yellow_coil = self.getStaticTrasformStamped("base_link", "yellow_coil",
                                                                     yellow_coil_world[0:3],
                                                                     rotation_quat)
        static_transform_heat_circuit = self.getStaticTrasformStamped("base_link", "heat_sink",
                                                                      heat_sink_world[0:3],
                                                                      rotation_quat)
        static_transform_green_circuit = self.getStaticTrasformStamped("base_link", "green_circuit",
                                                                       ref_green_world[0:3],
                                                                       rotation_quat)
        static_transform_yellow_circuit = self.getStaticTrasformStamped("base_link", "yellow_circuit",
                                                                        ref_yellow_world[0:3],
                                                                        rotation_quat)

        self.broadcaster.sendTransform(
            [static_transform_circuit,
             static_transform_yellow_coil,
             static_transform_heat_circuit,
             static_transform_green_circuit,
             static_transform_yellow_circuit])
        rospy.loginfo(GREEN + "Published tf" + END)

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

    def digits_detecion(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format("Digits detection"))

        for k in range(50):
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

            inference_result, img_with_box, _ = self.online_digits_inference.realtime_inference(img, conf_thres=0.4,
                                                                                                iou_thres=0.1,
                                                                                                agnostic_nms=False,
                                                                                                max_det=100,
                                                                                                view_img=True)

        return TriggerResponse(False, NOT_SUCCESSFUL)


def main():
    rospy.init_node("transferability_vision_system")
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
        weight_path_circuit = rospy.get_param("~weight_path_circuit")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("weight_path_circuit") + END)
        return 0
    try:
        weight_path_digits = rospy.get_param("~weight_path_digits")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("weight_path_digits") + END)
        return 0
    try:
        labels_path_circuit = rospy.get_param("~labels_path_circuit")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("labels_path_circuit") + END)
        return 0
    try:
        labels_path_digits = rospy.get_param("~labels_path_digits")
    except KeyError:
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("labels_path_digits") + END)
        return 0

    transferability_vision_system = TransferabilityVisionSystem(test,
                                                                images_path,
                                                                weight_path_circuit,
                                                                labels_path_circuit,
                                                                weight_path_digits,
                                                                labels_path_digits)

    rospy.Service(SERVICE_NAME_CIRCUIT, Trigger, transferability_vision_system.circuit_localization)
    rospy.Service(SERVICE_NAME_DIGITS, Trigger, transferability_vision_system.digits_detecion)

    rospy.spin()


if __name__ == "__main__":
    main()
