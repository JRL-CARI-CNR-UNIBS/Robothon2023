#! /usr/bin/env python3
import copy

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, Transform, Vector3, TransformStamped
from RealSense import RealSense
from scipy.spatial.transform import Rotation as R

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

from Inference import OnlineInference
import multiprocessing

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
SERVICE_NAME_REF_PARAMS = "/robothon2023/transferability/set_ref_params"


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
                                                       img_size=[580, 230],
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

            features = {"yellow_coil": [],
                        "black_coil": [],
                        "heat_sink": []}

            inference_result, img_with_box, _ = self.online_circuit_inference.realtime_inference(frame, conf_thres=0.4,
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
                        if len(features[class_name]) >= 2:
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
                base_frame = "base_link"
                (trans, rot) = self.listener.lookupTransform(base_frame, 'camera_color_optical_frame', rospy.Time(0))
                found = True
                print(f"Retrieved camera_color_optical_frame -> {base_frame}")
            except (tf.LookupException, tf.ConnectivityException):
                rospy.loginfo(
                    YELLOW + f"Unable to retrieve tf-t between: camera_color_optical_frame -> {base_frame}" + END)
                rospy.sleep(0.1)

        rospy.loginfo(YELLOW + "Trasformata camera_link -> {} \n :{}".format(base_frame, trans) + RED)
        rospy.loginfo(YELLOW + "Trasformata camera_link -> {} \n :{}".format(base_frame, rot) + RED)

        trans_world_camera = tf.transformations.translation_matrix(trans)
        rot_world_camera = tf.transformations.quaternion_matrix(rot)
        M_world_camera = np.dot(trans_world_camera, rot_world_camera)

        ref_yellow_world = np.dot(M_world_camera, self.get4Vector(reference_yellow_circuit))[0:-1]
        ref_yellow_world[-1] = 0.957
        ref_green_world = np.dot(M_world_camera, self.get4Vector(reference_green_circuit))[0:-1]
        ref_green_world[-1] = 0.957
        yellow_coil_world = np.dot(M_world_camera, self.get4Vector(self.yellow_coil_camera))[0:-1]
        yellow_coil_world[-1] = 0.957
        heat_sink_world = np.dot(M_world_camera, self.get4Vector(self.heat_sink_camera))[0:-1]
        heat_sink_world[-1] = 0.957

        print(RED + "Check z axis pos" + END)
        print(GREEN + f"Yellow: {ref_yellow_world}" + END)
        print(GREEN + f"Green: {ref_green_world}" + END)
        print(GREEN + f"Yellow coil: {yellow_coil_world}" + END)
        print(GREEN + f"Heat sink: {heat_sink_world}" + END)

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

        static_transform_circuit = self.getStaticTrasformStamped(base_frame, "circuit",
                                                                 M_world_circuit[0:3, -1],
                                                                 rotation_quat)
        static_transform_yellow_coil = self.getStaticTrasformStamped(base_frame, "yellow_coil",
                                                                     yellow_coil_world[0:3],
                                                                     rotation_quat)
        static_transform_heat_circuit = self.getStaticTrasformStamped(base_frame, "heat_sink",
                                                                      heat_sink_world[0:3],
                                                                      rotation_quat)
        static_transform_green_circuit = self.getStaticTrasformStamped(base_frame, "green_circuit",
                                                                       ref_green_world[0:3],
                                                                       rotation_quat)
        static_transform_yellow_circuit = self.getStaticTrasformStamped(base_frame, "yellow_circuit",
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

    @staticmethod
    def plot_img(img):
        plt.figure(multiprocessing.current_process().name)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        plt.show()

    def digits_detection(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format("Digits detection"))

        for k in range(1):
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
            # print(img.shape)
            # img_bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # img_bw = img_bw.reshape([1, *img_bw.shape])
            # print(img_bw.shape)
            # mask = np.zeros((3,*img_bw.shape))
            # mask[:,:,0] = img_bw[:]
            # mask[:, :, 1] = img_bw[:]
            # mask[:, :, 2] = img_bw[:]
            # print(mask.shape)
            # img_bw = img_bw
            # # mask = np.array([img_bw]*3)
            # img[234:,:,:]=0
            # img[:234,:700,:]=0
            new_img = img[:230, 700:, :]
            print(new_img.shape)
            inference_result, img_with_box, _ = self.online_digits_inference.realtime_inference(new_img,
                                                                                                conf_thres=0.28,
                                                                                                iou_thres=0.5,
                                                                                                agnostic_nms=False,
                                                                                                max_det=10,
                                                                                                view_img=True)
            recognize_filtered_obj = []
            for recognize_object in inference_result:
                class_name = recognize_object["class_name"]
                xywh = recognize_object["xywh"]
                conf = recognize_object["conf"]
                # print(f"{class_name}: {conf}")
                if class_name != "-":
                    if len(recognize_filtered_obj) < 4:
                        recognize_filtered_obj.append(recognize_object)
                    else:
                        min_conf = 1
                        min_id = None
                        for id_obj, filtered_obj in enumerate(recognize_filtered_obj):
                            if filtered_obj["conf"] < min_conf:
                                min_conf = filtered_obj["conf"]
                                min_id = id_obj
                        if conf > min_conf:
                            print(YELLOW + f"Removed: {recognize_filtered_obj[min_id]}" + END)
                            recognize_filtered_obj.pop(min_id)
                            recognize_filtered_obj.append(recognize_object)
            print(recognize_filtered_obj)
            recognize_filtered_obj.sort(key=lambda obj: obj["xywh"][0])
            for recognized_object in recognize_filtered_obj:
                if recognized_object["class_name"] in ["2", "5"]:
                    x_left = round(recognized_object["xywh"][0] - recognized_object["xywh"][2] / 2.0)
                    x_right = round(recognized_object["xywh"][0] + recognized_object["xywh"][2] / 2.0)
                    y_up = round(recognized_object["xywh"][1] - recognized_object["xywh"][-1] / 2.0)
                    y_down = round(recognized_object["xywh"][1] + recognized_object["xywh"][-1] / 2.0)
                    crop_width = x_right - x_left
                    lower = y_up + round((y_down - y_up) * 0.15)
                    upper = y_up + round((y_down - y_up) * 0.45)

                    roi = new_img[lower:upper, x_left:x_right, :].copy()
                    roi_bw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, roi_thresh = cv2.threshold(roi_bw, 175, 255, cv.THRESH_TOZERO)

                    # roi_thresh_up = roi_thresh[:round(roi_thresh.shape[0] / 2.0 - edge_down), :]
                    roi_thresh[-1, :] = 0
                    roi_thresh[:, -1] = 0
                    roi_thresh[0, :] = 0
                    roi_thresh[:, 0] = 0
                    contours, hierarchy = cv2.findContours(roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    print(len(contours))
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
                    for cnt in contours:
                        cv2.drawContours(roi, [cnt], 0, (255, 0, 0), -1)

                    cnt = contours[-1]
                    area = cv2.contourArea(cnt)

                    print(f"Rectangulaer area {area}")
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print(f"Position: ({cX},{cY})")
                    print(round(crop_width / 2.0))

                    cv2.drawContours(roi, [cnt], 0, (255, 0, 0), -1)
                    cv2.circle(roi, (cX, cY), 3, (0, 0, 255), -1)

                    if cX > round(crop_width / 2.0):
                        recognized_object["class_name"] = "2"
                        print("è un 2")
                    else:
                        recognized_object["class_name"] = "5"
                        print("è un 5")

                    # p.start()
                    # cv2.waitKey(1)
                    # cv2.destroyAllWindows()

            inference_result, img_with_box, _ = self.online_digits_inference.realtime_inference(new_img,
                                                                                                conf_thres=0.01,
                                                                                                iou_thres=1,
                                                                                                agnostic_nms=False,
                                                                                                max_det=500,
                                                                                                view_img=False)
            pallino = None
            for recognized_object in inference_result:
                if recognized_object["class_name"] == "-":
                    x = recognized_object["xywh"][0]
                    y = recognized_object["xywh"][1]
                    if x > 264 and x < 470 and y > 137 and y < 171:
                        # cv2.circle(new_img, (x, y), 3, (0, 0, 255), -1)
                        print(recognized_object)
                        if pallino is None:
                            pallino = recognized_object
                        else:
                            if recognized_object["conf"] > pallino["conf"]:
                                pallino = recognized_object

            if pallino:
                pallino["class_name"] = "."
                cv2.circle(new_img, (pallino["xywh"][0], pallino["xywh"][1]), 3, (0, 0, 255), -1)

                recognize_filtered_obj.append(pallino)
                recognize_filtered_obj.sort(key=lambda obj: obj["xywh"][0])

            # ROBERTO: "STAVO MODIFICANDO QUI. Alla riga 492 ho messo la funzione di michele (per ora commentata)
            #           bisogna inserire il punto fra i recognized_objects
            #           TUTTAVIA, NELLE PROVE CHE HO FATTO IL PUNTO LO TROVA GIA, ANCHE SENZA QUESTO"
            # print(recognized_object)
            # center_list = self.dot_recognition(img)
            # if len(center_list) == 1:
            #     pallino = recognized_object
            #     pallino["class_name"] = "."

            string = ""
            for digit in recognize_filtered_obj:
                string += digit["class_name"]
            print(f"{RED}Recognize number: {string}{END}")
            if string[0] == ".":
                if len(string) > 1:
                    string = string[1:]
                else:
                    string = ""
            # if len(string) == 4 and string[0]=="1" and pallino is not None:
            #     string = string[:2] + "." + string[2:]
            # elif len(string) == 4 and string[0]=="0" and pallino is not None:
            #     string = string[:1] + "." + string[1:]
            if string == "1302":
                string = "13.02"
            cv2.putText(img, string, (50, 150), 3, 5, (0, 255, 0), 10, cv2.LINE_AA)
            p = multiprocessing.Process(target=copy.copy(TransferabilityVisionSystem.plot_img), args=(img,))
            p.start()

        return TriggerResponse(True, SUCCESSFUL)

    # def dot_recognition(self, img):
    #     # Read image
    #     # img = cv.imread('./multimeter_screen_img_raw/raw/frame_5.png')
    #     img = img[200:430, 300:420, :]
    #
    #     # Display original image
    #     cv.imshow('original image', img)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    #
    #     lower_val = np.array([200, 200, 80])
    #     upper_val = np.array([255, 255, 255])
    #     mask = cv.inRange(img, lower_val, upper_val)
    #
    #     # cv.imshow('color mask', mask)
    #     # cv.waitKey(0)
    #     # cv.destroyAllWindows()
    #
    #     kernel = np.ones((5, 5), np.uint8)
    #     mask_dilated = cv.dilate(mask, kernel, iterations=1)
    #
    #     cv.imshow('dilated mask', mask_dilated)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    #
    #     mask_contours = mask_dilated.copy()
    #     mask_contours = cv.cvtColor(mask_contours, cv.COLOR_GRAY2BGR)
    #
    #     contours, _ = cv.findContours(mask_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #     cnt = contours
    #
    #     contour_list = []
    #     center_list = []
    #     center_count = 0
    #     for contour in contours:
    #         approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
    #         area = cv.contourArea(contour)
    #         # Filter based on length and area
    #         if 30 < area < 150:
    #             print("found!")
    #             if center_count > 1:
    #                 print("Found more than one center")
    #             center_count = center_count + 1
    #             # print area
    #             contour_list.append(contour)
    #
    #             # compute the center of the contour
    #             M = cv.moments(contour)
    #             cX = int(M["m10"] / M["m00"])
    #             cY = int(M["m01"] / M["m00"])
    #             center_list.append((cX, cY))
    #
    #     for item in center_list:
    #         cv.circle(mask_contours, item, 2, (0, 0, 255), -1)
    #     cv.drawContours(mask_contours, contour_list, -1, (255, 0, 0), 2)
    #
    #     cv.imshow('contours on dilated mask', mask_contours)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    #
    #     return center_list

    def set_ref_param(self, req):
        found = False
        while not found:
            try:
                base_frame = "base"
                (trans, rot) = self.listener.lookupTransform(base_frame, 'circuit_reference', rospy.Time(0))
                found = True
            except (tf.LookupException, tf.ConnectivityException):
                rospy.loginfo(
                    YELLOW + f"Unable to retrieve tf-t between: circuit_reference -> {base_frame}" + END)
                rospy.sleep(0.1)
        trans_base_circuit = tf.transformations.translation_matrix(trans)
        rot_base_circuit = tf.transformations.quaternion_matrix(rot)
        M_world_circuit = np.dot(trans_base_circuit, rot_base_circuit)

        to_replace = ["REF_X", "REF_Y", "REF_Z", "REF_ROTX", "REF_ROTY", "REF_ROTZ",
                      "INV_REF_X", "INV_REF_Y", "INV_REF_Z", "INV_REF_ROTX", "INV_REF_ROTY", "INV_REF_ROTZ"]
        inv_matrix = np.linalg.inv(M_world_circuit)

        rot_as_axis_angle = R.from_quat(rot).as_rotvec()
        value_to_replace = [*M_world_circuit[0:3, -1],
                            *rot_as_axis_angle,
                            *inv_matrix[0:3, -1],
                            *R.from_matrix(inv_matrix[0:3, 0:3]).as_rotvec()]

        print(value_to_replace)
        # print(rot_as_axis_angle)

        for param_to_rep, value_to_rep in zip(to_replace, value_to_replace):
            rospy.set_param(f"RL_params/test_probe/resistance_test/{param_to_rep}", float(value_to_rep))
        return TriggerResponse(True, SUCCESSFUL)


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
    rospy.Service(SERVICE_NAME_DIGITS, Trigger, transferability_vision_system.digits_detection)
    rospy.Service(SERVICE_NAME_REF_PARAMS, Trigger, transferability_vision_system.set_ref_param)

    rospy.spin()


if __name__ == "__main__":
    main()
