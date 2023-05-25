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
SERVICE_NAME_SCREEN_TARGET = "/robothon2023/screen_target"
SERVICE_NAME_SCREEN_TARGET_INIT = "/robothon2023/screen_target_init"

SLIDER_TRAVEL_MM = 9.1  # 8.5 #29.0
SLIDER_TRAVEL_PX = 56  # 56  # 49 #256.0


@dataclass
class BoardLocalization:
    test: bool
    folder_path: str
    weight_path: str
    labels_path: str
    realsense: RealSense = field(init=False)
    broadcaster: tf2_ros.StaticTransformBroadcaster = field(init=False)
    img_size: List[int] = field(init=False)
    listener: tf.TransformListener = field(init=False)
    first_identification: bool = field(init=False)
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

        # Tf Broadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.listener = tf.TransformListener()

        self.img_size = [1280, 720]

        self.image_publisher = rospy.Publisher("detected_image", sensor_msgs.msg.Image, queue_size=50)
        self.bridge = CvBridge()

        self.online_board_inference = OnlineInference(weights=self.weight_path,
                                                      device=0,
                                                      yaml=self.labels_path,
                                                      img_size=[1280, 720],
                                                      half=False)

        self.first_identification = True

        rospy.loginfo(GREEN + "Service alive ...." + END)

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

            features = {"red_button": [],
                        "door_handle": []}

            inference_result, _, _ = self.online_board_inference.realtime_inference(frame, conf_thres=0.6,
                                                                                    iou_thres=.45, agnostic_nms=False,
                                                                                    max_det=1000, view_img=False)

            print("---------------- INFERENCE RESULT -----------")
            print(inference_result)
            # det = self.make_inference(frame, 0.6)
            # img_ori = rgb_frame.copy()
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
        self.online_board_inference.realtime_inference(frame, conf_thres=0.6,
                                                       iou_thres=.45, agnostic_nms=False,
                                                       max_det=1000, view_img=True)
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

    def triangle_identification(self, request):
        rospy.loginfo(SERVICE_CALLBACK.format("Triangle identification"))

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

    vision_system = BoardLocalization(test, images_path, weight_path, labels_path)

    rospy.Service(SERVICE_NAME_BOARD, Trigger, vision_system.board_localization)
    rospy.Service(SERVICE_NAME_SCREEN_TARGET, Trigger, vision_system.triangle_identification)
    rospy.Service(SERVICE_NAME_SCREEN_TARGET_INIT, Trigger, vision_system.triangle_identification)

    rospy.spin()


if __name__ == "__main__":
    main()
