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


class BoardLocalization:

    def __init__(self, test, folder_path):

        self.realsense = RealSense()

        # Retrieve camera parameters
        rospy.loginfo(YELLOW + "Waiting camera parameters ..." + END)

        self.realsense.getCameraParam()
        self.realsense.waitCameraInfo()

        rospy.loginfo(GREEN + "Camera parameters retrived correctly" + END)

        # Estimated parameters
        self.depth = 300  # Estimated distance

        # Tf Broadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.test = test
        self.folder_path = folder_path
        rospy.loginfo(GREEN + "Service alive ...." + END)

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

        # Roi
        crop_img = getRoi(rgb_frame, 144, 669, 225, 1000)
        crop_img = rgb_frame

        # Color spaces
        hsv, lab, bw = getAllColorSpaces(crop_img)
        hue, saturation, value = cv2.split(hsv)
        l_col, a_col, b_col = cv2.split(lab)

        # Threshold
        ret, value_th = cv2.threshold(value, 90, 255, 0)

        # Get board contour
        board_cnt, contours_limited, contours = getBoardContour(value_th)

        # Screen identification
        ScreenPos, idx_screen = getScreen(a_col, contours_limited)

        contours_limited.pop(idx_screen)  # Remove screen contour from list

        rospy.loginfo(GREEN + "Screen position: {}".format(ScreenPos) + END)
        new_img = getRoi(rgb_frame, 144, 669, 360, 1150)
        new_img = cv2.circle(crop_img, (ScreenPos[0][0], ScreenPos[0][1]), 5, color=(255, 0, 0), thickness=2)

        rospy.loginfo(GREEN + "Identifying red button..." + END)
        new_img = getRoi(rgb_frame, 144, 669, 360, 1150)
        crop_img2 = getRoi(rgb_frame, 144, 669, 360, 1150)
        RedBlueButPos, BlueButtonPos, id_red_blue_contour = getRedBlueButtonsNewVersion2(saturation, b_col,
                                                                                         contours_limited, crop_img2,
                                                                                         ScreenPos, rgb_frame)
        rospy.loginfo(GREEN + "Identified red button" + END)
        print(RedBlueButPos)
        print(BlueButtonPos)
        contours_limited.pop(id_red_blue_contour)
        new_img = cv2.circle(crop_img, (RedBlueButPos[0], RedBlueButPos[1]), 5, color=(0, 0, 255), thickness=2)
        new_img = cv2.circle(crop_img, (BlueButtonPos[0], BlueButtonPos[1]), 5, color=(255, 0, 0), thickness=2)
        #
 #       cv2.imshow("Identificato", new_img)
 #       cv2.waitKey(-1)
 #       cv2.destroyAllWindows()

        crop_img = getRoi(rgb_frame, 144, 669, 360, 1150)

        depth_frame = self.realsense.getDistanceFrame()
        print(f"Shape depth: {depth_frame.shape}")
        print(f"Distanza: {depth_frame[RedBlueButPos[1], RedBlueButPos[0]]}")
        print(f"Distanza: {depth_frame[BlueButtonPos[1], BlueButtonPos[0]]}")

        red_button_camera = np.array(self.realsense.deproject(RedBlueButPos[0], RedBlueButPos[1],
                                                              depth_frame[RedBlueButPos[1], RedBlueButPos[0]])) / 1000.0
        blue_button_camera = np.array(self.realsense.deproject(BlueButtonPos[0], BlueButtonPos[1], depth_frame[
            BlueButtonPos[1], BlueButtonPos[0]])) / 1000.0
        screen_camera = np.array(self.realsense.deproject(ScreenPos[0][0], ScreenPos[0][1],
                                                          depth_frame[ScreenPos[0][1], ScreenPos[0][0]])) / 1000.0

        listener = tf.TransformListener()
        rospy.sleep(1.0)
        found = False
        while not found:
            try:
                (trans, rot) = listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
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
        # print(M_world_camera)

        # self.pubTF(red_button_camera,"red_prima","camera_color_optical_frame")
        # self.pubTF(key_lock_camera,"keuy_prima","camera_color_optical_frame")
        # self.pubTF(screen_camera,"screen_prima","camera_color_optical_frame")

        red_button_world = np.dot(M_world_camera, self.get4Vector(red_button_camera))
        blue_button_world = np.dot(M_world_camera, self.get4Vector(blue_button_camera))
        screen_world = np.dot(M_world_camera, self.get4Vector(screen_camera))
        print(f"Red button (in base_link) before set z: {red_button_world}")
        red_button_world_backup = red_button_world[0:-1]
        red_button_world = red_button_world[0:-1]
        red_button_world[-1] = 0.923  # 1.5
        blue_button_world = blue_button_world[0:-1]
        blue_button_world[-1] = 0.923  # red_button_world[-1]
        screen_world = screen_world[0:-1]
        screen_world[-1] = 0.923
        z_axis = np.array([0.0, 0.0, -1.0])

        print(f"Red button (in base_link) after set z: {red_button_world}")
        print(f"Blue button (in base_link) after set z: {blue_button_world}")
        print(f"Screen (in base_link) after set z: {screen_world}")

        x_axis = (blue_button_world - red_button_world) / np.linalg.norm(blue_button_world - red_button_world)
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
        static_transform_blue = self.getStaticTrasformStamped("camera_color_optical_frame", "blue",
                                                              blue_button_camera,
                                                              [0, 0, 0, 1])
        static_transform_screen = self.getStaticTrasformStamped("camera_color_optical_frame", "screen",
                                                                screen_camera,
                                                                [0, 0, 0, 1])

        rospy.loginfo(GREEN + "Published tf" + END)

        # static_transformStamped_reference = self.getStaticTrasformStamped("board", "reference",
        #                                                                   [0.137, 0.094, -0.155],
        #                                                                   [0.0, 0.0, 0.959, -0.284])

        self.broadcaster.sendTransform(
            [static_transformStamped_board, static_transform_red,
             static_transform_blue, static_transform_screen])
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

    board_localization = BoardLocalization(test, images_path)

    rospy.Service(SERVICE_NAME, Trigger, board_localization.callback)
    rospy.spin()


if __name__ == "__main__":
    main()
