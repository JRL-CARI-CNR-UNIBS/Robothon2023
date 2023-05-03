#! /usr/bin/env python3

from std_srvs.srv import Trigger, TriggerResponse
import sensor_msgs.msg
from RealSense import RealSense
import rospy
from cv_bridge import CvBridge
from detect_digits_on_screen import *
import cv2 as cv # REMOVE

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'

SERVICE_CALLBACK = GREEN + "Service call {} received" + END
PARAM_NOT_DEFINED_ERROR = "Parameter: {} not defined"
SUCCESSFUL = "Successfully executed"
NOT_SUCCESSFUL = "Not Successfully executed"

COLOR_FRAME_TOPIC = "/camera/color/image_raw"
SCREEN_DIGIT_TOPIC = "/robothon2023/screen_digits"


# Define parameters
n_clusters = 8                      # number of clusters for the k-means algorithm
color = np.asarray([250, 250, 230]) # color of the selected cluster (digits)
w_crop_corr = 15                    # screen cropping to select only the digits (width)
h_crop_corr = 25                    # screen cropping to select only the digits (height)
delta_w = 10                         # horizontal space between digits
n_digits = 4                        # number of digits on the screen
segment_width = 7                   # width of each digit segment


def callback(req):
    bridge = CvBridge()
    time_now = rospy.Time.now().to_sec()
    # print(f"Now: {time_now}")
    print(RED + "Digit call" + END)
    # Acquire image and wait for synchronization
    img_time = 0.0
    while img_time < time_now + 1:
        frame = rospy.wait_for_message(COLOR_FRAME_TOPIC, sensor_msgs.msg.Image, timeout=None)
        img = bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")
        img_time = frame.header.stamp.secs + 1e-9 * frame.header.stamp.nsecs
        # print(f"Image time: {img_time}")
    cv.imwrite('/home/galois/projects/robothon23/src/Robothon2023/robothon2023_vision/file/circuit_board.png')
    # Detect digits on screen
    # reading = read_digits_screen(img, n_clusters, color, w_crop_corr, h_crop_corr, n_digits, delta_w, segment_width)
    # print(f'Number read on the screen: {reading}')
    # rospy.loginfo(GREEN + f'Number read on the screen: {reading}' + END)
    # rospy.set_param("/transferability_demo/digits/reading", reading)

    return TriggerResponse(True, SUCCESSFUL)


def main():
    rospy.init_node("on_screen_digit_detection")

    realsense = RealSense()

    # Retrieve camera parameters
    rospy.loginfo(YELLOW + "Waiting camera parameters ..." + END)
    realsense.getCameraParam()
    realsense.waitCameraInfo()
    rospy.loginfo(GREEN + "Camera parameters retrieved correctly" + END)

    # Service for on-screen digit identification
    rospy.Service(SCREEN_DIGIT_TOPIC, Trigger, callback)
    # img = cv.imread('/home/galois/projects/robothon23/src/Robothon2023/robothon2023_vision/file/frame_0.png')
    # reading = read_digits_screen(img, n_clusters, color, w_crop_corr, h_crop_corr, n_digits, delta_w, segment_width)
    # print(f'Number read on the screen: {reading}')
    #
    rospy.spin()


if __name__ == "__main__":
    main()
