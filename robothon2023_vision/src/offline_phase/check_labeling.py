#!/usr/bin/env python3

import rospy
import numpy as np
from utils import *
import cv2


def main():
    rospy.init_node('keypoints_labeling')

    if not check_params_existence(
            ["dataset_path", "fig_prefix_name", "~image_label"]):
        return 0

    fig_prefix_name = rospy.get_param("fig_prefix_name")
    dataset_path = rospy.get_param("dataset_path")
    image_label = rospy.get_param("~image_label")

    for image_name in image_label:

        fig_path = f"{dataset_path}{image_name}"
        rospy.loginfo(UserMessages.CUSTOM_GREEN.value.format(f"Analyzing image: {fig_path}"))
        img = cv2.imread(str(fig_path))

        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cv2.imshow(image_name, img)
        c3_skel = np.zeros_like(img)
        for keypoint in image_label[image_name]:
            x = image_label[image_name][keypoint]["x"]
            y = image_label[image_name][keypoint]["y"]

            cv2.circle(img, (x, y), 2, (255, 0, 0), 2)

            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.putText(mask, str(keypoint), (x, y), 1, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

            M = cv2.getRotationMatrix2D((x, y), 40, 1)
            out = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]))
            c3_skel[:, :, 0] += out
            c3_skel[:, :, 1] += out
            c3_skel[:, :, 2] += out

        cv2.imshow(image_name, cv2.add(img, c3_skel))
        cv2.waitKey(-1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
