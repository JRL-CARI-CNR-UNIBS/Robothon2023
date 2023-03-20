#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from utils import *
import cv2


@dataclass
class ImageLabelling:
    img: np.ndarray
    acquired_keypoints: int = field(default=0, init=False)
    finished: bool = field(default=False, init=False)
    known_keypoints: List[str] = field(init=True)

    def __post_init__(self):
        pass

    def set_keypoint(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.is_finished():
            print(UserMessages.CUSTOM_GREEN.value.format(
                f"Keypoint: {self.known_keypoints[self.acquired_keypoints]} acquired"))
            image = cv2.circle(self.img, (x, y), 4, (255, 0, 0), 2)
            print(id(image))
            cv2.imshow('image', image)
            print(f"({x},{y})")
            self.acquired_keypoints += 1

    def is_finished(self):
        if self.acquired_keypoints >= len(self.known_keypoints):
            return True
        return False

    def get_labels(self):
        # TODO: Get image labels
        pass


def check_labellers_params(labellers_range, labeller_name):

    # Check that the labellers name is in the labeller range
    if labeller_name not in labellers_range.keys():
        rospy.logerr(
            UserMessages.CUSTOM_RED.value.format(f"Labeller name: {labeller_name} not defined in labellers range"))
        return False

    for name in labellers_range:
        range_min, range_max = labellers_range[name]["min"], labellers_range[name]["max"]
        other_labellers_name = list(labellers_range.keys())
        print(other_labellers_name)
        # Check intersection with other labellers
        for other_labeller in other_labellers_name:
            if other_labeller is name:
                continue
            other_min, other_max = labellers_range[other_labeller]["min"], labellers_range[other_labeller]["max"]
            if min([other_max, range_max]) - max([other_min, range_min]) > 0:
                rospy.loginfo(UserMessages.CUSTOM_RED.value.format(
                    f"Labellers range of {name} and {other_labeller} have not null intersection"))
                return False
    return True


def main():
    rospy.init_node('keypoints_labeling')

    if not check_params_existence(
            ["known_keypoints", "dataset_path", "fig_prefix_name", "~labellers", "~labeller_name"]):
        return 0

    fig_prefix_name = rospy.get_param("fig_prefix_name")
    dataset_path = rospy.get_param("dataset_path")
    known_keypoints = rospy.get_param("known_keypoints")

    labellers = rospy.get_param("~labellers")
    labeller_name = rospy.get_param("~labeller_name")

    # Check labellers
    if not check_labellers_params(labellers, labeller_name):
        return 0

    range_min, range_max = labellers[labeller_name]["min"], labellers[labeller_name]["max"]

    for n_fig in range(range_min, range_max+1):
        # Load image
        fig_path = f"{dataset_path}{fig_prefix_name}{n_fig}.png"
        rospy.loginfo(UserMessages.CUSTOM_GREEN.value.format(f"Analyzing image: {fig_path}"))
        img = cv2.imread(str(fig_path))

        # Check if image exists
        if img is None:
            rospy.logerr(UserMessages.CUSTOM_RED.value.format(f"Image: {fig_path} does not exist"))
            break

        # Manage user labelling errors
        mistake = True
        while mistake:

            img_labelling = ImageLabelling(img.copy(), known_keypoints)

            # Show base image
            cv2.imshow("image", img)

            # Callback for image  labelling
            cv2.setMouseCallback('image', img_labelling.set_keypoint)
            try:
                while not img_labelling.is_finished():
                    cv2.waitKey(1)
                if img_labelling.is_finished():
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

            except KeyboardInterrupt:
                print("Press Ctrl-C to terminate while statement")
                cv2.destroyAllWindows()
                break
            user_input = input(UserMessages.CUSTOM_YELLOW.value.format("Do you wonna go on? "))
            if not user_input:
                print("Go on with next image ...")
                mistake = False


if __name__ == "__main__":
    main()
