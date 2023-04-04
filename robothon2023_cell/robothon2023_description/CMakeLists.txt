cmake_minimum_required(VERSION 3.0.2)
project(robothon2023_description)
find_package(catkin REQUIRED COMPONENTS
ur_description
realsense2_description
robotiq_2f_85_gripper_visualization
)
catkin_package(
)

include_directories(
)
