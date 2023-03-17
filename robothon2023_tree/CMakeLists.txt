cmake_minimum_required(VERSION 3.0.2)
project(robothon2023_tree)

find_package(catkin REQUIRED COMPONENTS
  behaviortree_cpp_v3
  roscpp
  std_msgs
)

catkin_package(
 # INCLUDE_DIRS include
 # LIBRARIES robothon2023_tree
 # CATKIN_DEPENDS behaviortree_cpp_v3 roscpp std_msgs
 # DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
