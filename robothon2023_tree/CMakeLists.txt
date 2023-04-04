cmake_minimum_required(VERSION 3.0.2)
project(robothon2023_tree)
add_compile_options(-std=c++17 -funroll-loops -Wall -Ofast)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  skills_util
  skills_util_msgs
  tf
)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES robothon2023_tree
  CATKIN_DEPENDS roscpp
                 std_msgs
                 skills_util
                 skills_util_msgs
                 tf
  DEPENDS
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(        run_tree src/run_tree.cpp)
add_dependencies(      run_tree ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries( run_tree
        ${catkin_LIBRARIES}
)
