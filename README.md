# Robothon® 2023 The Grand-Challenge: <br> bisCARI Team Report
<p align="center">
Authors: Michele Delledonne, Roberto Fausti, Michele Ferrari, Samuele Sandrini
</p>

## Hardware Setup
<p align="center">
  <img height="500" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/setup_description.png">
</p>

A **6-DoF collaborative robot**, a Universal Robot **UR10e**, is used for object manipulation. The robot is mounted upside down on an actuated linear track. The linear guide is kept in a fixed position for challenge purposes, so the robotic system has 6 degrees of freedom.

The robot is equipped with a classic **two-finger gripper** for pick-place (the **Hand-E Robotiq** model) with **custom fingers** for the e-waste manipulation application that has been designed and 3D-printed by the team.

The setup is equipped with a **Vision System** that consists of an RGB-D Camera. The stereo-camera **Realsense-D435** was used, and it is rigidly mounted to the robot. Also, the camera support was designed and 3D-printed by the team.

In summary, the hardware setup consists of a hardware solution kept as **general purpose** as possible and low cost to facilitate transferability to the e-waste industrial world from the specific task-board application.

## Software Solution
Our software framework is developed using **ROS** (Robot Operating System).

The overall **framework** is summarized in figure below.
<p align="center">
  <img height="400" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/software_setup.png">
</p>

At the beginning, the vision system localizes the board and its feature points, so it reconstructs the position of the board in the robot workspace. Then, the sequence of tasks is executed with respect to such frames. The execution of each task combines different kind of controllers of the robot low-level controller.

The main **idea** of 2023 solution is to combine the use ROS as **coordinator** with the UR-robot **low-level controllers** in the seak of movement for their stability and fastness.

### Vision System
The **vision** system is based on a **two-step procedure**: one performed offline and one online for real-time board identification.
The **offline procedure** is required for the following:
1. Perform **Hand eye calibration** to estimate the transformation between the camera frame and the robot one. Since the camera is rigidly mounted to the robot's end-effector, the calibration was performed in the so-called "eye-in-hand" mode to estimate the transformation between the camera reference system and that of the end-effector. To perform this procedure,  the plugin [Hand-Eye Calibration](Hand-Eye Calibration) integrated into MoveIt is used.
2. In order to use a pre-trained neural network and customize it to the Robothon use case, it was necessary to create a dataset for the use case under consideration. For this purpose, an acquisition of 150 images was performed by randomly placing the task board on the worktable and under different conditions. The images were labeled, and label files in YOLO format were generated.
3. [Yolo V6](https://github.com/meituan/YOLOv6) is fine-tuned with our custom data obtained from step 2 and finally, the network weights are obtained.

The online procedure aims to:
1. Identify **task-board features**. To recognize the features in the RGB image acquired by the camera, YOLO v6 was used with the weights obtained in step 3 of the offline procedure. So at this stage, is only derived the network output with the task board real-time image provided as input to the net.
2. The features used for reconstructing the position of the task board are the red button and the door handle since they are the ones estimated most accurately.

The figure below shows the whole procedure.

<p align="center">
  <img height="400" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/vision_system_robothon.png">
</p>

### Task execution management
A **Behavior Tree** Executor manages the automatic execution of the tasks. BTs are a very efficient way of creating complex systems that are both modular and flexible. This tool makes it easy to manage the ability to change the order in which tasks are executed, thus creating a flexible application.

The implemented BT structure consists of a SubTree for each Robothon's task, in this way, their order can be changed quickly and easily. The leaves of these SubTrees correspond to the base skills that compose the task, these skills are the basic functions that the UR script proposes, like *movel* (linear move), *movej* (move in joints space) and so on. **Universal Robots** provides ROS drivers that allow to send UR scripts to the robot to perform them. Using this feature is possible to customize every single skill and then ask the robot to execute it.

The definition of skills is performed using [RL_task_framework](https://github.com/JRL-CARI-CNR-UNIBS/RL_task_framework/tree/robothon) that provides a simple BT leaf that, with a simple string input, generates a UR script that corresponds to the desired skill. RL_task_framework works with a task structure that breaks down the task into a sequence of actions and the actions into a sequence of skills. The user has to define ROS parameters for every skill specifying the skill type and its parameters, like velocity, distance, force. The BT leaf receiving the action and skill name can read its parameters and generate the UR command.

<p align="center">
  <img height="300" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/mainBT.png">
</p>

## Repository of software modules
- [Cell Configuration (Geometric configuration and controllers)](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/tree/master/robothon2023_cell)
- [Vision System](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/tree/master/robothon2023_vision)
- [Behavior Tree Definition](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/tree/master/robothon2023_tree)
- [Task Execution management](https://github.com/JRL-CARI-CNR-UNIBS/RL_task_framework/tree/robothon)

## Dependencies
Regarding behavior trees we use a well-known library in robotics ([BehaviorTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP)) that is based on C++ and it provides a framework to create BehaviorTrees in a flexible, easy to use, reactive and fast way.

Communication with the UR10e is performed thanks to the [Universal Robot packages](https://github.com/ros-industrial/universal_robot).

## Quick Start
To start the framework, four main launcher files are required:
1. [start_vision.launch](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_vision/launch/start_vision.launch): brings up all the services related to the vision system.
2. [real_start.launch](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_cell/robothon2023_configurations/launch/real_start.launch): setup of the cell, run moveit and connect to the robot.
3. [skills_servers.launch](https://github.com/JRL-CARI-CNR-UNIBS/RL_task_framework/blob/robothon/skills_util/launch/skills_servers.launch): run the skill server which define each skill and is responsible to run the behavior tree.
4. [run_all.launch](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_tree/launch/run_all.launch): provides the subtrees to run the tasks. This launcher must be the last since it will trigger the start the main behavior-tree.

## Authors

- Samuele Sandrini, [SamueleSandrini](https://github.com/SamueleSandrini)
- Michele Delledonne, [Michele Delledonne](https://github.com/MichiDelle)
- Michele Ferrari, [Michele Ferrari](https://github.com/MikFerrari)
- Roberto Fausti, [Roberto Fausti](https://github.com/RobertoFausti)
