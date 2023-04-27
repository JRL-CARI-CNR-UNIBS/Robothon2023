# Robothon® 2023 The Grand-Challenge: <br> bisCARI Team Report
<p align="center">
Authors: Michele Delledonne, Roberto Fausti, Michele Ferrari, Samuele Sandrini 
</p>

## Hardware Setup

A 6-DoF collaborative robot, a Universal Robot UR10e, is used for object manipulation. The robot is mounted upside down on an actuated linear track. The linear guide is kept in a fixed position for challenge purposes, so the robotic system has 6 degrees of freedom.

The robot is equipped with a classic two-finger gripper for pick-place (the Hand-E Robotiq model) with custom fingers for the e-waste manipulation application that are designed and 3D-printed by the team. 

The setup is equipped with a Vision System that consists of an RGB-D Camera. The stereo-camera Realsense-D435 was used, and it is rigidly mounted to the robot. Also, the camera support was designed and 3d printed by the team.

In summary, the hardware setup consists of a hardware solution kept as general purpose as possible and low cost to facilitate transferability to the e-waste industrial world from the specific task-board application.

## Software Solution
Our software framework is developed using ROS (Robot Operating System). The overall framework is summarized in figure below. At the beginning, the vision system localizes the board and its feature points and advertises the reference frames (Section 2.1). Then, the sequence of tasks is executed with respect to such frames. The execution of each task combines motion planning and different kind of controllers (Sections 2.2 and Section 2.3).

### Vision System


### Task execution management
The automatic execution of the tasks is managed by a Behavior Tree Executor. Each Robothon’s task corresponds to a SubTree. BTs are a very efficient way of creating complex systems that are both modular and flexible. This makes it easy to manage the ability to change the order in which tasks are executed, thus creating a flexible application.

### Planning and execution

### Controllers


## Description of tasks


## Repository of software modules
- [Cell Configuration (Geometric configuration and controllers)](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/tree/master/robothon2023_cell)
- [Vision System](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/tree/master/robothon2023_vision)
- [Behavior Tree Definition](https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/tree/master/robothon2023_tree)
- [Task Execution management](https://github.com/JRL-CARI-CNR-UNIBS/RL_task_framework)

## Dependencies 
Regarding behavior trees we use a well-known library in robotics ([BehaviorTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP)) that is based on C++ and it provides a framework to create BehaviorTrees in a flexible, easy to use, reactive and fast way.

## Authors

- Samuele Sandrini, [SamueleSandrini](https://github.com/SamueleSandrini)
- Michele Delledonne, [Michele Delledonne](https://github.com/MichiDelle)
- Michele Ferrari, [Michele Delledonne](https://github.com/MikFerrari)
- Roberto Fausti, [Roberto Pagani](https://github.com/RobertoFausti)