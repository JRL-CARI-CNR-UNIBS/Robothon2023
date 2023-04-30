## Behavior Tree Definition

This repository contains the Behavior Tree structures, the skills parameters and the UR script template.
All these files are used to define the task structure and its skills.

Below are the most significant trees related to the accomplishment of the tasks required by the competition.

## Main Behavior-Tree
The main behavior tree is simply a sequence of sub-trees containing the individual tasks to be performed in the order required by the competition (and editable as desired).

- 📷 Board identification (*init*)
- 🔵 Press blue button
- 🎚️ Match slider
- 🔌 Cable positioning
- 🚪 Open door
- ⚡ Test circuit (*probe testing*)
- 💥 Cable winding (*cable winding*)
- 🔴 Press red button
 
<p align="center">
  <img height="300" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/mainBT.png">
</p>

## Sub-Trees
This section shows the BTs that manage the execution of individual tasks.

### Board identification

This sub-tree represents a sequence in which:
1. The robot moves to the **starting configuration**.
2. The *camera image* is acquired, and the board's features are identified (if it fails, the procedure is repeated for up to 5 attempts).
3. The robot moves to a configuration *close* to the board so that a more *accurate localization* of the board can be performed.
4. The image from the camera is acquired once again, and the *localization* of the board is performed (up to a maximum of 5 attempts in case of fail).
When the sub-tree is finished, the board reference system is published, and the other tasks can be executed.

<p align="center">
  <img height="300" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/boardIdentificationBT.png">
</p>

### Slider (Triangles matching)

The sub-tree that handles triangle matching with the slider is slightly articulated so as to make task execution flexible as a result of inaccuracies, lags, etc.

<p align="center">
  <img height="400" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/sliderBT.png">
</p>

Specifically: 
1. The gripper is opened.
2. The robot is brought into a slider approach configuration, and the gripper is closed so that the slider is engaged.
3. The slider is moved to its center position.
4. **Triangle matching procedure**. The tree consists of a fallback with two branches. The first branch identifies the absence of a triangle to be reached (target). If a triangle is identified, the node returns False, and the second branch (the right branch) is executed, which moves the robot so that the slider reaches the goal position (three relative movements around the target). If it was necessary to execute this branch, a Failure is forced in order to repeat the triangle check procedure so that the move is performed in case 1) The target was not adequately reached, 2) In the initial phase the center target was not adequately reached. When the triangle recognition leaf node no longer identifies any triangles, it returns True, and the fallback node does not call back the right branch, so the matching is performed correctly. This procedure is repeated for a maximum of 10 times. 
5. The gripper is opened, and the task is completed by returning to the approach position.


