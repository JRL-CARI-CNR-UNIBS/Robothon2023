# Robothon bringsCARI Vision System
<p align="center">
Authors: Michele Delledonne, Roberto Fausti, Michele Ferrari, Samuele Sandrini
</p>

<p align="center">
  <img height="500" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/vision_system_robothon.png">
</p>

The **vision** system is based on a **two-step procedure**: one performed offline and one online for real-time board identification.
The **offline procedure** is required for the following:
1. Perform **Hand eye calibration** to estimate the transformation between the camera frame and the robot one. Since the camera is rigidly mounted to the robot's end-effector, the calibration was performed in the so-called "eye-in-hand" mode to estimate the transformation between the camera reference system and that of the end-effector. To perform this procedure,  the plugin [Hand-Eye Calibration](Hand-Eye Calibration) integrated into MoveIt is used.
2. In order to use a pre-trained neural network and customize it to the Robothon use case, it was necessary to create a dataset for the use case under consideration. For this purpose, an acquisition of 150 images was performed by randomly placing the task board on the worktable and under different conditions. The images were labeled, and label files in YOLO format were generated.
3. [Yolo V6](https://github.com/meituan/YOLOv6) is fine-tuned with our custom data obtained from step 2 and finally, the network weights are obtained.

The online procedure aims to:
1. Identify **task-board features**. To recognize the features in the RGB image acquired by the camera, YOLO v6 was used with the weights obtained in step 3 of the offline procedure. So at this stage, is only derived the network output with the task board real-time image provided as input to the net.
2. The features used for reconstructing the position of the task board are the red button and the door handle since they are the ones estimated most accurately.

## Triangles detections

The vision system is also responsible for recognizing triangles on the display. Classical vision using **opencv** functions is used for this purpose. In particular, the method involves applying thresholds on the color space since the display has a strong brightness. Next, triangles within the area of interest are identified.


## Transferability Demo

The same method used to identify task board features is applied to the transferability demo. It consists of a power circuit for a display. The vision system, based on **YOLO v6* is employed to recognize the heat sacchanger, the yellow coil and the two black coils.

<p align="center">
  <img height="500" src="https://github.com/JRL-CARI-CNR-UNIBS/Robothon2023/blob/master/robothon2023_images/transferability_identification.png">
</p>


## Authors

- Samuele Sandrini, [SamueleSandrini](https://github.com/SamueleSandrini)
- Michele Delledonne, [Michele Delledonne](https://github.com/MichiDelle)
- Michele Ferrari, [Michele Ferrari](https://github.com/MikFerrari)
- Roberto Fausti, [Roberto Fausti](https://github.com/RobertoFausti)
