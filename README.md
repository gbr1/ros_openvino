# **ros_openvino**
A ROS package to wrap openvino inference engine and get it working with **Myriad** and **Intel CPU/GPUs**<br>
The main topic is to accelerate robotics developing using ROS with OpenVINO.<br>
This package is designed on async api of [Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit) and allows an easy setup for **object detection**.<br><br>
Entire package is born to support Myriad X, so it works perfectly with:
- Movidius Neural Compute Stick 2
- Ai Core X
- Movidius Neural Compute Stick
- Ai Core

**object detection** node allows you to use a standard image topic and obtain a new topic with live result image. 
If you enable *depth analysis* you can add depth image topic to obtain 3d boxes, visible as markers in rviz.

## **Wiki**
[Wiki home](https://github.com/gbr1/ros_openvino/wiki)

## **Setup**
You need some steps to get ros_openvino working.<br>
If you are working on a Myriad based device (NCS, NCS2, AI Core, AI Core X) follow [this guide](https://github.com/gbr1/ros_openvino/wiki/Fast-setup-for-Myriad-based-devices) because is faster.

If you have already this kind of stuffs, please take care to modify steps for your case.

### **- Prerequisites**
1. Ubuntu 16.04/18.04
2. OpenVINO 2018 R5 or newer (latest officially tested is 2019 R3)
3. ROS Kinetic Kame/Melodic Morenia

### **- Setup** ***ros_openvino***
1. Clone repo in your **source folder**, assuming `catkin_ws` as your ROS workspace:<br>
`cd ~/catkin_ws/src`<br>
`git clone http://github.com/gbr1/ros_openvino.git`<br>
2. Compile!<br>
`cd ~/catkin_ws`<br>
`catkin_make`<br>
`catkin_make install`

all installation steps and a case of study configuration are described in [Setup environment wiki page](https://github.com/gbr1/ros_openvino/wiki/Setup-environment).


Remember that OpenVINO enviroment variables, ROS enviroment variables and ROS workspace need to be sourced in `bashrc`. <br>


### **- Tests**

There are some demo launch files.
1. **Webcam and GPU**
<br>you need to install usb_cam node:<br>
`sudo apt-get install ros-kinetic-usb-cam`
<br>then type:<br>
`roslaunch ros_openvino gpu_demo_webcam.launch`
<br>![gpu](https://user-images.githubusercontent.com/9216366/53649736-21db3400-3c43-11e9-9353-ee603390aedc.png)

2. **Webcam and Myriad**
<br>you need to install usb_cam node:<br>
`sudo apt-get install ros-kinetic-usb-cam`
<br>then type:<br>
`roslaunch ros_openvino myriad_demo_webcam.launch`
<br>![myriad_webcam](https://user-images.githubusercontent.com/9216366/53649817-5b13a400-3c43-11e9-963c-7f41e899b72c.png)

3. **Realsense D435/D435i/D415 and Myriad**
<br>you need to install realsense package, follow [here](https://github.com/intel-ros/realsense) step-by-step according your environment setup.<br>
After that, just type:<br>
`roslaunch ros_openvino myriad_demo_realsense.launch`
<br>![myriad](https://user-images.githubusercontent.com/9216366/53649915-98783180-3c43-11e9-81ab-2579c291af1e.png)
<br>

## **Nodes**
- [object_detection](https://github.com/gbr1/ros_openvino/wiki/Object-Detection), a node to detect objects in enviroment using a RGB or RGBD camera using MobileNet-SSD

<br>
<br>

---
## Note
Tests are done on [UP2](https://up-board.org/upsquared/specifications/) with Intel Atom, 4GB of RAM and 64GB eMMC and [AI Core X](https://up-board.org/ai-core-x/).
Results with a complete analysis with 4-5 object detected is about 30fps for rgb output and 16-18Hz for spatial calculation.
<br>
On GPU analysis drop down the frame rate.
<br><br>
If OpenVINO works on Ubuntu 18.04 this package should be compatible with ROS Melodic Morenia.

## Known issues
- CPU device doesn't work.  In the future yes, it is in "todo" list;
- GPU is worst than Myriad;
- resize of input image is not availble at the moment. It is in "todo" list.

## Future steps
- add face and body tracking
- improve depth analysis

<br>
<br>

***This package is distribuited under Affero GPLv3.*** <br>
***Copyright (c) 2019 Giovanni di Dio Bruno***

*Models:*
- *mobilenet-ssd.bin*
- *mobilenet-ssd.xml*

*are created using OpenVINO model downloader and converter.*<br>
*Original mobilenet-ssd model are under Intel OpenVINO licenses.*
