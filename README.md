<!--
 * @Author: Hongkun Luo
 * @Date: 2024-07-24 03:11:30
 * @LastEditors: luohongk luohongkun@whu.edu.cn
 * @Description: 
 * 
 * Hongkun Luo
-->

 <h1 align="center">SuperVINS: A Real-Time Visual-Inertial SLAM Framework for Challenging Imaging Conditions</h1>
  <h3 align="center">
    <a href="https://luohongkun.com/">Hongkun Luo</a>, <a href="">Yang Liu</a>, <a href="https://jszy.whu.edu.cn/guochi">Chi Guo</a>, <a href="https://cesi.cumt.edu.cn/info/1101/8625.htm">Zengke Li</a>, <a href="https://gnsscenter.whu.edu.cn/info/1301/1081.htm">Weiwei Song</a>
  </h3>
  <p align="center">
    <a href="https://luohongkun.com/SuperVINS/">Project Website</a> , <a href="https://ieeexplore.ieee.org/document/10949688">Paper (IEEE Sensors Journal)</a>
  </p>
  <p align="center">
    <a href="https://github.com/luohongk/SuperVINS">
      <img src="https://img.shields.io/badge/License-GPL3.0-yellow.svg" />
    </a>
    <a href="https://cmake.org/">
      <img src="https://img.shields.io/badge/built%20with-Cmake-red.svg" />
    </a>
  </p>
</p>

![Static Badge](https://img.shields.io/badge/VINS-Image_IMU-red) ![Static Badge](https://img.shields.io/badge/Cpp-14-blue) ![Static Badge](https://img.shields.io/badge/DeepLearning-SuperPoint_LightGlue-red) ![Static Badge](https://img.shields.io/badge/ROS1-noetic-blue) ![Static Badge](https://img.shields.io/badge/BoW-DBoW3-red) ![Static Badge](https://img.shields.io/badge/WHU-BRAIN_LAB-red) ![Static Badge](https://img.shields.io/badge/luohongk-blue) ![Static Badge](https://img.shields.io/badge/Wuhan-China-green)

<div align=center><img src="resources\SuperVINS.png" width =100%></div>

# News

- **March 2025**: Published in IEEE Sensors Journal.
- **March 7, 2025**: Added SuperVINS 1.0.
- **October 6, 2024**: Released base code.
- **August 7, 2024**: Added demo.
- **July 31, 2024**: Published preprint (Journal in submission). [View Preprint](https://arxiv.org/abs/2407.21348)

# Demo

### GIF

<div align=center><img src="resources\SuperVINS_demo.gif" width =100%></div>

### Video

If you want to watch the full demo video, please click the [link](resources/video.mp4)

# 1 Introduction

This project is improved based on VINS-Fusion. [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) is a well-known SLAM framework. The original version of VINS-Fusion front-end uses traditional geometric feature points and then performs optical flow tracking. This project uses the feature point method, introduces SuperPoint feature points and feature descriptors, and uses the LightGlue network for feature matching. In the loopback part, the original VINS-Fusion extracts the brief descriptor and uses DBoW2 for loopback detection. This project uses DBoW3 and SuperPoint deep learning descriptors for loopback detection. Created a SLAM system based on deep learning.

Why is it called SuperVINS? We named this project in honor of SuperPoint and VINS-Fusion. In this project, "Super" does not mean "super and excellent", it just means that the SuperPoint descriptor runs through the front-end and loop closure detection. "VINS" means that this project uses the visual-inertial fusion algorithm, Meanwhile，it is also to thank VINS-Fusion for its outstanding contribution.

# 2 Build Project

### 2.1 **Ubuntu** and **ROS**

Ubuntu 64-bit 20.04.
ROS Noetic. **[ROS Installation](http://wiki.ros.org/ROS/Installation)**

### 2.2 **OpenCV**

OpenCV4.2.0. **[OpenCV4.2.0](https://github.com/opencv/opencv/archive/refs/tags/4.2.0.zip)**

if you use Ubuntu 20.04, you can install it by:`sudo apt-get install libopencv-dev`

### 2.3 **Ceres Solver**

Follow **[Ceres Installation](http://ceres-solver.org/installation.html)**.
**[Ceres 2.1.0](https://github.com/ceres-solver/ceres-solver/releases/tag/2.1.0)**.

### 2.4 **ONNX RUNTIME**

**[onnxruntime-linux-x64-gpu-1.16.3](https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz)**

### 2.5 **install libraries to the specified path**

**If you want to install the third-party library in the specified path, you can follow the steps below**

```bash
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX="/some/where/local"  ..
make -j4
make install
```

# 3 Create a ROS1 workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src/
catkin_init_workspace
cd ~/catkin_ws
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bash
source ~/.bashrc
```

# 4 How does it work?

### 4.1 Clone project

```bash
  cd ~/catkin_ws/src
  git clone https://github.com/luohongk/SuperVINS.git
```

### 4.2 Data download

You can download the specified data set yourself, or you can use download_data.sh to download the data set. The download method is as follows.

```bash
cd ~/catkin_ws/src/SuperVINS
chmod +x download_data.sh
./download_data.sh
```

### 4.3 Path change

* file:vins_estimator\CMakeLists.txt , supervins_loop_fusion\CMakeLists.txt , camera_models\CMakeLists.txt

```bash
change
set(ONNXRUNTIME_ROOTDIR "/home/lhk/Thirdparty/onnxruntime")
find_package(Ceres REQUIRED PATHS "/home/lhk/Thirdparty/Ceres")
to
set(ONNXRUNTIME_ROOTDIR "your onnxruntime path")
find_package(Ceres REQUIRED PATHS "you Ceres path")
```

### 4.4 Compile project

```bash
cd ~/catkin_ws
catkin_make
```

### 4.5 Run the project

```bash
roslaunch supervins supervins_rviz.launch
rosrun supervins supervins_node ~/catkin_ws/src/SuperVINS/config/euroc/euroc_mono_imu_config.yaml
(SuperVINS1.0 currently does not support loop detection)rosrun supervins_loop_fusion supervins_loop_fusion_node ~/catkin_ws/src/SuperVINS/config/euroc/euroc_mono_imu_config.yaml
rosbag play ~/catkin_ws/src/SuperVINS/data/V2_01_easy.bag
```

### 4.6 Train vocabulary

```bash
Please refer to the official repository of DBoW3 yourself. This project does not provide
https://github.com/rmsalinas/DBow3
```

### 4.7 Paper citation method

```bash
@article{luo2025supervins,
  title={SuperVINS: A Real-Time Visual-Inertial SLAM Framework for Challenging Imaging Conditions},
  author={Luo, Hongkun and Liu, Yang and Guo, Chi and Li, Zengke and Song, Weiwei},
  journal={IEEE Sensors Journal},
  year={2025},
  publisher={IEEE}
}
```

# Thanks

* [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)，[SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)，[DBoW3](https://github.com/rmsalinas/DBow3)，[LightGlue-OnnxRunner](https://github.com/OroChippw/LightGlue-OnnxRunner)
* I would like to thank the [Wuhan University BRAIN Lab](https://www.zhiyuteam.com/) for its strong support for this project. Please continue to pay attention to the [latest research](https://zhiyuteam.com/html/web//yanjiuchengguo/qikanlunwen/index.html) of the Wuhan University BRAIN Lab. Welcome to follow Wuhan University BRAIN Lab WeChat Official Account！

<img src="./resources/BRAIN.png" style="height: 100px; ">
