# SuperVINS Dockerfile（ Hongkun Luo email:luohongkun0715@gmail.com ）
# Base: Ubuntu 20.04 + CUDA 11.8 + ROS Noetic

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    lsb-release \
    gnupg2 \
    ca-certificates \
    software-properties-common \
    python3-pip \
    python3-dev \
    libboost-all-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-tf \
    ros-noetic-nav-msgs \
    ros-noetic-geometry-msgs \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true && rosdep update

# Source ROS in bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc


RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget -q https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0.tar.gz && \
    tar -xzf 2.1.0.tar.gz && \
    cd ceres-solver-2.1.0 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF && \
    make -j8 && \
    make install && \
    cd /tmp && rm -rf ceres-solver-2.1.0 2.1.0.tar.gz


WORKDIR /opt
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.16.3.tgz && \
    mv onnxruntime-linux-x64-gpu-1.16.3 /opt/onnxruntime && \
    rm onnxruntime-linux-x64-gpu-1.16.3.tgz

ENV ONNXRUNTIME_ROOTDIR=/opt/onnxruntime
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_ROOTDIR}/lib:${LD_LIBRARY_PATH}

# Create symlinks so that hardcoded paths in CMakeLists.txt work
# even when source is mounted from host (bypassing sed modifications)
RUN mkdir -p /home/lhk/Thirdparty && \
    ln -s /opt/onnxruntime /home/lhk/Thirdparty/onnxruntime && \
    ln -s /usr/local /home/lhk/Thirdparty/Ceres


RUN mkdir -p /root/catkin_ws/src
COPY .. /root/catkin_ws/src/SuperVINS


RUN sed -i 's|set(ONNXRUNTIME_ROOTDIR.*)|set(ONNXRUNTIME_ROOTDIR "/opt/onnxruntime")|g' \
    /root/catkin_ws/src/SuperVINS/supervins_estimator/CMakeLists.txt \
    /root/catkin_ws/src/SuperVINS/supervins_loop_fusion/CMakeLists.txt && \
    sed -i 's|find_package(Ceres REQUIRED PATHS.*)|find_package(Ceres REQUIRED PATHS "/usr/local")|g' \
    /root/catkin_ws/src/SuperVINS/supervins_estimator/CMakeLists.txt \
    /root/catkin_ws/src/SuperVINS/supervins_loop_fusion/CMakeLists.txt \
    /root/catkin_ws/src/SuperVINS/camera_models/CMakeLists.txt

WORKDIR /root/catkin_ws
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && [ -f /root/catkin_ws/devel/setup.bash ] && source /root/catkin_ws/devel/setup.bash; exec \"$@\"", "--"]
CMD ["bash"]
