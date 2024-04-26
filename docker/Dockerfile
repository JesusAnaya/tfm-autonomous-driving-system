# Stage 1: Build and install OpenCV with CUDA
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS opencv_builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev && \
    rm -rf /var/lib/apt/lists/*

# Download OpenCV and OpenCV contrib
RUN mkdir /opt/opencv && cd /opt/opencv && \
    git clone --branch 4.9.0 --depth 1 https://github.com/opencv/opencv.git && \
    git clone --branch 4.9.0 --depth 1 https://github.com/opencv/opencv_contrib.git

# Create build directory and switch to it
RUN mkdir /opt/opencv/opencv/build && cd /opt/opencv/opencv/build

WORKDIR /opt/opencv/build

# Run cmake to configure build. Enable only what you need!
RUN cd /opt/opencv/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
    -D CUDA_ARCH_BIN=8.6 \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_java=OFF \
    -D WITH_GSTREAMER=OFF \
    -D WITH_GTK=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv/opencv_contrib/modules \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 \
    ../

# Install OpenCV
RUN cd /opt/opencv/opencv/build && make -j$(nproc) && make install


# Stage 2: Build ROS2 workspace
FROM  ros:humble AS builder

WORKDIR /ros2_ws

RUN apt update && apt install -y \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    setuptools==58.2.0 \
    numpy \
    pandas

COPY ./src/ /ros2_ws/src/

RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build && \
    rm -rf /ros2_ws/build /ros2_ws/log

# Stage 3: Install ROS and copy OpenCV from previous stage
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS ros-humble-cuda-cudnn

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG en_US.UTF-8
ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV COLCON_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib
ENV PATH=/opt/ros/humble/bin:$PATH
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2
ENV DEBIAN_FRONTEND=""

# Install language, timezone, common programs and ROS2
RUN apt-get update && apt-get install -y \
    locales \
    tzdata \
    curl \
    gnupg2 \
    software-properties-common \
    python3-argcomplete \
    python3-pip \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-core \
    && rm -rf /var/lib/apt/lists/*

# Stage 4: Install ROS2 and copy OpenCV from previous stage
FROM carla_ros_bridge:latest as bridge
FROM ros-humble-cuda-cudnn AS base

# Copy OpenCV libraries from build stage
COPY --from=opencv_builder /opt/opencv/opencv_contrib/modules/ /opt/opencv/opencv_contrib/modules/
COPY --from=opencv_builder /usr/local/lib/python3.10/dist-packages/cv2/ /usr/local/lib/python3.10/dist-packages/cv2/
COPY --from=opencv_builder /usr/local/lib/libopencv_* /usr/local/lib/
COPY --from=bridge /opt/carla-ros-bridge/install /opt/carla-ros-bridge/install

RUN apt update && apt install -y \
    libdc1394-25 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy pandas

WORKDIR /ros2_ws

# Copy ROS2 workspace from previous stage
COPY --from=builder /ros2_ws/install /ros2_ws/install

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

COPY ./docker/recording-rosbag.sh /recording-rosbag.sh
COPY ./docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
