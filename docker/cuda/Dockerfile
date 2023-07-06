# Stage 1: Build and install OpenCV with CUDA
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS opencv_builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev

# Download OpenCV and OpenCV contrib
RUN mkdir /opt/opencv && cd /opt/opencv && \
    git clone --branch 4.7.0 --depth 1 https://github.com/opencv/opencv.git && \
    git clone --branch 4.7.0 --depth 1 https://github.com/opencv/opencv_contrib.git

# Create build directory and switch to it
RUN mkdir /opt/opencv/opencv/build && cd /opt/opencv/opencv/build

WORKDIR /opt/opencv/build

# Run cmake to configure build. Enable only what you need!
RUN cd /opt/opencv/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=8.6 \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_java=OFF \
    -D WITH_GSTREAMER=OFF \
    -D WITH_GTK=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv/opencv_contrib/modules \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7 \
    ../

# Install OpenCV
RUN cd /opt/opencv/opencv/build && make -j$(nproc) && make install

# Remove unnecessary OpenCV source files
WORKDIR /opt
RUN rm -rf /opt/opencv

# Stage 2: Install ROS and copy OpenCV from previous stage
FROM  osrf/ros:humble-desktop AS base
COPY --from=opencv_builder /usr/local /usr/local
COPY --from=opencv_builder /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/

WORKDIR /ros2_ws

RUN apt update && apt install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    libgtk2.0-0

RUN pip install --upgrade pip \
    setuptools==58.2.0

COPY ./src/carla_ros_interfaces /ros2_ws/src/carla_ros_interfaces
COPY ./src/vehicle_inference /ros2_ws/src/vehicle_inference

RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build --packages-select carla_ros_interfaces vehicle_inference && \
    rm -rf /ros2_ws/build /ros2_ws/log

ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

COPY ./docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

