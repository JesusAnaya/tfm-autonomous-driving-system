FROM  osrf/ros:humble-desktop AS base

WORKDIR /ros2_ws

RUN apt update && apt install -y \
    ros-$ROS_DISTRO-rosbag2 \
    ros-$ROS_DISTRO-rosbag2-storage-mcap \
    ros-$ROS_DISTRO-cv-bridge

RUN apt update && apt install -y \
    python3-pip \
    python3.10-dev \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    setuptools==58.2.0 \
    opencv-python \
    numpy \
    pandas \
    torch \
    torchvision

COPY ./src /ros2_ws/src

RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build && rm -rf /ros2_ws/build /ros2_ws/log

COPY ./docker/recording-rosbag.sh /recording-rosbag.sh

COPY ./docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
