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
    pandas

COPY ./src/carla_ros_interfaces /ros2_ws/src/carla_ros_interfaces
COPY ./src/data_collector /ros2_ws/src/data_collector

RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select carla_ros_interfaces data_collector && \
    rm -rf /ros2_ws/build /ros2_ws/log

COPY ./docker/data_collector/entrypoint.sh /entrypoint-data-collector.sh
RUN chmod +x /entrypoint-data-collector.sh
ENTRYPOINT ["/entrypoint-data-collector.sh"]
