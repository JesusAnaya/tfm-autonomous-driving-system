FROM  osrf/ros:humble-desktop AS base

WORKDIR /ros2_ws

RUN apt update && apt install -y \
    ros-$ROS_DISTRO-cv-bridge

RUN apt update && apt install -y \
    python3-pip \
    python3.10-dev \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    setuptools==58.2.0 \
    opencv-python \
    torch \
    torchvision

COPY ./src/carla_ros_interfaces /ros2_ws/src/carla_ros_interfaces
COPY ./src/vehicle_inference /ros2_ws/src/vehicle_inference

RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select  carla_ros_interfaces vehicle_inference && \
    rm -rf /ros2_ws/build /ros2_ws/log

COPY ./docker/data_collector/entrypoint.sh /entrypoint-vehicle-inference.sh
RUN chmod +x /entrypoint-vehicle-inference.sh
ENTRYPOINT ["/entrypoint-vehicle-inference.sh"]
