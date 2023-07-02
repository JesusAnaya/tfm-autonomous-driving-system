!/usr/bash

set -e

source /opt/ros/$ROS_DISTRO/setup.bash

ros2 bag record -s mcap -o /data/redosding_1.mcap \
  /carla/vehicle/control /carla/vehicle/camera_center \
  /carla/vehicle/camera_left /carla/vehicle/camera_right
