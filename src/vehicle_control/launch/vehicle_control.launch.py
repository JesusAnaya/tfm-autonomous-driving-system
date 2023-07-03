from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    node = Node(
        package='vehicle_control',
        executable='vehicle_control_node',
        output='screen',
    )
    return LaunchDescription([node])
