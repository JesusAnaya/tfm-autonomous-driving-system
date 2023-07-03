import rclpy
from rclpy.node import Node


class VehicleInferenceNode(Node):
    pass


def main(args=None):
    rclpy.init(args=args)

    vehicle_control_node = VehicleInferenceNode()

    rclpy.spin(vehicle_control_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vehicle_control_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
