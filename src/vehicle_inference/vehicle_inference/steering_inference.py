import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from carla_ros_interfaces.msg import EgoVehicleSteeringControl
from .inference import InferenceOpenCV
from .utils import ros_image_to_opencv_rgb

# Create a custom QoS profile
qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class VehicleInferenceNode(Node):
    def __init__(self):
        super().__init__("vehicle_inference_node")

        # Action inference
        self.inference_on = False
        self.inference = InferenceOpenCV("/models/drive_net_model.onnx")

        self.camera_center_subscription = self.create_subscription(
            Image, "/carla_bridge/ego/camera_center", self.image_callback, qos_profile=qos
        )

        self.turn_on_inference_subscription = self.create_subscription(
            Bool, "/av/ego/turn_on_inference", self.turn_on_inference_callback, 10
        )

        self.ego_vehicle_steering_publisher = self.create_publisher(
            EgoVehicleSteeringControl, "/av/ego/steer_inference", 10
        )

    def turn_on_inference_callback(self, msg: Bool):
        if msg.data == self.inference_on:
            return

        self.inference_on = msg.data

    def image_callback(self, msg_image: Image):
        if not self.inference_on:
            return

        # Convert ROS Image to OpenCV image
        cv_image = ros_image_to_opencv_rgb(msg_image)

        # Run inference
        steer_angle = self.inference.run_inference(cv_image)

        # Print warning if any doing inference
        if self.inference.get_warning():
            self.get_logger().warn(self.inference.get_warning())

        # Publish the control command
        ego_vehicle_steering_control = EgoVehicleSteeringControl()
        ego_vehicle_steering_control.steer = steer_angle
        self.ego_vehicle_steering_publisher.publish(ego_vehicle_steering_control)


def main(args=None):
    rclpy.init(args=args)

    vehicle_inference_node = VehicleInferenceNode()

    rclpy.spin(vehicle_inference_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vehicle_inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
