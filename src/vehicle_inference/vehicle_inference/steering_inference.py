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
from cv_bridge import CvBridge, CvBridgeError
from carla_ros_interfaces.msg import EgoVehicleSteeringControl
import cv2

# Create a custom QoS profile
qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def crop_down(image, top=60, bottom=15):
    return image[top:-bottom or None, :, :]


class VehicleInferenceNode(Node):
    def __init__(self):
        super().__init__("vehicle_inference_node")

        # Action inference
        self.inference_on = False

        # CV Bridge to convert ROS Image to OpenCV image
        self.bridge = CvBridge()

        self.camera_center_subscription = self.create_subscription(
            Image, "/carla_bridge/ego/camera_center", self.image_callback, qos_profile=qos
        )

        self.turn_on_inference_subscription = self.create_subscription(
            Bool, "/av/ego/turn_on_inference", self.turn_on_inference_callback, 10
        )

        self.ego_vehicle_steering_publisher = self.create_publisher(
            EgoVehicleSteeringControl, "/av/ego/steer_inference", 10
        )

    def turn_on_inference_callback(self, msg):
        if msg.data == self.inference_on:
            return

        self.inference_on = msg.data

    def image_callback(self, msg):
        if not self.inference_on:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"Failed to convert image: {str(e)}")
            return

        # Crop the image
        cropped_image = crop_down(cv_image)

        # resize the image
        frame = cv2.resize(cropped_image, (200, 66))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize the image to [0, 1]
        rgb_frame = (rgb_frame / 255.0).astype('float32')

        # Perform inference
        # Change the shape from (height, width, channels) to (batch_size, channels, height, width)
        blob = cv2.dnn.blobFromImage(rgb_frame)

        net = cv2.dnn.readNetFromONNX('/models/drive_net_model.onnx')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        net.setInput(blob)
        output = net.forward()

        # Assume output is a single value tensor representing the steering angle
        steer_angle = float(output[0][0])

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
