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
from .model import NvidiaModel
import torch
import torchvision.transforms as transforms
import cv2

# Create a custom QoS profile
qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def crop_down(image, top=70, bottom=20):
    return image[top:-bottom or None, :, :]


class VehicleInferenceNode(Node):
    def __init__(self):
        super().__init__("vehicle_inference_node")

        # Action inference
        self.inference_on = False

        self.device = get_device()

        # Load your pretrained PyTorch model here
        self.model = NvidiaModel()
        self.model.load_state_dict(torch.load("/models/model.pt", map_location=torch.device(self.device)))
        self.model.to(self.device)  # Move the model to the device
        self.model.eval()  # Set the model to evaluation mode

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

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        frame_torch = transforms.functional.to_tensor(rgb_frame).to(self.device)
        batch_t = torch.unsqueeze(frame_torch, 0)

        with torch.no_grad():
            output = self.model(batch_t)

        # Assume output is a single value tensor representing the steering angle
        steer_angle = output.item()

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
