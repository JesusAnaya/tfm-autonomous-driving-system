from threading import Thread
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from carla_msgs.msg import CarlaEgoVehicleControl
from .inference import InferenceOpenCV
from .utils import ros_image_to_opencv_rgb


class VehicleInferenceNode(CompatibleNode):
    def __init__(self):
        super().__init__("vehicle_inference_node")
        fast_qos = QoSProfile(depth=10)
        self.role_name = self.get_param("role_name", "ego_vehicle")

        # Action inference
        self.inference_on = False
        self.inference = InferenceOpenCV("/models/drive_net_model.onnx")

        self.camera_center_subscription = self.new_subscription(
            Image,
            "/carla/{}/rgb_center/image".format(self.role_name),
            self.image_callback,
            qos_profile=fast_qos
        )

        self.turn_on_inference_subscription = self.new_subscription(
            Bool,
            "/carla/{}/turn_on_inference".format(self.role_name),
            self.turn_on_inference_callback, 10
        )

        self.ego_vehicle_steer_inference_publisher = self.new_publisher(
            CarlaEgoVehicleControl,
            "/carla/{}/steer_inference".format(self.role_name),
            10
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
        ego_vehicle_control = CarlaEgoVehicleControl()
        ego_vehicle_control.steer = min(max(steer_angle, -1.0), 1.0)
        self.ego_vehicle_steer_inference_publisher.publish(ego_vehicle_control)


def main(args=None):
    roscomp.init("vehicle_control_node_node", args=args)
    try:
        vehicle_inference_node = VehicleInferenceNode()
        executor = roscomp.executors.MultiThreadedExecutor()
        executor.add_node(vehicle_inference_node)
        vehicle_inference_node.spin()
    except (IOError, RuntimeError) as e:
        roscomp.logerr("Error: {}".format(e))
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()
