import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)
from carla_ros_interfaces.msg import (
    EgoVehicleControl,
    EgoVehicleSteeringControl,
    VehicleControl,
    VehicleInfo
)
from .filters import ExponentialMovingAverageFilter
from .pid import PIDController


def clip(steering_angle):
    return max(min(steering_angle, 1.0), -1.0)


qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class VehicleControlNode(Node):
    def __init__(self):
        super().__init__("vehicle_control_node_node")

        # PID Controller, max speed = 30km/h
        self.pid_controller = PIDController(
            kp=1.0, ki=0.01, kd=0.05, max_integral=10.0, min_speed=0, max_speed=30.0
        )

        # Placeholder for the current vehicle control state
        self.filter = ExponentialMovingAverageFilter(alpha=0.09)

        # Placeholder for the current vehicle control state
        self.vehicle_control = VehicleControl()
        self.vehicle_control.steer = 0.0
        self.vehicle_control.throttle = 0.0
        self.vehicle_control.brake = 0.0
        self.vehicle_control.reverse = False

        # Vehicle Info
        self.vehicle_info = VehicleInfo()

        self.desired_speed = 0.0

        # Send the current vehicle control state to the bridge
        self.bridge_control_publisher = self.create_publisher(
            VehicleControl, "/carla_bridge/ego/control", 10
        )

        # Subscribe to the vehicle control topic
        self.bridge_control_subscription = self.create_subscription(
            VehicleControl, "/carla_bridge/ego/control", self.bridge_control_callback, 10
        )

        self.egp_info_subscription = self.create_subscription(
            VehicleInfo, 'carla_bridge/ego/vehicle_info', self.ego_info_callback, 10
        )

        self.ego_control_subscription = self.create_subscription(
            EgoVehicleControl, '/av/ego/control', self.ego_control_callback, 10
        )

        self.ego_steer_inference_subscription = self.create_subscription(
            EgoVehicleSteeringControl, "/av/ego/steer_inference", self.steer_inference_callback, 10
        )

        # set timer for updating the vehicle control state
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def bridge_control_callback(self, msg):
        # Store the current state of the vehicle control
        self.vehicle_control = msg

    def ego_control_callback(self, msg):
        self.desired_speed = msg.speed

        self.filter.add(msg.steer)
        self.vehicle_control.steer = self.filter.get_value()
        self.vehicle_control.brake = msg.brake
        self.vehicle_control.reverse = msg.reverse
        self.bridge_control_publisher.publish(self.vehicle_control)

    def steer_inference_callback(self, msg):
        self.filter.add(clip(msg.steer))
        self.vehicle_control.steer = self.filter.get_value()

        # Public the current vehicle control state
        self.bridge_control_publisher.publish(self.vehicle_control)

    def ego_info_callback(self, msg):
        # Store the current state of the vehicle control
        self.vehicle_info = msg

    def timer_callback(self):
        # Reduce the speed if the vehicle is turning
        if abs(self.vehicle_control.steer) > 0.4:
            desired_speed = self.desired_speed / 2.0
        else:
            desired_speed = self.desired_speed

        # Update the vehicle control state. Reduce the speed if the vehicle is turning
        throttle = self.pid_controller.control(float(desired_speed), float(self.vehicle_info.velocity))

        # Public the current vehicle control state
        self.vehicle_control.throttle = throttle
        self.bridge_control_publisher.publish(self.vehicle_control)


def main(args=None):
    rclpy.init(args=args)

    vehicle_control_node = VehicleControlNode()

    rclpy.spin(vehicle_control_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vehicle_control_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
