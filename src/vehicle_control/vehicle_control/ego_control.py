from threading import Thread
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from std_msgs.msg import Bool
from .filters import MedianFilter, ExponentialMovingAverageFilter, KalmanFilter
from .pid import PIDController, SteeringPIDController


def clip(steering_angle):
    return max(min(steering_angle, 1.0), -1.0)


class VehicleControlNode(CompatibleNode):
    def __init__(self):
        super().__init__("vehicle_control_node_node")

        fast_qos = QoSProfile(depth=10)
        fast_latched_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.role_name = self.get_param("role_name", "ego_vehicle")
        self.inference_on = False

        # PID Controller, max speed = 30km/h
        self.pid_controller = PIDController(
            kp=1.0, ki=0.01, kd=0.05, max_integral=10.0, min_speed=0, max_speed=30.0
        )

        # Placeholder for the current vehicle control state
        self.filter = ExponentialMovingAverageFilter(alpha=0.08)
        self.kalmam_filter = KalmanFilter(
            process_variance=1e-5,
            measurement_variance=1e-4,
            estimated_measurement_variance=1e-5
        )

        # Placeholder for the current vehicle control state
        self.vehicle_control = CarlaEgoVehicleControl()
        self.vehicle_control.steer = 0.0
        self.vehicle_control.throttle = 0.0
        self.vehicle_control.brake = 0.0
        self.vehicle_control.reverse = False

        # Vehicle Info
        self.vehicle_status = CarlaEgoVehicleStatus()

        # Desired speed in km/h
        self.desired_speed = 5.0

        self.vehicle_status_subscriber = self.new_subscription(
            CarlaEgoVehicleStatus, "/carla/{}/vehicle_status".format(self.role_name),
            self.vehicle_status_updated, qos_profile=10
        )

        self.vehicle_control_publisher = self.new_publisher(
            CarlaEgoVehicleControl,
            "/carla/{}/vehicle_control_cmd_manual".format(self.role_name),
            qos_profile=fast_qos
        )

        self.vehicle_inference_subscription = self.new_subscription(
            CarlaEgoVehicleControl,
            "/carla/{}/steer_inference".format(self.role_name),
            self.vehicle_inference_callback, 10
        )

        self.turn_on_inference_subscription = self.new_subscription(
            Bool,
            "/carla/{}/turn_on_inference".format(self.role_name),
            self.turn_on_inference_callback, 10
        )

        self.vehicle_control_manual_override_publisher = self.new_publisher(
            Bool,
            "/carla/{}/vehicle_control_manual_override".format(self.role_name),
            qos_profile=fast_latched_qos
        )

    def turn_on_inference_callback(self, msg: Bool):
        if msg.data == self.inference_on:
            return

        self.vehicle_control_manual_override_publisher.publish((Bool(data=msg.data)))

        if not msg.data:
            self.vehicle_control.steer = 0.0
            self.vehicle_control.throttle = 0.0
            self.vehicle_control.brake = 0.0

        self.inference_on = msg.data

    def vehicle_inference_callback(self, msg):
        if not self.inference_on:
            return

        # self.filter.add(clip(msg.steer))
        # steer = self.filter.get_value()

        steer = self.kalmam_filter.update(msg.steer)
        self.vehicle_control.steer = clip(steer)

        # Public the current vehicle control state
        self.vehicle_control_publisher.publish(self.vehicle_control)

    def vehicle_status_updated(self, msg: CarlaEgoVehicleStatus):
        if not self.inference_on:
            return

        self.vehicle_control.steer = msg.control.steer
        self.vehicle_control.brake = msg.control.brake

        # Reduce the speed if the vehicle is turning
        if abs(msg.control.steer) > 0.3:
            desired_speed = self.desired_speed / 2.0
        else:
            desired_speed = self.desired_speed

        # Update the vehicle control state. Reduce the speed if the vehicle is turning
        velocity_kmh = 3.6 * msg.velocity
        throttle = self.pid_controller.control(float(desired_speed), float(velocity_kmh))
        self.vehicle_control.throttle = throttle

        # Public the current vehicle control state
        self.vehicle_control_publisher.publish(self.vehicle_control)


def main(args=None):
    roscomp.init("vehicle_control_node_node", args=args)
    try:
        vehicle_control_node = VehicleControlNode()
        executor = roscomp.executors.MultiThreadedExecutor()
        executor.add_node(vehicle_control_node)
        vehicle_control_node.spin()
    except (IOError, RuntimeError) as e:
        roscomp.logerr("Error: {}".format(e))
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()
