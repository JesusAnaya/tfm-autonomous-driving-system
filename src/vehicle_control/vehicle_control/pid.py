import time
from typing import Optional


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, max_integral: float, min_speed: float, max_speed: float):
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.max_integral: float = max_integral
        self.min_speed: float = min_speed
        self.max_speed: float = max_speed
        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.prev_time: Optional[float] = time.time()

    def control(self, desired_speed: float, current_speed: float) -> float:
        current_time: float = time.time()
        dt: float = current_time - self.prev_time
        error: float = desired_speed - current_speed

        self.integral += error * dt
        self.integral = max(min(self.integral, self.max_integral), -self.max_integral)

        derivative: float = (error - self.prev_error) / dt if dt > 0 else 0.0

        output: float = self.kp*error + self.ki*self.integral + self.kd*derivative
        output = max(min(output, 1.0), 0.0)  # assuming throttle value ranges between 0 and 1

        self.prev_error = error
        self.prev_time = current_time

        return output


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.state = None

    def filter(self, value):
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state


class SteeringPIDController:
    def __init__(self, kp: float, ki: float, kd: float, max_integral: float, alpha: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.error_filter = LowPassFilter(alpha)  # Initialize the filter here for the error

    def control(self, desired_steering: float, current_steering: float) -> float:
        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time else 0
        raw_error = desired_steering - current_steering

        # Filter the error
        error_filtered = self.error_filter.filter(raw_error)

        # Integral is calculated based on the filtered error
        self.integral += error_filtered * dt
        self.integral = max(min(self.integral, self.max_integral), -self.max_integral)

        # Derivative is calculated based on the filtered error
        derivative = (error_filtered - self.prev_error) / dt if dt > 0 else 0.0

        # PID output calculation using filtered error
        output = self.kp * error_filtered + self.ki * self.integral + self.kd * derivative
        output = max(min(output, 1.0), -1.0)

        self.prev_error = error_filtered
        self.prev_time = current_time

        return output
