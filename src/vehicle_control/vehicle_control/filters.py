import numpy as np


class AverageFilter:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.data = np.array([])

    def add(self, value):
        self.data = np.append(self.data, value)
        if len(self.data) > self.max_size:
            self.data = np.delete(self.data, 0)

    def get_average(self):
        return np.average(self.data) if len(self.data) else 0

    def get_value(self):
        return self.get_average()


class MedianFilter:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.data = []

    def add(self, value):
        self.data.append(value)
        self.data = self.data[-self.max_size:]

    def get_value(self):
        return np.median(self.data) if self.data else 0


class ExponentialSmoothingFilter(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def add(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value

    def get_value(self):
        return self.value


class ExponentialMovingAverageFilter(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = 0.0

    def add(self, value):
        self.value = self.alpha * value + (1 - self.alpha) * self.value

    def get_value(self):
        return self.value


class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        # Initializes the Kalman Filter
        self.process_variance = process_variance  # Variance of the vehicle model
        self.measurement_variance = measurement_variance  # Variance of the sensor output
        self.estimated_measurement_variance = estimated_measurement_variance  # Initially estimated variance of the measurement

        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Measurement update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate
