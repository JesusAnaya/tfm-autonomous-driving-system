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
