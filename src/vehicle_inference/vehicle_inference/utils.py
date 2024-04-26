import numpy as np
import cv2
import rclpy
from sensor_msgs.msg import Image


def ros_image_to_opencv_rgb(image_msg: Image) -> np.ndarray:
    """
    Converts a ROS Image message to a numpy array (OpenCV image).

    Parameters
    ----------
    image_msg : sensor_msgs.msg.Image
        The ROS Image message to convert.

    Returns
    -------
    numpy.ndarray
        The converted OpenCV image.
    """
    # Convert the CARLA image from BGRA to RGB format
    if image_msg.encoding == "bgra8":
        # 'bgra8': 8-bit BGR color image with an alpha channel
        dtype = np.uint8
        n_channels = 4
    else:
        # Add conditions for other encodings as needed
        rclpy.logerr("Unsupported encoding: {}".format(image_msg.encoding))
        return None

    # Convert the raw data to a NumPy array with shape (height, width, channels)
    cv_image = np.ndarray(shape=(image_msg.height, image_msg.width, n_channels),
                          dtype=dtype, buffer=image_msg.data)

    # Assuming we want to discard the alpha channel and convert to RGB
    if image_msg.encoding == "bgra8":
        # Convert BGRA to BGR, then to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)

    return cv_image
