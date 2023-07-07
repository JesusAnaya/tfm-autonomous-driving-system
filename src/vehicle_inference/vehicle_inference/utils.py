import numpy as np
from sensor_msgs.msg import Image


def ros_to_opencv(image_msg: Image) -> np.ndarray:
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
    # Convert the ROS Image message to a numpy array
    image_arr = np.frombuffer(
        image_msg.data, dtype=np.uint8
    ).reshape(image_msg.height, image_msg.width, -1)

    # If the image has a 'bgr8' encoding, convert it to RGB
    if image_msg.encoding == 'bgr8':
        image_arr = image_arr[..., ::-1]  # Reverse the color channels

    elif image_msg.encoding != 'rgb8':
        raise NotImplementedError(f"Encoding '{image_msg.encoding}' is not supported.")

    return image_arr
