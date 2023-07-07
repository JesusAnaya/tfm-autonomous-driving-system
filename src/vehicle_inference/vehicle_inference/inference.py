import cv2
import numpy as np


def crop_down(image, top=75, bottom=20):
    return image[top:-bottom or None, :, :]


class InferenceOpenCV(object):
    def __init__(self, model_onnx_path: str):
        self.model: cv2.dnn_Net = cv2.dnn.readNetFromONNX(model_onnx_path)
        self.warning: bool = False
        self.warning_message: str = ''

        # Check if CUDA is available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Use CUDA
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            # Use CPU
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # If CUDA is not available, set warning
            self.warning = True
            self.warning_message = 'CUDA is not available. Using CPU instead.'

    def run_inference(self, image: np.ndarray) -> float:
        # Crop the image
        cropped_image = crop_down(image)

        # resize the image
        frame = cv2.resize(cropped_image, (200, 66))

        # Normalize the image to [0, 1]
        rgb_frame = (frame / 255.0).astype('float32')

        # Apply a bit of blur
        rgb_frame = cv2.GaussianBlur(rgb_frame, (3, 3), 0)

        # Perform inference
        # Change the shape from (height, width, channels) to (batch_size, channels, height, width)
        blob = cv2.dnn.blobFromImage(rgb_frame)

        # Set the input to the model
        self.model.setInput(blob)

        # Run the forward pass
        output = self.model.forward()

        # Assume output is a single value tensor representing the steering angle
        steer_angle = float(output[0][0])

        return steer_angle

    def get_warning(self) -> bool:
        return self.warning

    def get_warning_message(self) -> str:
        return self.warning_message
