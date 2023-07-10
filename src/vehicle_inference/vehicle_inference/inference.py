import cv2
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


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
        rgb_frame = image[60:-20, :, :]

        # Perform inference
        # Change the shape from (height, width, channels) to (batch_size, channels, height, width)
        # Size parameter is (width, height)
        # Change the image from 255 to 1.0

        rgb_frame = cv2.resize(rgb_frame, (200, 66))
        rgb_frame = np.float32(((rgb_frame / 255.0) - mean) / std)

        blob = cv2.dnn.blobFromImage(rgb_frame, scalefactor=1)

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
