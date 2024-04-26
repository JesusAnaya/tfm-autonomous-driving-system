import cv2
import numpy as np


class InferenceOpenCV(object):
    def __init__(self, model_onnx_path: str):
        self.model: cv2.dnn_Net = cv2.dnn.readNet(model_onnx_path)
        self.warning: bool = False
        self.warning_message: str = ''

        # Check if CUDA is available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print('CUDA is available.')
            # Use CUDA
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print('CUDA is not available.')
            # Use CPU
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # If CUDA is not available, set warning
            self.warning = True
            self.warning_message = 'CUDA is not available. Using CPU instead.'

    def run_inference(self, image: np.ndarray) -> float:
        # Crop the image
        frame = image.copy()

        # Perform inference
        # Resize the image to 200x66
        frame = cv2.resize(frame, (224, 224))

        # Normalize the image
        frame = frame.astype(np.float32) / 255.0

        # Convert the image from RGB to a blog for input to the model
        blob = cv2.dnn.blobFromImage(frame, 1.0)

        # change (None, 3, 66, 200) to (None, 3, 200, 66)
        blob = blob.transpose(0, 2, 3, 1)

        # Set the input to the model
        self.model.setInput(blob)

        # Run the forward pass
        output = self.model.forward()

        # Get the value from the output
        value = output.flatten().item()

        # Assume output is a single value tensor representing the steering angle
        steer_angle = round(float(value), 4)

        return steer_angle

    def get_warning(self) -> bool:
        return self.warning

    def get_warning_message(self) -> str:
        return self.warning_message
