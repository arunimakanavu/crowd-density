import numpy as np
import openvino as ov


class CrowdDensityInference:
    """
    Wraps OpenVINO compiled CSRNet model for density map inference.
    """

    def __init__(self, model_path: str = "assets/models/csrnet.xml", device: str = "AUTO"):
        """
        Args:
            model_path: path to csrnet.xml
            device:     OpenVINO device string — AUTO, CPU, GPU, NPU
        """
        core = ov.Core()

        print(f"Loading model from {model_path} on device {device} ...")
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)

        self.infer_request = self.compiled_model.create_infer_request()

        # cache input/output node names
        self.input_node  = self.compiled_model.input(0)
        self.output_node = self.compiled_model.output(0)

        print("Model loaded successfully.")

    def infer(self, nchw_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on a preprocessed NCHW tensor.

        Args:
            nchw_tensor: float32 NCHW array from preprocess.py

        Returns:
            density_map: float32 array of shape (1, 1, H/8, W/8)
        """
        self.infer_request.infer({self.input_node: nchw_tensor})
        density_map = self.infer_request.get_output_tensor(0).data
        return density_map

    def get_input_shape(self) -> tuple:
        """
        Returns the expected input shape as (N, C, H, W).
        """
        return tuple(self.input_node.shape)
