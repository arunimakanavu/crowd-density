import cv2
import numpy as np

# ImageNet mean and std — CSRNet was trained with these
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame: np.ndarray, input_height: int = 720, input_width: int = 1280) -> np.ndarray:
    """
    Prepare a raw BGR frame from OpenCV for CSRNet inference.

    Args:
        frame:        BGR frame from cv2.VideoCapture
        input_height: model input height
        input_width:  model input width

    Returns:
        NCHW float32 tensor ready for OpenVINO infer request
    """

    # resize
    resized = cv2.resize(frame, (input_width, input_height))

    # BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # HWC uint8 → HWC float32 in [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # subtract ImageNet mean, divide by std — channel-wise
    normalized = (normalized - MEAN) / STD

    # HWC → CHW
    chw = np.transpose(normalized, (2, 0, 1))

    # CHW → NCHW (add batch dim)
    nchw = np.expand_dims(chw, axis=0)

    return nchw


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """
    Extract basic properties from an opened VideoCapture object.

    Returns:
        dict with width, height, fps, frame_count
    """
    return {
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
