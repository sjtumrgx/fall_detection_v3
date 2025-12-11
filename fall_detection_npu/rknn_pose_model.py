"""
RKNN Pose Model wrapper for YOL011n-pose on RK3588.
Encapsulates model loading, preprocessing, inference, and postprocessing.
"""
import os
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise RuntimeError(
        "rknnlite is required for NPU inference. "
        "Install with: pip install rknnlite"
    ) from exc

from nms_utils import xywh2xyxy, nms_boxes


def letterbox(im: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """Letterbox resize maintaining aspect ratio with padding.

    EXACT COPY from export/infer_rknn_pose.py lines 17-33
    CRITICAL: Do not modify this function

    Args:
        im: Input image (H, W, 3) BGR
        new_shape: Target size (square)
        color: Padding color

    Returns:
        Letterboxed image (new_shape, new_shape, 3)
    """
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im


class RKNNPoseModel:
    """RKNN pose detection model wrapper."""

    def __init__(self, model_path: str, imgsz: int = 640, core_mask: Optional[int] = None):
        """Initialize RKNN pose model.

        Args:
            model_path: Path to .rknn file
            imgsz: Input image size (square)
            core_mask: NPU core selection (e.g., RKNNLite.NPU_CORE_0)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model not found: {model_path}")

        self.model_path = model_path
        self.imgsz = imgsz
        self.rknn = RKNNLite()

        # Load model
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")

        # Initialize runtime
        if core_mask is None:
            core_mask = getattr(RKNNLite, "NPU_CORE_0", None)

        if core_mask is not None:
            ret = self.rknn.init_runtime(core_mask=core_mask)
        else:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime")

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess frame for RKNN inference.

        Args:
            frame_bgr: Input frame (H, W, 3) BGR

        Returns:
            x: (1, imgsz, imgsz, 3) NHWC float32 for inference
            img_letterbox: (imgsz, imgsz, 3) letterboxed image for visualization
        """
        img_letterbox = letterbox(frame_bgr, self.imgsz)
        x = img_letterbox.astype(np.float32)
        # CRITICAL: NO /255 normalization - built into RKNN model
        x = x[np.newaxis, ...]  # Add batch dimension -> (1, imgsz, imgsz, 3)
        return x, img_letterbox

    def inference(self, x: np.ndarray) -> np.ndarray:
        """Run RKNN inference.

        Args:
            x: (1, imgsz, imgsz, 3) NHWC float32

        Returns:
            (8400, 56) array: [cx, cy, w, h, conf, kpt1_x, kpt1_y, kpt1_conf, ...]
        """
        outputs = self.rknn.inference(inputs=[x])
        pred = outputs[0]

        # Handle shape variations: (1, 56, 8400, 1) or (1, 56, 8400)
        # From export/infer_rknn_pose.py lines 116-130
        if pred.ndim == 4 and pred.shape[0] == 1 and pred.shape[-1] == 1:
            pred = np.squeeze(pred, axis=(0, 3))  # -> (56, 8400)
        else:
            pred = np.squeeze(pred)

        # Transpose to (8400, 56)
        if pred.shape[0] == 56 and pred.shape[1] == 8400:
            pred = pred.transpose(1, 0)
        elif pred.shape[0] == 8400 and pred.shape[1] == 56:
            pass  # Already correct
        else:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}, expected (8400, 56) or (56, 8400)")

        return pred

    def postprocess(
        self,
        pred: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply NMS and extract boxes, scores, keypoints.

        Args:
            pred: (8400, 56) raw predictions
            conf_thres: Confidence threshold
            iou_thres: NMS IoU threshold

        Returns:
            boxes_xyxy: (N, 4) in letterbox coordinates
            scores: (N,)
            keypoints: (N, 17, 3) - x, y, conf for each keypoint
        """
        boxes_xywh = pred[:, :4]
        scores = pred[:, 4]
        kpts_raw = pred[:, 5:]

        # Confidence filtering
        mask = scores > conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        kpts_raw = kpts_raw[mask]

        if len(boxes_xywh) == 0:
            return (np.empty((0, 4)), np.empty((0,)), np.empty((0, 17, 3)))

        # Convert to xyxy for NMS
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        keep = nms_boxes(boxes_xyxy, scores, iou_thres)

        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        kpts_raw = kpts_raw[keep]

        # Reshape keypoints to (N, 17, 3)
        num_kpts = (pred.shape[1] - 5) // 3  # Should be 17 for COCO format
        keypoints = kpts_raw.reshape(-1, num_kpts, 3)

        return boxes_xyxy, scores, keypoints

    def release(self):
        """Clean up RKNN resources."""
        if hasattr(self, 'rknn'):
            self.rknn.release()
