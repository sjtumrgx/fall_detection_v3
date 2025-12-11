"""
NMS utilities for RKNN inference.
Pure numpy implementations copied from export/infer_rknn_pose.py and export/infer_rknn_seg.py
"""
import numpy as np
from typing import List


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center format to corner format.

    Args:
        x: (N, 4) array in format [center_x, center_y, width, height]

    Returns:
        (N, 4) array in format [x1, y1, x2, y2]
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes.

    Args:
        box1: (N, 4) array in xyxy format
        box2: (M, 4) array in xyxy format

    Returns:
        (N, M) array of IoU values
    """
    inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * \
        np.maximum(inter_y2 - inter_y1, 0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5) -> List[int]:
    """Pure numpy NMS for bounding boxes.

    From export/infer_rknn_pose.py lines 60-71

    Args:
        boxes: (N, 4) array in xyxy format
        scores: (N,) confidence scores
        iou_thres: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        iou = box_iou(boxes[i][None], boxes[idxs[1:]])[0]
        idxs = idxs[1:][iou <= iou_thres]
    return keep


def mask_nms(masks_bin: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5) -> List[int]:
    """Mask-based NMS for segmentation instances.

    From export/infer_rknn_seg.py lines 24-54

    Args:
        masks_bin: (N, H, W) binary masks
        scores: (N,) confidence scores
        iou_thres: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    N = masks_bin.shape[0]
    if N == 0:
        return []

    flat = masks_bin.reshape(N, -1).astype(np.uint8)
    areas = flat.sum(axis=1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        inter = np.minimum(flat[i], flat[rest]).sum(axis=1)
        union = areas[i] + areas[rest] - inter
        ious = inter / (union + 1e-6)

        rest = rest[ious <= iou_thres]
        order = rest

    return keep


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    From export/infer_rknn_seg.py lines 114-115

    Args:
        x: Input array

    Returns:
        Sigmoid of input
    """
    return 1.0 / (1.0 + np.exp(-x))
