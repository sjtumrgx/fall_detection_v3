"""
RKNN Segmentation Model wrapper for YOLO11n-seg on RK3588.
Used for bed detection in fall detection system.
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

from nms_utils import xywh2xyxy, nms_boxes, mask_nms, sigmoid


def letterbox(im: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Letterbox resize maintaining aspect ratio with padding.

    Based on export/infer_rknn_seg.py lines 57-73
    Modified to return actual content size (MUST MATCH rknn_pose_model.letterbox)

    Args:
        im: Input image (H, W, 3) BGR
        new_shape: Target size (square)
        color: Padding color

    Returns:
        im_letterboxed: Letterboxed image (new_shape, new_shape, 3)
        content_size: (content_width, content_height) - actual content area before padding
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
    return im, new_unpad  # Return (image, (width, height))


class RKNNSegModel:
    """RKNN segmentation model wrapper for bed detection."""

    def __init__(self, model_path: str, imgsz: int = 640, core_mask: Optional[int] = None):
        """Initialize RKNN segmentation model.

        Args:
            model_path: Path to .rknn file
            imgsz: Input image size (square)
            core_mask: NPU core selection (e.g., RKNNLite.NPU_CORE_0)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model not found: {model_path}")

        self.model_path = model_path
        self.imgsz = imgsz
        self.NUM_CLASSES = 80  # COCO classes
        self.NUM_MASKS = 32  # Prototype masks
        self.PROTO_H = 160
        self.PROTO_W = 160
        self.BED_CLASS_ID = 59  # COCO bed class

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

    def inference(
        self,
        frame_bgr: np.ndarray,
        conf_thres: float = 0.3,
        iou_thres: float = 0.45,
        mask_thres: float = 0.5
    ) -> Optional[np.ndarray]:
        """Detect bed and return binary mask.

        Args:
            frame_bgr: Input frame (any size, will be letterboxed)
            conf_thres: Confidence threshold
            iou_thres: NMS threshold
            mask_thres: Mask binarization threshold

        Returns:
            Binary mask (imgsz, imgsz) uint8 or None if no bed detected
        """
        # 1. Letterbox and preprocess
        img, _ = letterbox(frame_bgr, self.imgsz)  # Discard content_size (not needed for bed detection)
        x = img.astype(np.float32)[np.newaxis, ...]  # (1, imgsz, imgsz, 3) NHWC
        # CRITICAL: NO /255 normalization

        # 2. RKNN inference
        outputs = self.rknn.inference(inputs=[x])
        if len(outputs) != 2:
            raise RuntimeError(f"Expected 2 outputs, got {len(outputs)}")

        # 3. Parse outputs: proto and pred
        # From export/infer_rknn_seg.py lines 160-183
        out0, out1 = outputs

        # Determine which is proto based on channel count
        if out0.shape[1] == self.NUM_MASKS:  # 32
            proto_raw, pred_raw = out0, out1
        else:
            proto_raw, pred_raw = out1, out0

        proto = np.squeeze(proto_raw, axis=0)  # (32, 160, 160)

        # Handle pred shape variations
        pred = pred_raw
        if pred.ndim == 4 and pred.shape[0] == 1 and pred.shape[-1] == 1:
            pred = np.squeeze(pred, axis=(0, 3))
        else:
            pred = np.squeeze(pred)

        if pred.shape[0] == 116:  # (116, 8400)
            pred = pred.transpose(1, 0)  # -> (8400, 116)

        # 4. Split into components
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:4 + self.NUM_CLASSES]
        mask_coeffs = pred[:, 4 + self.NUM_CLASSES:]

        # 5. Apply sigmoid to class scores
        cls_scores = sigmoid(cls_scores)

        # 6. Filter by confidence and bed class
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        mask_keep = scores > conf_thres
        boxes_xywh = boxes_xywh[mask_keep]
        scores = scores[mask_keep]
        cls_ids = cls_ids[mask_keep]
        mask_coeffs = mask_coeffs[mask_keep]

        # Filter by bed class
        bed_mask = (cls_ids == self.BED_CLASS_ID)
        boxes_xywh = boxes_xywh[bed_mask]
        scores = scores[bed_mask]
        mask_coeffs = mask_coeffs[bed_mask]

        if len(boxes_xywh) == 0:
            return None

        # 7. Box NMS
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        keep = nms_boxes(boxes_xyxy, scores, iou_thres)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        mask_coeffs = mask_coeffs[keep]

        if len(boxes_xyxy) == 0:
            return None

        # 8. Generate masks (lines 251-261)
        proto_flat = proto.reshape(self.NUM_MASKS, -1)  # (32, 25600)
        masks = mask_coeffs @ proto_flat  # (N, 25600)
        masks = sigmoid(masks)
        masks = masks.reshape(-1, self.PROTO_H, self.PROTO_W)  # (N, 160, 160)

        # Upsample to imgsz×imgsz
        up_masks = []
        for i in range(len(masks)):
            m = cv2.resize(masks[i], (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            up_masks.append(m)
        up_masks = np.stack(up_masks, axis=0)  # (N, imgsz, imgsz)

        # 9. Mask NMS (lines 264-273)
        masks_bin = (up_masks > mask_thres).astype(np.uint8)
        mask_keep = mask_nms(masks_bin, scores, iou_thres=0.6)

        if len(mask_keep) == 0:
            return None

        boxes_xyxy = boxes_xyxy[mask_keep]
        scores = scores[mask_keep]
        masks_bin = masks_bin[mask_keep]
        up_masks = up_masks[mask_keep]

        # 10. Select best bed using quality scoring (lines 284-345)
        best_mask = self._select_best_bed(boxes_xyxy, scores, masks_bin, up_masks, mask_thres)
        return best_mask

    def _select_best_bed(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        masks_bin: np.ndarray,
        up_masks: np.ndarray,
        mask_thres: float
    ) -> Optional[np.ndarray]:
        """Select most plausible bed from multiple candidates.

        From export/infer_rknn_seg.py lines 284-345

        Args:
            boxes_xyxy: (N, 4) bed bounding boxes
            scores: (N,) confidence scores
            masks_bin: (N, H, W) binary masks
            up_masks: (N, H, W) float masks
            mask_thres: Mask threshold

        Returns:
            Binary mask (H, W) or None if no valid bed
        """
        num_dets = len(boxes_xyxy)

        if num_dets == 1:
            # Only one candidate, use it directly
            return masks_bin[0]

        # Multiple candidates - use quality scoring
        H, W = self.imgsz, self.imgsz
        quality_scores = []

        # Quality thresholds (from lines 290-292)
        MIN_AREA_RATIO = 0.03  # Mask占整幅图最小比例
        MAX_AREA_RATIO = 0.45  # Mask占整幅图最大比例
        MIN_COVERAGE = 0.4  # Mask在bbox内的最小覆盖率

        for box, score, m_bin in zip(boxes_xyxy, scores, masks_bin):
            x1, y1, x2, y2 = box.astype(int)
            x1 = np.clip(x1, 0, W - 1)
            y1 = np.clip(y1, 0, H - 1)
            x2 = np.clip(x2, 0, W - 1)
            y2 = np.clip(y2, 0, H - 1)

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            box_area = box_w * box_h
            img_area = float(H * W)

            # Mask在bbox内的像素数 & 覆盖率
            inside = m_bin[y1:y2, x1:x2].sum()
            coverage = inside / (box_area + 1e-6)

            # Mask整体面积比例
            total_area = m_bin.sum()
            area_ratio = total_area / img_area

            # 合理性判断
            ok = True
            if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                ok = False
            if coverage < MIN_COVERAGE:
                ok = False

            if not ok:
                quality_scores.append(-1.0)
                continue

            # 合理的候选，用 score * coverage 做质量分
            q = float(score) * coverage
            quality_scores.append(q)

        # 如果所有候选都"不合理"，返回 None
        if len(quality_scores) == 0:
            return None

        best_idx = int(np.argmax(quality_scores))
        if quality_scores[best_idx] < 0:
            return None

        return masks_bin[best_idx]

    def release(self):
        """Clean up RKNN resources."""
        if hasattr(self, 'rknn'):
            self.rknn.release()
