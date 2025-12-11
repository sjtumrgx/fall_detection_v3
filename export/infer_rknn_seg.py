import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ==================== 配置 ====================
RKNN_PATH = 'weights/yolo11n-seg.rknn'
VIDEO_PATH = './dataset/Real/fall/video1.mp4'
SAVE_PATH = 'output_rknn_seg_bed_640.mp4'

downsample_ratio = 0.5
conf_thres = 0.35        # 稍微调高一点，少些假框
iou_thres = 0.6
mask_thres = 0.5
show_window = False
IMG_SIZE = 640

NUM_CLASSES = 80         # COCO
NUM_MASKS = 32           # proto 通道数
PROTO_H = 160
PROTO_W = 160

BED_CLASS_ID = 59        # COCO 中 bed 的类别 id，若自定义数据集请改这里

def mask_nms_bin(masks_bin: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5):
    """
    masks_bin: (N,H,W) 的二值 mask
    scores:    (N,)   置信度
    返回保留的索引列表
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

# ==================== 工具函数 ====================
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """和 Ultralytics 一致的 letterbox，实现到方形 640x640"""
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
    return im  # 只返回 letterbox 后的图


def xywh2xyxy(x):
    """(n,4)  cx,cy,w,h -> x1,y1,x2,y2"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    """box1:(N,4), box2:(M,4)"""
    inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * \
        np.maximum(inter_y2 - inter_y1, 0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def nms(boxes, scores, iou_thres=0.5):
    """纯 numpy NMS"""
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ==================== 主逻辑 ====================
def main():
    # ---- 初始化 RKNNLite ----
    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_PATH) != 0:
        raise RuntimeError('load_rknn failed')

    core_mask = getattr(RKNNLite, "NPU_CORE_0", None)
    ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
    if ret != 0:
        raise RuntimeError('init_runtime failed')

    # ---- 视频 ----
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if 0 < downsample_ratio < 1.0:
            frame = cv2.resize(frame, None,
                               fx=downsample_ratio, fy=downsample_ratio,
                               interpolation=cv2.INTER_AREA)

        img = letterbox(frame, IMG_SIZE)          # (640,640,3) BGR
        vis = img.copy()

        # ★★ 关键：输入用 NHWC，float32，不 /255（rknn 内部有 std=255） ★★
        x = img.astype(np.float32)                # (640,640,3)
        x = x[np.newaxis, ...]                    # (1,640,640,3) NHWC

        outputs = rknn.inference(inputs=[x])

        if len(outputs) != 2:
            raise RuntimeError(f'Expect 2 outputs, got {len(outputs)}')

        out0, out1 = outputs

        # 根据通道数判断哪个是 proto
        if out0.shape[1] == NUM_MASKS:
            proto_raw, pred_raw = out0, out1
        else:
            proto_raw, pred_raw = out1, out0

        # proto: (1,32,160,160) -> (32,160,160)
        proto = np.squeeze(proto_raw, axis=0)     # (32,160,160)

        # pred: (1,116,8400,1) -> (8400,116)
        pred = pred_raw
        if pred.ndim == 4 and pred.shape[0] == 1 and pred.shape[-1] == 1:
            pred = np.squeeze(pred, axis=(0, 3))  # (116,8400)
        else:
            pred = np.squeeze(pred)

        if pred.shape[0] == 116:
            pred = pred.transpose(1, 0)           # (8400,116)
        elif pred.shape[1] == 116:
            pass
        else:
            raise RuntimeError(f'Unexpected pred shape: {pred.shape}')

        # ---- 拆出 box、分类分数、mask 系数 ----
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:4 + NUM_CLASSES]       # (8400,80)
        mask_coeffs = pred[:, 4 + NUM_CLASSES:]       # (8400,32)

        # YOLO11 seg 没有 obj 分支，cls_scores 已经融合了目标置信度
        # ONNX 里一般已经过 sigmoid，这里稳妥起见再 sigmoid 一下：
        cls_scores = sigmoid(cls_scores)

        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

        print(f'Frame {frame_idx} score max/min: {scores.max():.4f}, {scores.min():.4f}')

        # 置信度筛选（先对所有类别过滤）
        mask_keep = scores > conf_thres
        boxes_xywh = boxes_xywh[mask_keep]
        scores = scores[mask_keep]
        cls_ids = cls_ids[mask_keep]
        mask_coeffs = mask_coeffs[mask_keep]

        # 只保留“床”这一类
        bed_mask = (cls_ids == BED_CLASS_ID)
        boxes_xywh = boxes_xywh[bed_mask]
        scores = scores[bed_mask]
        cls_ids = cls_ids[bed_mask]
        mask_coeffs = mask_coeffs[bed_mask]

        num_candidates = boxes_xywh.shape[0]
        print(f'Frame {frame_idx}: bed candidates(before NMS)={num_candidates}')

        if writer is None:
            writer = cv2.VideoWriter(
                SAVE_PATH, fourcc,
                fps if fps > 0 else 25,
                (IMG_SIZE, IMG_SIZE)
            )

        if num_candidates == 0:
            writer.write(vis)
            if show_window:
                cv2.imshow('seg_bed', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ---- NMS（只对床做）----
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        keep = nms(boxes_xyxy, scores, iou_thres=iou_thres)

        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        mask_coeffs = mask_coeffs[keep]

        num_dets = boxes_xyxy.shape[0]
        print(f'Frame {frame_idx}: bed_dets(after NMS)={num_dets}')

        if num_dets == 0:
            writer.write(vis)
            if show_window:
                cv2.imshow('seg_bed', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ---- 生成床的分割掩码 ----
        proto_flat = proto.reshape(NUM_MASKS, -1)           # (32,25600)
        masks = mask_coeffs @ proto_flat                    # (N,25600)
        masks = sigmoid(masks)
        masks = masks.reshape(num_dets, PROTO_H, PROTO_W)   # (N,160,160)

        # 上采样到 640x640
        up_masks = []
        for i in range(num_dets):
            m = cv2.resize(masks[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            up_masks.append(m)
        up_masks = np.stack(up_masks, axis=0)               # (N,640,640)

        # ★ 先二值化，用于 mask NMS
        masks_bin = (up_masks > mask_thres).astype(np.uint8)    # (N,640,640)

        # ★ 在 mask 上做一次 NMS，去掉高度重叠的“重复床”
        mask_keep = mask_nms_bin(masks_bin, scores, iou_thres=0.6)
        boxes_xyxy = boxes_xyxy[mask_keep]
        scores = scores[mask_keep]
        up_masks = up_masks[mask_keep]
        masks_bin = masks_bin[mask_keep]
        num_dets = boxes_xyxy.shape[0]
        print(f'Frame {frame_idx}: bed_dets(after mask-NMS)={num_dets}')

        if num_dets == 0:
            writer.write(vis)
            if show_window:
                cv2.imshow('seg_bed', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ★ 再选一个最靠谱的床：考虑“mask 填满 bbox 的程度”
        if num_dets > 1:
            # ---- 选出“最靠谱的床” ----
            H, W = IMG_SIZE, IMG_SIZE
            quality_scores = []

            # 一些可以微调的超参数（按你画面大概调一下）
            MIN_AREA_RATIO = 0.03   # mask 占整幅图最小比例，太小视为无效
            MAX_AREA_RATIO = 0.45   # mask 占整幅图最大比例，太大视为无效（半个房间都变床）
            MIN_COVERAGE   = 0.4    # mask 在 bbox 内的最小覆盖率

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

                # mask 在 bbox 内的像素数 & 覆盖率
                inside = m_bin[y1:y2, x1:x2].sum()
                coverage = inside / (box_area + 1e-6)

                # mask 整体面积比例
                total_area = m_bin.sum()
                area_ratio = total_area / img_area

                # ---------- 合理性判断 ----------
                ok = True
                if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                    ok = False
                if coverage < MIN_COVERAGE:
                    ok = False

                if not ok:
                    # 给一个很小的得分，表示“宁可不要”
                    quality_scores.append(-1.0)
                    continue

                # ---------- 合理的候选，用 score * coverage 做质量分 ----------
                q = float(score) * coverage
                quality_scores.append(q)

            # 如果所有候选都“不合理”，这一帧就认为没有床
            best_idx = int(np.argmax(quality_scores)) if len(quality_scores) > 0 else -1
            if best_idx < 0 or quality_scores[best_idx] < 0:
                # 清空，后面会走“没有床”的分支
                boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                up_masks = np.zeros((0, H, W), dtype=np.float32)
                num_dets = 0
            else:
                boxes_xyxy = boxes_xyxy[best_idx:best_idx+1]
                scores = scores[best_idx:best_idx+1]
                up_masks = up_masks[best_idx:best_idx+1]
                masks_bin = masks_bin[best_idx:best_idx+1]
                num_dets = 1

            print(f'Frame {frame_idx}: bed_dets(final)={num_dets}')


        # ---- 画床的框 + 掩码（现在最多只剩 1 个）----
        for box, score, mask in zip(boxes_xyxy, scores, up_masks):
            x1, y1, x2, y2 = box.astype(int)
            x1 = np.clip(x1, 0, IMG_SIZE - 1)
            y1 = np.clip(y1, 0, IMG_SIZE - 1)
            x2 = np.clip(x2, 0, IMG_SIZE - 1)
            y2 = np.clip(y2, 0, IMG_SIZE - 1)

            m_bin = (mask > mask_thres).astype(np.uint8)

            m_vis = np.zeros_like(m_bin, dtype=np.uint8)
            m_vis[y1:y2, x1:x2] = m_bin[y1:y2, x1:x2]

            color = (255, 0, 0)  # 蓝色
            colored = np.zeros_like(vis, dtype=np.uint8)
            colored[:, :, 0] = 255
            alpha = 0.5
            vis = np.where(m_vis[:, :, None].astype(bool),
                        (vis * (1 - alpha) + colored * alpha).astype(np.uint8),
                        vis)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                vis, f'bed {score:.2f}', (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )

        writer.write(vis)
        if show_window:
            cv2.imshow('seg_bed', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()
    rknn.release()
    print('Done, saved to', SAVE_PATH)


if __name__ == '__main__':
    main()
