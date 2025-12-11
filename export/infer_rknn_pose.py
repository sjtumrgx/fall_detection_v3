import cv2
import numpy as np
from rknnlite.api import RKNNLite

RKNN_PATH = 'weights/yolo11n-pose.rknn'
VIDEO_PATH = './dataset/Real/fall/video1.mp4'
SAVE_PATH = 'output_rknn_640.mp4'

downsample_ratio = 0.5
conf_thres = 0.25
iou_thres = 0.45
kpt_thres = 0.2
show_window = False
IMG_SIZE = 640


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
    return im  # 这里只返回 letterbox 后的 640x640 图


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


def main():
    # ----------------- 初始化 RKNNLite -----------------
    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_PATH) != 0:
        raise RuntimeError('load_rknn failed')

    core_mask = getattr(RKNNLite, "NPU_CORE_0", None)
    ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
    if ret != 0:
        raise RuntimeError('init_runtime failed')

    # ----------------- 打开视频 -----------------
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

        # 可选下采样（在送入 letterbox 之前）
        if 0 < downsample_ratio < 1.0:
            frame = cv2.resize(frame, None,
                               fx=downsample_ratio, fy=downsample_ratio,
                               interpolation=cv2.INTER_AREA)

        # letterbox 到 640x640（BGR）
        img = letterbox(frame, IMG_SIZE)              # (640,640,3) BGR
        vis = img.copy()                              # 直接在这个 640x640 图上画框

        # ★★ 关键点：给 RKNN 输入 NHWC，不要转成 CHW ★★
        x = img.astype(np.float32)                    # (640,640,3)
        x = x[np.newaxis, ...]                        # (1,640,640,3) NHWC

        # rknn.config 里已经设置了 mean=0, std=255，相当于内部会 /255
        outputs = rknn.inference(inputs=[x])          # 默认 data_format=None -> NHWC

        # ----------------- 解析输出 -----------------
        pred = outputs[0]  # 预期 shape: (1, 56, 8400, 1) 或 (1, 56, 8400)

        # 去掉 batch 维和尾部的 1
        if pred.ndim == 4 and pred.shape[0] == 1 and pred.shape[-1] == 1:
            pred = np.squeeze(pred, axis=(0, 3))      # -> (56, 8400)
        else:
            pred = np.squeeze(pred)                   # 兜底

        # 变成 (8400, 56)：每一行一个候选框
        if pred.shape[0] == 56:
            pred = pred.transpose(1, 0)               # (8400, 56)
        elif pred.shape[1] == 56:
            pass                                      # 已经是 (8400, 56)
        else:
            raise RuntimeError(f'Unexpected pred shape: {pred.shape}')

        boxes_xywh = pred[:, :4]
        scores = pred[:, 4]

        num_kpts = (pred.shape[1] - 5) // 3           # 对 yolo11n-pose 是 17
        kpts = pred[:, 5:].reshape(-1, num_kpts, 3)   # (8400, 17, 3)

        # 打印当前帧分数分布，看模型是不是“正常”
        print(f'Frame {frame_idx} score max/min: {scores.max():.4f}, {scores.min():.4f}')

        # 置信度筛选
        mask = scores > conf_thres
        boxes_xywh, scores, kpts = boxes_xywh[mask], scores[mask], kpts[mask]

        persons = boxes_xywh.shape[0]
        kpt_count = int((kpts[:, :, 2] > kpt_thres).sum()) if persons > 0 else 0
        print(f'Frame {frame_idx}: persons={persons}, keypoints={kpt_count}')

        if persons == 0:
            # 没人就直接写 letterbox 后的原图
            if writer is None:
                writer = cv2.VideoWriter(SAVE_PATH, fourcc,
                                         fps if fps > 0 else 25,
                                         (IMG_SIZE, IMG_SIZE))
            writer.write(vis)
            if show_window:
                cv2.imshow('pose', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ----------------- NMS + 可视化 -----------------
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        keep = nms(boxes_xyxy, scores, iou_thres=iou_thres)
        boxes_xyxy, scores, kpts = boxes_xyxy[keep], scores[keep], kpts[keep]

        if writer is None:
            writer = cv2.VideoWriter(SAVE_PATH, fourcc,
                                     fps if fps > 0 else 25,
                                     (IMG_SIZE, IMG_SIZE))

        for box, score, kp in zip(boxes_xyxy, scores, kpts):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis, f'{score:.2f}', (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

            kp_xy = kp[:, :2]
            kp_conf = kp[:, 2]
            for (kx, ky), kc in zip(kp_xy, kp_conf):
                if kc > kpt_thres:
                    cv2.circle(vis, (int(kx), int(ky)), 3, (0, 128, 255), -1)

        writer.write(vis)
        if show_window:
            cv2.imshow('pose', vis)
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
