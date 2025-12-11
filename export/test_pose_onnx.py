import cv2
import numpy as np
import onnxruntime as ort

# 路径
onnx_path = "weights/yolo11n-pose.onnx"
video_path = "./dataset/Real/fall/video1.mp4"
save_path = "output.mp4"

# 可调参数
downsample_ratio = 0.5  # 0<r<=1，1 表示不降采样
conf_thres = 0.25
iou_thres = 0.45
kpt_thres = 0.2
show_window = True

# 基础工具
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
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
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (left, top)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_thres=0.5):
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

def box_iou(box1, box2):
    inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])
    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2 - inter + 1e-7)

def scale_coords(coords, ratio, pad):
    coords[..., 0] -= pad[0]
    coords[..., 1] -= pad[1]
    coords[..., 0] /= ratio[0]
    coords[..., 1] /= ratio[1]
    return coords

# 创建 ONNX Runtime 会话
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
_, _, in_h, in_w = session.get_inputs()[0].shape  # 应为 640, 640

# 视频读取/写出
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 等比例降采样
    if 0 < downsample_ratio < 1.0:
        frame = cv2.resize(frame, None, fx=downsample_ratio, fy=downsample_ratio, interpolation=cv2.INTER_AREA)

    img, ratio, pad = letterbox(frame, (in_h, in_w))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # (1,3,H,W)

    outputs = session.run(None, {input_name: x})
    pred = outputs[0]  # expect (1, 56, 8400) or (1, 8400, 56)

    # 统一成 (num_dets, 56)
    if pred.ndim == 3:
        if pred.shape[1] < pred.shape[2]:  # (1, 56, 8400)
            pred = pred[0].transpose(1, 0)
        else:  # (1, 8400, 56)
            pred = pred[0]
    else:
        pred = pred

    boxes_xywh = pred[:, :4]
    scores = pred[:, 4]
    num_kpts = (pred.shape[1] - 5) // 3
    kpts = pred[:, 5:].reshape(-1, num_kpts, 3)

    # 置信度过滤
    conf_mask = scores > conf_thres
    boxes_xywh = boxes_xywh[conf_mask]
    scores = scores[conf_mask]
    kpts = kpts[conf_mask]

    if boxes_xywh.shape[0] == 0:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(save_path, fourcc, fps if fps > 0 else 25, (w, h))
        writer.write(frame)
        if show_window:
            cv2.imshow("pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # NMS
    keep = nms(boxes_xyxy, scores, iou_thres=iou_thres)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    kpts = kpts[keep]

    # 初始化 writer
    if writer is None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(save_path, fourcc, fps if fps > 0 else 25, (w, h))

    vis = frame.copy()

    # 可视化：框 + 关键点
    for box, score, kp in zip(boxes_xyxy, scores, kpts):
        box = box.reshape(-1, 2)
        box = scale_coords(box, ratio, pad).reshape(-1)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        kp_xy = kp[:, :2].copy()
        kp_xy = scale_coords(kp_xy, ratio, pad)
        kp_conf = kp[:, 2]
        for (kx, ky), kc in zip(kp_xy, kp_conf):
            if kc > kpt_thres:
                cv2.circle(vis, (int(kx), int(ky)), 3, (0, 128, 255), -1)

    writer.write(vis)

    if show_window:
        cv2.imshow("pose", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if writer:
    writer.release()
if show_window:
    cv2.destroyAllWindows()
print("Done, saved to", save_path)
