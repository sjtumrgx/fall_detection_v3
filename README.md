# 摔倒检测系统 v3

基于 YOLO11 姿态估计和床铺分割的实时摔倒检测系统，支持 GPU 和 NPU 两种部署方式。

## 项目简介

本系统通过分析视频中人体姿态和床铺位置，实时检测摔倒事件。系统采用多规则融合策略，包括：
- 身体姿态角度分析
- 躯干水平化检测
- 时序平滑与追踪
- 床铺区域判定（区分"摔倒"与"躺在床上"）

## 版本说明

| 版本 | 硬件平台 | 推理框架 | 性能 | 功耗 |
|------|---------|---------|------|------|
| **GPU 版本** | NVIDIA GPU | PyTorch + Ultralytics | 15-20 FPS | 高 |
| **NPU 版本** | Rockchip RK3588 | RKNN (rknnlite 1.5.2) | 25-35 FPS | 低 |

---

## GPU 版本使用说明

### 环境要求

```bash
pip install ultralytics opencv-python numpy
```

### 运行方式

```bash
# 基本使用（默认视频 dataset/Real/fall/video1.mp4）
python fall_detection_gpu/main.py

# 指定视频路径
python fall_detection_gpu/main.py path/to/video.mp4

# 使用下采样加速（1-8，数值越大速度越快但质量越低）
python fall_detection_gpu/main.py --downsample 4 path/to/video.mp4

# 调整日志级别
python fall_detection_gpu/main.py --log-level DEBUG --log-file debug.log

# 可视化床铺检测（显示床铺掩码和中心点）
python fall_detection_gpu/main.py --visual-mask --visual-center

# 调整床铺检测参数
python fall_detection_gpu/main.py --seg-size n --bed-center-offset 10 -30
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--downsample` | 下采样倍数 (1-8) | 4 |
| `--log-level` | 日志级别 (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--log-file` | 日志文件路径 | 无 |
| `--log-every` | 每 N 帧输出一次日志 | 30 |
| `--seg-size` | 分割模型大小 (n/s/m/l/x) | n |
| `--bed-center-offset` | 床中心偏移量 (像素) | 10 -30 |
| `--visual-center` | 可视化中心点 | False |
| `--visual-mask` | 可视化床铺掩码 | False |

### 模型选择

编辑 `fall_detection_gpu/main.py` 第 109 行，修改 `model_size` 参数：
- `"n"` - nano（最快，精度较低）
- `"s"` - small
- `"m"` - medium（推荐，平衡性能和精度）
- `"l"` - large（最精确，速度较慢）

### 输出

- **输出路径**: `results/<镜像输入路径>/<视频名>_fall_detected.avi`
- **输出内容**: 带有检测框、状态标签和骨架可视化的视频

---

## NPU 版本使用说明

### 环境要求

```bash
# 仅在 RK3588 设备上运行
pip install rknnlite opencv-python numpy
```

### 模型准备

在 `weights/` 目录下放置以下 RKNN 模型：
- **必需**: `yolo11n-pose.rknn` (姿态检测)
- **必需**: `yolo11n-seg.rknn` (床铺分割)
- **可选**: `yolo11n-pose-{imgsz}.rknn` (特定尺寸模型，如 `yolo11n-pose-480.rknn`)

> 模型转换方法参考 `export/` 目录下的转换脚本

### 运行方式

```bash
# 基本使用（默认 640×640 输入）
python fall_detection_npu/main.py

# 指定视频路径
python fall_detection_npu/main.py path/to/video.mp4

# 使用不同输入尺寸（需要对应的 RKNN 模型）
python fall_detection_npu/main.py --imgsz 640 path/to/video.mp4

# 可视化床铺检测
python fall_detection_npu/main.py --visual-mask --visual-center

# 完整参数示例
python fall_detection_npu/main.py \
    --imgsz 640 \
    --log-level INFO \
    --bed-center-offset 10 -30 \
    --visual-mask \
    path/to/video.mp4
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--imgsz` | 输入尺寸 (320/480/640/1280) | 640 |
| `--log-level` | 日志级别 (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--log-file` | 日志文件路径 | 无 |
| `--log-every` | 每 N 帧输出一次日志 | 30 |
| `--bed-center-offset` | 床中心偏移量 (像素) | 10 -30 |
| `--visual-center` | 可视化中心点 | False |
| `--visual-mask` | 可视化床铺掩码 | False |

### 关键差异

相比 GPU 版本，NPU 版本有以下特点：
- ✅ **无需下采样参数** - Letterbox 预处理自动处理缩放
- ✅ **固定输出尺寸** - 视频输出为 imgsz×imgsz（不映射回原始尺寸）
- ✅ **自动命名** - 输出文件名自动添加 `_npu` 后缀
- ✅ **更高性能** - 在 RK3588 上达到 25-35 FPS
- ✅ **更低功耗** - 相比 GPU 版本功耗显著降低

### 输出

- **输出路径**: `results/<镜像输入路径>/<视频名>_fall_detected_npu.avi`
- **输出尺寸**: imgsz×imgsz (默认 640×640)
- **输出内容**: 带有检测框、状态标签和骨架可视化的视频

---

## 检测状态说明

系统会为每个检测到的人标注以下状态之一：

| 状态 | 颜色 | 说明 |
|------|------|------|
| **No Falling** | 绿色 | 正常站立或行走 |
| **May Fall** | 橙色 | 检测到疑似摔倒姿态（未满 3 秒） |
| **Falling** | 红色 | 确认摔倒（连续 3 秒疑似摔倒姿态） |
| **On Bed** | 蓝色 | 在床上（避免误报） |
| **Image Incomplete** | 灰色 | 图像不完整或关键点不足 |

### 检测逻辑

1. **基础判定**: 通过姿态角度、身体水平化等多规则评分（阈值 0.6）
2. **时序确认**: 需连续 3 秒处于"May Fall"状态才升级为"Falling"
3. **床铺过滤**: 自动识别床铺区域，过滤正常躺床行为

---

## 目录结构

```
fall_detection_v3/
├── fall_detection_gpu/       # GPU 版本（PyTorch）
│   ├── main.py               # CLI 入口
│   ├── fall_detector.py      # 主处理管道
│   ├── fall_pose_utils.py    # 评分和工具函数
│   └── seg_bed.py            # 床铺分割封装
├── fall_detection_npu/       # NPU 版本（RKNN）
│   ├── main.py               # CLI 入口
│   ├── fall_detector_npu.py  # 主处理管道（NPU 适配）
│   ├── fall_pose_utils.py    # 评分和工具函数（与 GPU 一致）
│   ├── rknn_pose_model.py    # RKNN 姿态模型封装
│   ├── rknn_seg_model.py     # RKNN 分割模型封装
│   └── nms_utils.py          # NMS 实现（纯 numpy）
├── export/                   # 模型转换工具
│   ├── test_pose_onnx.py     # ONNX 测试
│   ├── convert_to_rknn.py    # ONNX→RKNN 转换
│   ├── infer_rknn_pose.py    # RKNN 姿态推理测试
│   └── infer_rknn_seg.py     # RKNN 分割推理测试
├── dataset/                  # 输入视频（不上传）
├── weights/                  # 模型权重（.pt, .onnx, .rknn）
├── results/                  # 输出视频
└── CLAUDE.md                 # 开发者文档

```

---

## 调试与优化

### 查看详细日志

```bash
# GPU 版本
python fall_detection_gpu/main.py --log-level DEBUG --log-every 1 path/to/video.mp4

# NPU 版本
python fall_detection_npu/main.py --log-level DEBUG --log-every 1 path/to/video.mp4
```

### 调整检测灵敏度

修改以下文件中的阈值参数：
- **摔倒评分阈值**: `fall_pose_utils.py` 第 423 行（`smoothed_score >= 0.6`）
- **确认时长**: `fall_detector.py` / `fall_detector_npu.py` 第 695 行（`3 * fps`）
- **床铺重叠阈值**: `fall_detector.py` / `fall_detector_npu.py` 第 768 行（`0.6`）

### 性能优化

**GPU 版本**:
- 增大 `--downsample` 参数（1→8）
- 使用更小的模型（l→m→s→n）

**NPU 版本**:
- 使用更小的输入尺寸（640→480→320）
- 确保 RKNN 模型已正确转换并优化

---

## 常见问题

### Q: 模型文件在哪里下载？
**A**:
- GPU 版本：首次运行时自动从 Ultralytics 下载，或手动放置到 `weights/` 目录
- NPU 版本：需要自行转换 ONNX 模型为 RKNN 格式（参考 `export/convert_to_rknn.py`）

### Q: 如何提高检测准确率？
**A**:
1. 使用更大的模型（GPU: m/l, NPU: 使用更高分辨率模型）
2. 调整床铺中心偏移量 `--bed-center-offset`
3. 开启可视化检查床铺检测是否准确 `--visual-mask --visual-center`

### Q: NPU 版本输出视频分辨率较低？
**A**: 这是设计行为。NPU 版本输出固定为 imgsz×imgsz（默认 640×640），以保持与模型输入一致。如需更高分辨率，可使用 `--imgsz 1280`（需对应的 RKNN 模型）。

### Q: 误检率较高怎么办？
**A**:
1. 提高摔倒评分阈值（修改 `fall_pose_utils.py`）
2. 延长确认时长（修改第 695 行的 `3 * fps`）
3. 确保床铺检测准确（调整 `--bed-center-offset`）

---

## 技术支持

- **开发文档**: 查看 `CLAUDE.md` 了解详细架构和实现细节
- **问题反馈**: 提交 Issue 到 GitHub 仓库

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
