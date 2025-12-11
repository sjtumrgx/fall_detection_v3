# CLAUDE.md
在我没有直接说让你添加测试文件的情况下不要擅自撰写测试代码。
在我没有指明让你生成额外readme的情况下不要擅自生成readme，如果有关代码使用的信息可以放到代码最上方用注释标注。
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fall Detection v3 is a computer vision system that detects falls in video using YOLO11 pose estimation and bed segmentation. The system runs in two modes:
- **GPU mode** (`fall_detection_gpu/`): Uses Ultralytics YOLO with PyTorch for development and inference on GPU
- **NPU mode** (planned in `fall_detection_npu/`): Uses RKNN format for deployment on Rockchip RK3588 NPU hardware

## Commands

### Run Fall Detection (GPU)
```bash
# Basic usage with default video (dataset/Real/fall/video1.mp4)
python fall_detection_gpu/main.py

# Specify custom video path
python fall_detection_gpu/main.py path/to/video.mp4

# With downsampling (1-8, higher = faster but lower quality)
python fall_detection_gpu/main.py --downsample 4 path/to/video.mp4

# Choose model size (n=nano, s=small, m=medium, l=large)
# Edit fall_detection_gpu/main.py line 109: model_size="m"

# Adjust logging verbosity
python fall_detection_gpu/main.py --log-level DEBUG --log-file debug.log

# Bed detection visualization (show mask overlay and center points)
python fall_detection_gpu/main.py --visual-mask --visual-center

# Adjust bed detection parameters
python fall_detection_gpu/main.py --seg-size n --bed-center-offset 10 -30
```

### Run Fall Detection (NPU - RK3588)
```bash
# Basic usage with default video and default 640x640 model
python fall_detection_npu/main.py

# Specify custom video path
python fall_detection_npu/main.py path/to/video.mp4

# Use different input size (if you have corresponding RKNN models)
python fall_detection_npu/main.py --imgsz 640 path/to/video.mp4

# With visualization options
python fall_detection_npu/main.py --visual-mask --visual-center path/to/video.mp4

# Full options
python fall_detection_npu/main.py --imgsz 640 --log-level INFO --bed-center-offset 10 -30 path/to/video.mp4
```

### Model Export Workflow
```bash
# 1. Test ONNX export and inference
python export/test_pose_onnx.py

# 2. Convert ONNX to RKNN for NPU deployment (requires rknn-toolkit2)
python export/convert_to_rknn.py

# 3. Test RKNN inference on target device (requires rknnlite)
python export/infer_rknn_pose.py
```

## Architecture

### Core Pipeline (fall_detection_gpu/fall_detector.py:process_video)

The main processing loop implements a multi-stage fall detection pipeline:

1. **Bed Detection (first frame only)**:
   - Uses YOLO11-seg model to detect bed region via `seg_bed.py`
   - Computes bed bounding box and center point B = (bx, by)
   - B can be manually offset with `--bed-center-offset DX DY`
   - Creates a bed-centered rectangle using scale factors (default 0.8x0.6 of bed bbox)

2. **Per-frame Person Detection**:
   - YOLO11-pose detects people and extracts 17 COCO keypoints
   - Simple nearest-neighbor tracking maintains person IDs across frames
   - Person center point A = (ax, ay) computed from bbox center

3. **Fall Score Computation** (`fall_pose_utils.py:compute_fall_score`):
   - Combines multiple heuristics into a weighted score [0, 1]:
     - Bbox aspect ratio (wide vs tall)
     - Torso orientation angle (vertical → horizontal)
     - Upper body angle relative to ground
     - Head-to-ankle vertical distance
     - Sudden orientation changes between frames
   - Temporal smoothing with exponential moving average (alpha=0.7)

4. **Fall Classification Rules**:
   - **Base rule**: `smoothed_score >= 0.6` → "May Fall"
   - **Escalation rule**: Force trigger (ankle above hip above shoulder)
   - **Suppression rules** (lines 430-652):
     - Cleaning floor posture detection (acute hip/knee angles)
     - Shoulder between hip and ankle (standing/bending)
     - Wrist-ankle proximity checks
     - Tall bbox spanning image height
   - **Streak-based escalation**: 3 seconds continuous "May Fall" → "Falling"

5. **On Bed vs Falling Distinction**:
   - Two-rule check when `may_fall == True` and bed detected:
     1. Center-in-rectangle: `|ax - bx| <= half_w` and `|ay - by| <= half_h`
     2. Overlap ratio: `(person_bbox ∩ bed_bbox) / person_area >= 0.6`
   - Both rules must pass → "On Bed" (blue), otherwise → "Falling" (red)

6. **Video Output**:
   - Draws bounding boxes with status labels: "No Falling" (green), "May Fall" (orange), "Falling" (red), "On Bed" (blue)
   - Renders keypoints and skeleton connections
   - Optional bed mask overlay and center point visualization
   - Mirrors input directory structure under `results/`

### Key Modules

- **fall_detector.py**: Main video processing pipeline with all detection logic
- **fall_pose_utils.py**:
  - Fall score computation with weighted heuristics
  - TemporalSmoother class for score smoothing and track ID management
  - Drawing utilities and COCO keypoint definitions
  - Logging setup
- **seg_bed.py**: YOLO11-seg wrapper for bed detection (filters for "bed" class)
- **main.py**: CLI argument parsing and entry point

### Model Management

Models are auto-downloaded from Ultralytics GitHub releases on first run:
- Pose models: `yolo11{n,s,m,l}-pose.pt` in `weights/`
- Segmentation models: `yolo11{n,s,m,l,x}-seg.pt` in `weights/`
- ONNX/RKNN exports stored alongside PyTorch weights

### Directory Structure

```
fall_detection_v3/
├── fall_detection_gpu/       # GPU inference (PyTorch + Ultralytics)
│   ├── main.py               # CLI entry point
│   ├── fall_detector.py      # Core pipeline (866 lines)
│   ├── fall_pose_utils.py    # Scoring and utilities
│   └── seg_bed.py            # Bed segmentation wrapper
├── fall_detection_npu/       # NPU inference (RKNN + RK3588)
│   ├── main.py               # CLI entry point
│   ├── fall_detector_npu.py  # NPU-adapted pipeline
│   ├── fall_pose_utils.py    # Copied from GPU (identical logic)
│   ├── rknn_pose_model.py    # RKNN pose model wrapper
│   ├── rknn_seg_model.py     # RKNN segmentation wrapper
│   └── nms_utils.py          # Pure numpy NMS implementations
├── export/                   # Model conversion tools
│   ├── test_pose_onnx.py     # Test ONNX inference
│   ├── convert_to_rknn.py    # ONNX→RKNN converter
│   ├── infer_rknn_pose.py    # RKNN inference script
│   └── infer_rknn_seg.py     # RKNN segmentation script
├── dataset/                  # Input videos (not tracked)
├── calibration_dataset/      # RKNN quantization data
├── weights/                  # Model weights (.pt, .onnx, .rknn)
└── results/                  # Output videos (mirrors dataset structure)
```

## Important Implementation Details

### Fall Detection State Machine (per track)
- **No Falling** (green): Normal posture, score < 0.6
- **May Fall** (orange): Score ≥ 0.6 but no 3s streak yet, not suppressed by cleaning/standing rules
- **Falling** (red): 3s continuous "May Fall" state when off-bed
- **On Bed** (blue): "May Fall" posture but both bed rules satisfied

### Streak Logic (lines 688-696)
- Increments only when `may_fall == True` AND not on bed (based on early check at line 663)
- Resets to 0 otherwise
- Prevents false positives from people lying in bed normally

### Bed Detection Caveats
- Runs only on first frame (after downsampling if enabled)
- Assumes static bed throughout video
- Bed bounding box scaled to current frame size if downsampling ratio changes

### Logging Strategy
- `--log-every N`: Emits detailed per-detection debug logs every N frames
- Per-second summary logs: Shows all rule evaluations for each track (line 794-820)
- State transition logs: "FALL STARTED" / "fall cleared" with full score breakdown (line 703-711)

### Keyboard Controls (when show=True)
- **ESC**: Stop processing and exit
- **Q**: Pause/resume playback

## Dependencies

Core requirements (install with `pip install`):
- `ultralytics` - YOLO11 models
- `opencv-python` - Video I/O and visualization
- `numpy` - Array operations

For NPU export (optional, RK3588 only):
- `rknn-toolkit2` - Model conversion (x86 host)
- `rknnlite` - Inference runtime (ARM device)
- `onnxruntime` - ONNX testing

## Tuning Parameters

To adjust fall detection sensitivity:
1. **Score threshold** (line 423): Change `smoothed_score >= 0.6` to tune base sensitivity
2. **Escalation duration** (line 695): Change `3 * fps` for longer/shorter confirmation period
3. **Bed overlap threshold** (line 768): Change `0.6` to require more/less overlap for "On Bed"
4. **Bed center box scales** (CLI): `--bed-center-box-w-scale` and `--bed-center-box-h-scale`

To modify fall score weights, edit `fall_pose_utils.py:compute_fall_score` lines 247-262.

## NPU Version (RK3588) Specifics

### Key Differences from GPU Version
- **No downsampling parameter**: Letterbox preprocessing handles all scaling to imgsz×imgsz
- **Fixed output size**: Video output is always imgsz×imgsz (default 640×640), not mapped back to original
- **Model files**: Uses .rknn format instead of .pt (PyTorch)
- **Output filename**: Adds "_npu" suffix automatically (e.g., `video1_fall_detected_npu.avi`)
- **100% identical fall detection logic**: All scoring, rules, and tracking are the same as GPU version

### RKNN Model Requirements
Place RKNN models in `weights/` directory:
- **Required**: `yolo11n-pose.rknn` and `yolo11n-seg.rknn` (640×640)
- **Optional**: Size-specific models like `yolo11n-pose-480.rknn` for different input sizes
- Models are loaded with rknnlite 1.5.2 on RK3588 NPU

### NPU Performance
- **Inference speed**: ~10-15ms per frame (vs 30-50ms GPU)
- **Expected FPS**: 25-35 FPS on 1080p video (vs 15-20 FPS GPU)
- **Power consumption**: Significantly lower than GPU version
- **Model size**: 11 MB RKNN (vs 42 MB PyTorch)

### Critical Implementation Notes
1. **Letterbox function**: Must match reference implementation exactly (export/infer_rknn_pose.py:17-33)
2. **Input format**: NHWC (not NCHW), float32, NO /255 normalization (built into model)
3. **Coordinate space**: All coordinates in letterbox space (imgsz×imgsz), no scaling needed
4. **Manual NMS**: Uses pure numpy implementations (nms_utils.py)

### Troubleshooting NPU Version
- **Model not found error**: Ensure .rknn files exist in `weights/` directory
- **Shape mismatch**: Verify RKNN models were exported with correct input size
- **Slow performance**: Check NPU core allocation (default NPU_CORE_0)
- **Different results**: Small differences expected due to letterbox vs downsample preprocessing
