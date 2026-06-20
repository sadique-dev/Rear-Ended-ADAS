# Camera-Based Rear-End ADAS Collision Warning System

## Software Architecture Document

**Version:** 1.0  
**Status:** Draft — Awaiting Approval  
**Author:** Architecture Design  
**Date:** 2025-06-19

---

## 1. Overall System Workflow

The system ingests a forward-facing dashcam video, detects and tracks vehicles in the ego lane / field of view, estimates distance and relative speed using monocular geometry, computes Time-To-Collision (TTC), and renders risk-level overlays on each frame before writing the annotated output video.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REAR-END ADAS PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

  Input Video (.mp4)
        │
        ▼
  ┌──────────────┐
  │ Video Reader │  Decode frames @ native FPS/resolution
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ YOLO Detector│  Pre-trained Ultralytics model (COCO: car, truck, bus, …)
  └──────┬───────┘
         │  Bounding boxes + class + confidence per frame
         ▼
  ┌──────────────┐
  │   Tracker    │  ByteTrack / BoT-SORT — persistent track IDs across frames
  └──────┬───────┘
         │  Tracked detections with stable IDs
         ▼
  ┌──────────────┐
  │ Lead Vehicle │  Select primary threat vehicle (closest in ego lane)
  │   Selector   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Distance   │  Monocular depth via known vehicle width + pinhole model
  │  Estimator   │
  └──────┬───────┘
         │  distance (m), smoothed over time
         ▼
  ┌──────────────┐
  │    Speed     │  Finite-difference: Δdistance / Δtime per track
  │  Estimator   │
  └──────┬───────┘
         │  relative speed (m/s), smoothed (EMA / Kalman)
         ▼
  ┌──────────────┐
  │     TTC      │  TTC = distance / |relative_speed|  (when closing)
  │  Calculator  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Risk Classifier│  Map TTC + distance → SAFE | CAUTION | DANGER
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Overlay    │  Bounding boxes, HUD, TTC gauge, color-coded warnings
  │   Renderer   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Video Writer │  Save annotated output (.mp4)
  └──────────────┘
```

### Per-Frame Processing Loop

1. **Read** the next frame and timestamp from the input video.
2. **Detect** all vehicle-class objects using YOLO inference.
3. **Track** detections to assign persistent IDs and filter spurious boxes.
4. **Select** the lead (threat) vehicle — the tracked object closest to the camera center / bottom of frame in the driving lane.
5. **Estimate distance** from bounding-box width using the pinhole camera model.
6. **Estimate relative speed** by differentiating smoothed distance over time for the selected track.
7. **Compute TTC** when the relative speed indicates closing behavior.
8. **Classify risk** into SAFE, CAUTION, or DANGER based on configurable thresholds.
9. **Render** visual overlays (boxes, labels, HUD panel, warning banner).
10. **Write** the annotated frame to the output video.
11. Repeat until end-of-video; release resources and log summary statistics.

### Entry Points

| Entry Point | Purpose |
|---|---|
| `python -m src.main --input video.mp4 --output out.mp4` | CLI — primary interface |
| `python -m src.main --input 0` | Live webcam mode (optional stretch goal) |

---

## 2. Folder Structure

```
Rear-End-ADAS-System/
│
├── README.md                          # Project overview, setup, usage, demo GIF
├── PROJECT_PLAN.md                    # This architecture document
├── requirements.txt                   # Pinned Python dependencies
├── .gitignore                         # Ignore venv, outputs, models cache, __pycache__
├── setup.py                           # Optional: installable package metadata
│
├── config/
│   └── default.yaml                   # All tunable parameters (thresholds, camera, model)
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # CLI entry point, orchestrates pipeline
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py           # Ultralytics YOLO wrapper
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py                 # ByteTrack / BoT-SORT integration
│   │
│   ├── estimation/
│   │   ├── __init__.py
│   │   ├── distance.py                # Monocular distance from bbox width
│   │   ├── speed.py                   # Relative speed via temporal differentiation
│   │   └── ttc.py                     # Time-to-collision computation
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   └── classifier.py              # SAFE / CAUTION / DANGER logic
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── overlay.py                 # Draw boxes, HUD, warnings on frames
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── video_reader.py            # Frame decoding + timestamps
│   │   └── video_writer.py            # Annotated video encoding
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── adas_pipeline.py           # Wires all modules into processing loop
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # YAML config loader + validation
│       ├── geometry.py                # Pinhole math, lane ROI helpers
│       ├── smoothing.py               # EMA, moving average filters
│       └── logger.py                  # Structured logging setup
│
├── tests/
│   ├── __init__.py
│   ├── test_distance.py               # Unit tests for distance formula
│   ├── test_speed.py                  # Unit tests for speed estimation
│   ├── test_ttc.py                    # Unit tests for TTC edge cases
│   └── test_risk_classifier.py        # Unit tests for threshold logic
│
├── data/
│   ├── samples/                       # Small sample input videos (or .gitkeep)
│   └── outputs/                       # Processed output videos (gitignored)
│
├── models/                            # Optional local model weights (gitignored)
│   └── .gitkeep                       # YOLO weights auto-download on first run
│
└── assets/
    └── demo.gif                       # README demo animation (post-processing)
```

---

## 3. Python Modules and Responsibilities

### 3.1 `src/main.py` — CLI Entry Point

- Parse command-line arguments (`--input`, `--output`, `--config`, `--model`, `--display`).
- Load configuration from YAML.
- Instantiate and run `ADASPipeline`.
- Handle graceful shutdown and error reporting.

### 3.2 `src/io/video_reader.py` — Video Input

- Open video file or webcam via OpenCV `VideoCapture`.
- Yield `(frame, frame_index, timestamp_seconds)` tuples.
- Expose metadata: FPS, width, height, total frame count.

### 3.3 `src/io/video_writer.py` — Video Output

- Initialize `VideoWriter` with matching FPS and resolution.
- Accept annotated frames and write to disk.
- Support common codecs (H.264 via `mp4v` or `avc1`).

### 3.4 `src/detection/yolo_detector.py` — Object Detection

- Load pre-trained Ultralytics YOLO model (default: `yolov8n.pt` or `yolov8s.pt`).
- Run inference on each frame.
- Filter detections to vehicle-relevant COCO classes: `car (2)`, `motorcycle (3)`, `bus (5)`, `truck (7)`.
- Return structured detection objects: `[{bbox, confidence, class_id, class_name}]`.

### 3.5 `src/tracking/tracker.py` — Multi-Object Tracking

- Integrate Ultralytics built-in tracker (`bytetrack.yaml` or `botsort.yaml`).
- Maintain persistent track IDs across frames.
- Return tracked detections: `[{bbox, track_id, confidence, class_name}]`.
- Handle track birth, update, and deletion lifecycle.

### 3.6 `src/estimation/distance.py` — Monocular Distance Estimation

- Apply pinhole camera model:

  ```
  distance (m) = (real_vehicle_width × focal_length_pixels) / bbox_width_pixels
  ```

- Default assumed vehicle width: **1.8 m** (average passenger car).
- Focal length derived from configured horizontal FOV and frame width.
- Apply temporal smoothing (EMA) to reduce jitter.
- Clamp distance to reasonable range (e.g., 1 m – 100 m).

### 3.7 `src/estimation/speed.py` — Relative Speed Estimation

- Compute per-track distance history (deque of last N frames).
- Estimate speed: `v_rel = (d_prev − d_current) / Δt`.
  - Positive `v_rel` → closing (approaching).
  - Negative `v_rel` → receding.
- Apply EMA or 1D Kalman filter for noise reduction.
- Require minimum history length before reporting speed (warm-up period).

### 3.8 `src/estimation/ttc.py` — Time-To-Collision

- Compute: `TTC = distance / v_rel` when `v_rel > threshold` (vehicle is closing).
- Return `TTC = ∞` (or `None`) when not closing or speed is below noise floor.
- Cap displayed TTC at a maximum (e.g., 10 s) for readability.

### 3.9 `src/risk/classifier.py` — Risk Classification

- Map `(distance, TTC, relative_speed)` → risk level using configurable thresholds.

| Level | Condition (defaults) |
|---|---|
| **DANGER** | TTC < 2.0 s **OR** distance < 5 m |
| **CAUTION** | TTC < 4.0 s **OR** distance < 15 m |
| **SAFE** | All other cases |

- Hysteresis optional: require N consecutive frames at a level before switching (prevents flicker).

### 3.10 `src/visualization/overlay.py` — Visual Rendering

- Draw bounding boxes color-coded by risk level (green / yellow / red).
- Display track ID, class, confidence, distance, speed, TTC on each box.
- Render HUD panel (top-left or bottom): current risk level, lead vehicle stats.
- Flash warning banner on DANGER.
- Optional: distance bar or TTC countdown gauge.

### 3.11 `src/pipeline/adas_pipeline.py` — Pipeline Orchestrator

- Wire all modules together in the per-frame loop.
- Implement lead vehicle selection logic (closest vehicle in central ROI).
- Manage per-track state dictionaries (distance history, smoothed values).
- Collect and log run statistics (frames processed, danger frame count, avg FPS).

### 3.12 `src/utils/` — Shared Utilities

| Module | Responsibility |
|---|---|
| `config.py` | Load and validate YAML config; provide typed dataclass |
| `geometry.py` | Pinhole math, ROI polygon, bbox center/bottom computation |
| `smoothing.py` | EMA filter, moving average, optional Kalman 1D |
| `logger.py` | Configure `logging` with console + optional file handler |

---

## 4. Libraries Required

### Core Dependencies

| Library | Version (target) | Purpose |
|---|---|---|
| **ultralytics** | ≥ 8.1 | YOLO detection + built-in ByteTrack/BoT-SORT |
| **opencv-python** | ≥ 4.8 | Video I/O, drawing overlays, image operations |
| **numpy** | ≥ 1.24 | Numerical computation, array operations |
| **PyYAML** | ≥ 6.0 | Configuration file parsing |
| **torch** | ≥ 2.0 | Backend for Ultralytics (installed as ultralytics dep) |
| **torchvision** | ≥ 0.15 | Vision utilities (installed as ultralytics dep) |

### Development / Testing Dependencies

| Library | Purpose |
|---|---|
| **pytest** | Unit testing framework |
| **pytest-cov** | Test coverage reporting |
| **black** | Code formatting |
| **ruff** | Linting |
| **mypy** | Optional static type checking |

### `requirements.txt` (draft)

```
ultralytics>=8.1.0
opencv-python>=4.8.0
numpy>=1.24.0
PyYAML>=6.0
```

### Hardware Requirements

- **CPU:** Functional but slow (~5–10 FPS with YOLOv8n).
- **GPU (recommended):** NVIDIA GPU with CUDA for real-time inference (~30+ FPS with YOLOv8n).
- **RAM:** ≥ 4 GB.
- **Storage:** ~500 MB for model weights (auto-downloaded on first run).

---

## 5. Data Flow

### 5.1 Data Structures

```python
# Detection (single frame, pre-tracking)
Detection = {
    "bbox": (x1, y1, x2, y2),       # pixel coordinates
    "confidence": float,              # 0.0 – 1.0
    "class_id": int,                  # COCO class index
    "class_name": str,                # e.g., "car"
}

# Tracked Detection (post-tracking)
TrackedDetection = Detection + {
    "track_id": int,                  # persistent ID
}

# Per-Track State (maintained across frames)
TrackState = {
    "track_id": int,
    "distance_history": deque[float], # meters, last N frames
    "distance_smoothed": float,       # EMA-smoothed distance
    "speed_smoothed": float,          # m/s, relative (positive = closing)
    "ttc": float | None,              # seconds
    "risk_level": str,                # "SAFE" | "CAUTION" | "DANGER"
    "frames_tracked": int,
}

# Frame Result (output of one pipeline iteration)
FrameResult = {
    "frame_index": int,
    "timestamp": float,
    "tracked_detections": list[TrackedDetection],
    "lead_vehicle": TrackState | None,
    "annotated_frame": np.ndarray,
    "inference_ms": float,
}
```

### 5.2 Data Flow Diagram

```
Config (YAML)
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
Video File ──► VideoReader ──► Frame (BGR ndarray)
                                    │
                                    ▼
                              YOLODetector ──► List[Detection]
                                    │
                                    ▼
                              Tracker ──► List[TrackedDetection]
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                   Lead Vehicle           Other Tracks
                   Selector              (drawn but not
                          │               used for TTC)
                          ▼
                   DistanceEstimator
                   (bbox width → meters)
                          │
                          ▼
                   SpeedEstimator
                   (Δdistance / Δt)
                          │
                          ▼
                   TTCCalculator
                   (distance / speed)
                          │
                          ▼
                   RiskClassifier
                   (thresholds → level)
                          │
                          ▼
                   OverlayRenderer ──► Annotated Frame
                          │
                          ▼
                   VideoWriter ──► Output .mp4
```

### 5.3 State Management

- **Stateless modules:** Detector, TTC Calculator, Risk Classifier, Overlay, Video I/O.
- **Stateful modules:** Tracker (internal track states), Speed Estimator (per-track history deques), Pipeline (master `dict[track_id → TrackState]`).

---

## 6. Algorithms

### 6.1 Object Detection — YOLOv8 (Ultralytics)

- **Model:** Pre-trained `yolov8n.pt` (nano, fast) or `yolov8s.pt` (small, more accurate).
- **Why:** State-of-the-art real-time detection, simple API, built-in tracking support.
- **Classes used:** COCO vehicle classes filtered post-inference.
- **Confidence threshold:** 0.4 (configurable).
- **NMS IoU threshold:** 0.5 (default Ultralytics).

### 6.2 Multi-Object Tracking — ByteTrack

- **Algorithm:** ByteTrack (via Ultralytics `model.track()`).
- **Why:** Robust to occlusions, handles low-confidence detections, no extra dependencies.
- **Alternative:** BoT-SORT (available in Ultralytics, uses appearance features — slightly heavier but more robust to ID switches).
- **Default choice:** ByteTrack for simplicity and speed.

### 6.3 Monocular Distance Estimation — Pinhole Camera Model

- **Formula:**

  ```
  D = (W_real × f_px) / w_bbox
  ```

  Where:
  - `D` = distance to object (meters)
  - `W_real` = assumed real-world width of vehicle (1.8 m default)
  - `f_px` = focal length in pixels = `(image_width / 2) / tan(FOV_h / 2)`
  - `w_bbox` = bounding box width in pixels

- **Assumption:** Vehicle is roughly perpendicular to the camera (rear or front face visible).
- **Smoothing:** Exponential Moving Average (α = 0.3 default).

### 6.4 Relative Speed Estimation — Temporal Differentiation

- **Formula:**

  ```
  v_rel = (D_{t-1} - D_t) / (t_t - t_{t-1})
  ```

- **Sign convention:** Positive = closing (distance decreasing).
- **Smoothing:** EMA on speed values (α = 0.2) or 1D Kalman filter.
- **Noise floor:** Ignore speed estimates below 0.5 m/s (configurable).
- **Warm-up:** Require ≥ 5 frames of history before reporting speed.

### 6.5 Time-To-Collision (TTC)

- **Formula:**

  ```
  TTC = D / v_rel    (when v_rel > speed_threshold)
  ```

- **Constant closing speed assumption** — valid for short prediction horizons (< 5 s).
- **Edge cases:**
  - Not closing (`v_rel ≤ 0`) → TTC = None (∞ displayed).
  - Very small speed → TTC capped at max display value (10 s).
  - Distance ≤ 0 → immediate DANGER.

### 6.6 Risk Classification — Threshold-Based Rules

```
if distance < danger_distance OR ttc < danger_ttc:
    → DANGER
elif distance < caution_distance OR ttc < caution_ttc:
    → CAUTION
else:
    → SAFE
```

- **Optional hysteresis:** Level must persist for 3 consecutive frames before UI update.
- Thresholds fully configurable via `config/default.yaml`.

### 6.7 Lead Vehicle Selection

- Define a trapezoidal **Region of Interest (ROI)** in the lower-center of the frame (approximates ego lane).
- Among tracked vehicles with bottom-center inside the ROI, select the one with the **largest bounding box area** (closest to camera).
- Fallback: if no vehicle in ROI, select globally closest (largest bbox).

---

## 7. Limitations

### 7.1 Distance Estimation Accuracy

- **Single assumed vehicle width (1.8 m)** — trucks, buses, and motorcycles will produce significant distance errors.
- **No camera calibration** — focal length is estimated from FOV, not measured. Real dashcam intrinsics would improve accuracy.
- **Flat road assumption** — distance is derived from bbox width, which breaks on hills, banked curves, or when the vehicle is angled.
- **Typical error:** ±20–40% depending on vehicle type and camera setup.

### 7.2 Speed Estimation

- **Differentiation amplifies noise** — distance jitter translates to speed noise; requires heavy smoothing.
- **Lag from smoothing** — EMA/Kalman filters introduce 0.5–1 s delay in speed/TTC response.
- **Constant-speed TTC assumption** — does not account for lead vehicle braking or acceleration.

### 7.3 Detection & Tracking

- **Occlusions** — partial or full occlusion can cause track loss or ID switches.
- **Night / rain / glare** — pre-trained COCO model performance degrades in adverse conditions.
- **Non-vehicle obstacles** — system only detects COCO vehicle classes; pedestrians, debris, or animals are ignored.

### 7.4 General

- **Not production-ready** — this is a portfolio/demonstration project, not certified for real-world ADAS deployment.
- **No lane detection** — ROI-based lane approximation is crude compared to dedicated lane detection models.
- **Monocular only** — no stereo depth or LiDAR fusion.
- **Processing speed** — CPU-only inference may not achieve real-time on high-resolution video.
- **Fixed camera assumption** — camera must be stationary relative to the vehicle (mounted dashcam). Handheld video will produce erratic results.

---

## 8. Future Improvements

### Near-Term (v1.1)

| Improvement | Description |
|---|---|
| **Class-specific vehicle widths** | Use different `W_real` for car (1.8 m), truck (2.5 m), bus (2.5 m), motorcycle (0.8 m) |
| **Lane detection integration** | Add UFLD or similar lightweight lane model for accurate ego-lane ROI |
| **Kalman filter for distance/speed** | Replace EMA with a proper constant-velocity Kalman filter |
| **Webcam / RTSP live mode** | Real-time processing from live camera feed |
| **Configurable HUD themes** | Light/dark overlay themes, customizable panel layout |

### Medium-Term (v2.0)

| Improvement | Description |
|---|---|
| **Camera auto-calibration** | Estimate focal length from vanishing point / lane geometry |
| **Multi-threat ranking** | Track and display TTC for multiple vehicles simultaneously |
| **Brake detection heuristic** | Detect lead vehicle brake lights (color blob in bbox upper region) |
| **Dashboard web UI** | Flask/Streamlit app for upload, processing, and result viewing |
| **Performance profiling** | Benchmark table across YOLO model sizes and hardware |

### Long-Term (v3.0+)

| Improvement | Description |
|---|---|
| **Stereo / dual-camera depth** | True depth map from stereo pair for accurate distance |
| **Sensor fusion** | Combine visual TTC with GPS speed and IMU data |
| **Custom fine-tuned model** | Train YOLO on dashcam-specific dataset for improved adverse-weather performance |
| **Edge deployment** | Optimize for NVIDIA Jetson or mobile NPU (TensorRT / ONNX export) |
| **FCW standard alignment** | Map thresholds to NHTSA/ISO FCW test protocol parameters |

---

## Configuration Reference (Preview)

```yaml
# config/default.yaml (not yet implemented)

model:
  name: "yolov8n.pt"
  confidence: 0.4
  iou: 0.5
  classes: [2, 3, 5, 7]          # car, motorcycle, bus, truck

tracker:
  type: "bytetrack"
  persist: true

camera:
  fov_horizontal_deg: 90        # degrees
  assumed_vehicle_width_m: 1.8

estimation:
  distance_ema_alpha: 0.3
  speed_ema_alpha: 0.2
  speed_min_mps: 0.5            # noise floor
  speed_warmup_frames: 5
  ttc_max_display_s: 10.0

risk:
  danger:
    ttc_s: 2.0
    distance_m: 5.0
  caution:
    ttc_s: 4.0
    distance_m: 15.0
  hysteresis_frames: 3

visualization:
  show_all_tracks: true
  hud_position: "top-left"
  danger_flash: true

io:
  output_codec: "mp4v"
  output_fps: null                # null = match input
```

---

## Approval Checklist

Before implementation begins, please confirm or adjust:

- [ ] Folder structure and module breakdown
- [ ] YOLOv8n as default model (vs YOLOv8s for accuracy)
- [ ] ByteTrack as default tracker (vs BoT-SORT)
- [ ] Risk thresholds (DANGER: TTC < 2 s / dist < 5 m, CAUTION: TTC < 4 s / dist < 15 m)
- [ ] Monocular distance approach (known width + pinhole model)
- [ ] Assumed camera FOV (90° horizontal default)
- [ ] Any additional features for v1 scope

**No code will be written until this document is approved.**
