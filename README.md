# Camera-Based Rear-End ADAS Collision Warning System

A portfolio-grade computer vision project that detects vehicles in dashcam footage, tracks them across frames, estimates distance and relative speed using monocular vision, computes Time-To-Collision (TTC), and renders collision risk overlays on the output video.

## Features

- **Vehicle Detection** — Pre-trained Ultralytics YOLOv8 (no custom training)
- **Multi-Object Tracking** — ByteTrack for persistent vehicle IDs
- **Monocular Distance Estimation** — Pinhole camera model from bounding-box width
- **Relative Speed & TTC** — Temporal differentiation with smoothing
- **Risk Classification** — SAFE / CAUTION / DANGER warning levels
- **Visual Overlays** — Color-coded bounding boxes and HUD panel
- **Configurable** — All thresholds and parameters in YAML

## Project Structure

```
Rear-End-ADAS-System/
├── config/default.yaml      # Tunable parameters
├── src/
│   ├── main.py              # CLI entry point
│   ├── detection/           # YOLO detector
│   ├── tracking/            # ByteTrack tracker
│   ├── estimation/          # Distance, speed, TTC
│   ├── risk/                # Risk classifier
│   ├── visualization/       # HUD overlays
│   ├── io/                  # Video reader/writer
│   ├── pipeline/            # Pipeline orchestrator
│   └── utils/               # Config, logging, helpers
├── tests/                   # Unit tests
├── data/samples/            # Input videos
├── data/outputs/            # Processed output videos
└── models/                  # YOLO weights (auto-downloaded)
```

## Requirements

- Python 3.10+
- NVIDIA GPU recommended (CUDA) for real-time inference
- CPU-only supported (slower)

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Rear-End-ADAS-System.git
cd Rear-End-ADAS-System

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Optional: install in editable mode with CLI entry point
pip install -e .
```

## Usage

```bash
# Process a video with default settings
python -m src.main --input data/samples/drive.mp4

# Specify output path and custom config
python -m src.main \
    --input data/samples/drive.mp4 \
    --output data/outputs/result.mp4 \
    --config config/default.yaml

# Override model and show live preview
python -m src.main \
    --input data/samples/drive.mp4 \
    --model yolov8s.pt \
    --display
```

## Configuration

All parameters live in `config/default.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `model` | YOLO model name, confidence, IoU, vehicle class IDs |
| `tracker` | ByteTrack / BoT-SORT selection |
| `camera` | Horizontal FOV, assumed vehicle width |
| `estimation` | EMA smoothing, speed noise floor, TTC cap |
| `risk` | DANGER / CAUTION distance and TTC thresholds |
| `visualization` | HUD position, overlay options |
| `io` | Output codec and FPS |

## Development Status

| Module | Status |
|---|---|
| Project Setup | Done |
| Video Reader & Writer | Pending |
| YOLO Detection | Pending |
| ByteTrack Tracking | Pending |
| Lead Vehicle Selection | Pending |
| Distance Estimation | Pending |
| Speed Estimation | Pending |
| TTC Calculation | Pending |
| Risk Classification | Pending |
| Visualization / HUD | Pending |
| Pipeline Integration | Pending |
| Testing | Pending |

## Architecture

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full software architecture document.

## License

MIT
