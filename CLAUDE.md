# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time object detection on a **DJI Tello** drone video stream using **YOLOv8** (Ultralytics) and **OpenCV**, with autonomous target-following via PID flight control. The target is a ground robotic Arduino car.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `djitellopy`, `opencv-python`, `ultralytics`

## Running

**Main entry point** (requires physical Tello drone connected via Wi-Fi):
```bash
python atello.py
```

**Sanity-check flight controller without a drone:**
```bash
python flight_controller.py
```

**Train a new model:**
```python
from detector import train_model
best_weights = train_model(
    data="DatasetTest1/drone_vision.v3i.yolov8/data.yaml",
    epochs=100,
)
```

Press `q` to quit the live feed; the drone will land automatically.

## Architecture

The system is composed of four modules that form a pipeline:

```
Tello camera → YOLODetector → positionEstimator → FlightController → Tello RC
```

- **[detector.py](detector.py)** — `YOLODetector` wraps a YOLOv8 model. `detect()` returns a list of dicts with `bbox`, `confidence`, `class_id`, `class_name`. Also contains `train_model()` for fine-tuning on custom datasets.

- **[estimation.py](estimation.py)** — `positionEstimator` converts bounding-box pixel width to distance (cm) using the pinhole camera formula `d = (known_width × focal_length) / bbox_width_px`. Uses a rolling-average buffer (default window=5) for smoothing.

- **[flight_controller.py](flight_controller.py)** — `FlightController` runs three independent PID controllers (yaw, vertical, distance/forward). `compute()` takes a detection dict + position dict and returns a `(left_right, forward_backward, up_down, yaw)` tuple of integers in `[-100, 100]` for `tello.send_rc_control()`. Left_right is always 0 — lateral alignment is handled via yaw only.

- **[atello.py](atello.py)** — Main orchestration script. Connects to Tello, starts video stream, validates the first frame, calls `takeoff()`, then runs the detect → estimate → control loop. Uses only the highest-confidence detection per frame (`detections[0]`). When no target is visible, sends `(0,0,0,0)` and resets PID state to prevent integral windup.

## Dataset & Training

Datasets are stored under `DatasetTest1/` with versioned Roboflow exports (`v1i`, `v2i`, `v3i`). The active training config points to `v3i` (single class: `ArduinoCar3`). Trained weights land in `runs/detect/<name>/weights/best.pt`.

## Key Parameters to Tune

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `target_distance` | `FlightController.__init__` | 150 cm | Standoff distance from target |
| `known_width` | `positionEstimator.__init__` | 24 cm | Real width of target for distance estimation |
| `focal_length` | `positionEstimator.__init__` | 902.55 px | Camera intrinsic; calibrate per device |
| `yaw/vertical/distance_gains` | `FlightController.__init__` | see file | PID tuning per axis |
| `target_classes` | `atello.py main()` | `["laptop"]` | Class filter passed to `YOLODetector` |

## Notes
- Hardware: DJI Tello (720p stream, WiFi-only)
- Custom target class: `ArduinoCar` (real-world width: 24 cm)
- Camera calibration: focal length = 902.55 px (empirically calibrated)
- Model weights are not committed; train locally or download manually
- OpenCV handles all frame display, annotation, and UI