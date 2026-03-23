# Drone YOLOv8 Real-Time Object Detection & PID Flight Controller

Real-time object detection on a **DJI Tello** video stream using **YOLOv8** (Ultralytics) and **OpenCV**, with autonomous target-following via PID flight control. The target is a ground robotic Arduino car.

## How It Works

The system runs a four-stage pipeline on every video frame:

```
Tello camera → YOLODetector → positionEstimator → FlightController → Tello RC commands
```

1. **Detection** — A fine-tuned YOLOv8 model identifies the Arduino car in the live video frame and returns its bounding box.
2. **Distance Estimation** — The bounding box pixel width is converted to a real-world distance (cm) using the pinhole camera formula, smoothed with a rolling average.
3. **PID Control** — Three independent PID controllers compute yaw, vertical, and forward/backward RC commands to keep the car centered in frame at a set standoff distance.
4. **RC Output** — Commands are sent to the Tello via `send_rc_control(left_right, forward_backward, up_down, yaw)`. When no target is detected, the drone hovers and PID state is reset to prevent integral windup.

## Requirements

- DJI Tello drone
- Python 3.9+
- Wi-Fi connection to the Tello

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model inference and training |
| `opencv-python` | Video capture and frame rendering |
| `djitellopy` | Tello SDK wrapper |

## Usage

**Connect to the Tello's Wi-Fi**, then run:

```bash
python atello.py
```

Press `q` to quit — the drone will land automatically.

**To sanity-check the flight controller without a drone:**

```bash
python flight_controller.py
```

## Training a Custom Model

Datasets are versioned Roboflow exports located in `DatasetTest1/`. The active dataset (`v3i`) has one class: `ArduinoCar3`.

```python
from detector import train_model

best_weights = train_model(
    data="DatasetTest1/drone_vision.v3i.yolov8/data.yaml",
    base_model="yolov8n.pt",
    epochs=100,
    imgsz=640,
)
print("Best weights saved to:", best_weights)
```

Trained weights are saved to `runs/detect/<name>/weights/best.pt`. Update the `model_path` in `atello.py` to use your custom weights.

## Configuration

Key parameters to adjust in `atello.py` and the module constructors:

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `yolov8n.pt` | Path to YOLOv8 weights (pretrained or custom) |
| `target_classes` | `["laptop"]` | Class names/IDs to track; `None` = all classes |
| `target_distance` | `150 cm` | Desired standoff distance from the target |
| `known_width` | `24 cm` | Real physical width of the target object |
| `focal_length` | `902.55 px` | Camera intrinsic — calibrate per device |

PID gains (yaw, vertical, distance) can be tuned via `FlightController` constructor arguments or the `set_*_gains()` helper methods at runtime.

## Project Structure

```
atello.py            # Main script — orchestrates drone + detection loop
detector.py          # YOLODetector class + train_model() utility
estimation.py        # positionEstimator — bounding box → distance (cm)
flight_controller.py # PIDController + FlightController → RC commands
requirements.txt
DatasetTest1/        # Versioned YOLOv8 datasets (Roboflow exports)
runs/detect/         # Training outputs and model weights
```
