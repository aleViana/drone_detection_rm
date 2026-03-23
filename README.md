# Tello + YOLOv8 real-time object detection

Real-time object detection on the **DJI Tello** video stream using **YOLOv8** (Ultralytics) and **OpenCV**, with **djitellopy** for drone connection. Bounding box coordinates are printed in real time. The target is a ground robotic car 

## Setup

1. Connect your computer to the Tello WiFi network.

2. **Conda (recommended)** — create and activate the environment:

```bash
conda env create -f environment.yml
conda activate drone_vision
```


**Without conda** — use a venv and pip:

```bash
pip install -r requirements.txt
```

---

## Steps to test before training the model

Use this checklist to confirm the pipeline works **before** training a custom model.



```bash
python tello_yolo_detect.py
```

- You should see: `Connecting to Tello...` then `Stream ready. Detecting...`.
- A window shows the live stream; terminal prints detections in real time.
- Press **q** to stop.

If connection fails: check WiFi, ensure only one Tello is on, and that no firewall is blocking the drone.

### 3. Confirm output format

- Point the drone at objects (person, phone, bottle, etc.).
- Check that printed lines look like: `bbox=[x1,y1,x2,y2]  conf=0.850  class=person`.
- Confirm boxes on screen match the printed coordinates.

Once all steps pass, the pipeline is ready; you can then **train your own model** and switch `model_path` in `atello.py` to your trained `.pt` file.

## Run

```bash
make run
```

- A window shows the live stream with drawn bounding boxes.
- The terminal prints one line per detection.
- Press **q** in the window to quit.

## Configuration

Edit the config block in `atello.py`:

| Option | Description |
|--------|-------------|
| `model_path` | `"yolov8n.pt"` (COCO, downloads automatically) or path to your custom `.pt` |
| `target_classes` | `None` (all classes) or e.g. `["person", "car"]` to detect only those |
| `conf_threshold` | Minimum confidence (0.0–1.0) |

## Output format

Each detection is printed as:

```
bbox=[x1,y1,x2,y2]  conf=0.850  class=person
```

- **bbox**: pixel coordinates in the displayed frame (top-left `(x1,y1)`, bottom-right `(x2,y2)`).
- **conf**: detection confidence.
- **class**: class name from the model (e.g. COCO: person, car, bottle).

## Files

- **atello.py** – Main script: Tello stream + YOLOv8 + real-time bbox output.
- **detector.py** – `YOLODetector` (generic) and `ArduinoCarDetector` (legacy); used by the main script.
