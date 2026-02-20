# Tello + YOLOv8 real-time object detection

Real-time object detection on the **DJI Tello** video stream using **YOLOv8** (Ultralytics) and **OpenCV**, with **djitellopy** for drone connection. Bounding box coordinates are printed in real time.

## Setup

1. Connect your computer to the Tello WiFi when you’re ready to test the drone.

2. **Conda (recommended)** — create and activate the environment:

```bash
conda env create -f environment.yml
conda activate tello-yolo
```

Or create the env manually and install with pip:

```bash
conda create -n tello-yolo python=3.10 -y
conda activate tello-yolo
pip install -r requirements.txt
```

**Without conda** — use a venv and pip:

```bash
pip install -r requirements.txt
```

---

## Steps to test before training the model

Use this checklist to confirm the pipeline works **before** training a custom model.

### 1. Test dependencies and YOLOv8 (no drone)

Run detection on your **webcam** to verify the environment and pretrained YOLOv8:

```bash
python test_without_drone.py
```

- A window should open with the webcam feed and bounding boxes.
- Terminal should print lines like `bbox=[x1,y1,x2,y2]  conf=0.xx  class=...`.
- Press **q** to quit.

If this fails, fix your Python/OpenCV/Ultralytics setup before using the Tello.

### 2. Test Tello connection only

- Power on the Tello.
- Connect your computer to the Tello’s WiFi (e.g. `TELLO-XXXXXX`).
- Run the full script:

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

Once all steps pass, the pipeline is ready; you can then **train your own model** and switch `model_path` in `tello_yolo_detect.py` to your trained `.pt` file.

## Run

```bash
python tello_yolo_detect.py
```

- A window shows the live stream with drawn bounding boxes.
- The terminal prints one line per detection: `bbox=[x1,y1,x2,y2]  conf=0.xx  class=name`
- Press **q** in the window to quit.

## Configuration

Edit the config block in `tello_yolo_detect.py`:

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

- **tello_yolo_detect.py** – Main script: Tello stream + YOLOv8 + real-time bbox output.
- **detector.py** – `YOLODetector` (generic) and `ArduinoCarDetector` (legacy); used by the main script.
