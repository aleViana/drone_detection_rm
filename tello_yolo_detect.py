#!/usr/bin/env python3
"""
Real-time object detection on DJI Tello video stream using YOLOv8 and OpenCV.
Uses djitellopy to connect to the drone and outputs bounding box coordinates in real time.

Usage:
  python tello_yolo_detect.py

  Options (edit in __main__):
  - model_path: YOLOv8 model (e.g. "yolov8n.pt" for COCO, or your custom .pt)
  - target_classes: Only detect these classes (e.g. ["person"]) or None for all
  - conf_threshold: Minimum confidence (0.0–1.0)
"""

import time
import cv2
from djitellopy import Tello
from detector import YOLODetector


def main():
    # --- Config (edit as needed) ---
    model_path = '/Users/alejandroviana/Desktop/drone_detection_rm/runs/detect/runs/detect/arduino_car3/weights/best.pt'# downloads on first run; or use "models/arduino_car.pt"
    conf_threshold = 0.5
    target_classes = ["ArduinoCar"]  # e.g. ["person", "car"] or None for all COCO classes
    display_size = (960, 720)  # resize for display (Tello is 720p)
    # ------------------------------

    print("Loading YOLOv8 model...")
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
    )

    print("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    tello.streamon()

    frame_read = tello.get_frame_read()
    # Allow stream to start (first frame can be delayed on some systems)
    time.sleep(1.0)
    frame = None
    for _ in range(50):
        f = frame_read.frame
        if f is not None and f.size > 0:
            frame = f
            break
        time.sleep(0.1)
    if frame is None:
        print("Could not get first frame from Tello. Check connection and try again.")
        tello.streamoff()
        return

    print("Stream ready. Detecting... (press 'q' in window to quit)")
    print("Output format: bbox=[x1,y1,x2,y2]  conf=<confidence>  class=<name>\n")

    while True:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            continue

        frame_display = cv2.resize(frame, display_size)
        detections = detector.detect(frame_display)
        detector.draw(frame_display, detections)

        # Real-time bounding box coordinate output
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            print(
                f"bbox=[{x1},{y1},{x2},{y2}]  "
                f"conf={d['confidence']:.3f}  "
                f"class={d['class_name']}"
            )

        cv2.imshow("Tello YOLOv8", frame_display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tello.streamoff()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
