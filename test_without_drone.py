#!/usr/bin/env python3
"""
Test the YOLOv8 + OpenCV pipeline without the Tello drone (e.g. webcam).
Use this to verify dependencies and detection before testing with the drone.

Usage:
  python test_without_drone.py

Press 'q' in the window to quit.
"""

import cv2
from detector import YOLODetector


def main():
    model_path = "yolov8n.pt"
    conf_threshold = 0.5
    target_classes = None  # or e.g. ["person"]
    cap_index = 0  # default webcam; change if you have multiple cameras

    print("Loading YOLOv8 model (yolov8n.pt may download on first run)...")
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
    )

    print("Opening webcam...")
    cap = cv2.VideoCapture(cap_index)
    if not cap.isOpened():
        print("Could not open webcam. Check camera index or permissions.")
        return

    print("Stream ready. Detecting... (press 'q' in window to quit)")
    print("Output format: bbox=[x1,y1,x2,y2]  conf=<confidence>  class=<name>\n")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        detections = detector.detect(frame)
        detector.draw(frame, detections)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            print(
                f"bbox=[{x1},{y1},{x2},{y2}]  "
                f"conf={d['confidence']:.3f}  "
                f"class={d['class_name']}"
            )

        cv2.imshow("YOLOv8 (no drone)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
