#!/usr/bin/env python3
"""
main script to run YOLOv8 to detect a target object and control the Tello drone to follow it.
- estimation
- flight control
- detector
"""

import time
import cv2
import av
from djitellopy import Tello
from detector import YOLODetector
from estimation import positionEstimator
from flight_controller import FlightController

# Suppress H.264 decode warnings from PyAV/FFmpeg (benign Wi-Fi packet loss artifacts)
av.logging.set_level(av.logging.PANIC)


def main():
    #configuration for detection model and display
    model_path = 'runs/detect/carv32/weights/best.pt'  # path to custom model
    conf_threshold = 0.28
    target_classes = ["ArduinoCar3"]  #change classes name
    display_size = (960, 720)  # resize for display (Tello is 720p)

    # Set known_width to the real physical width (cm) of whatever object you're tracking:
    #   sports ball  ≈ 22 cm
    #   Arduino car  ≈ 24 cm
    estimator = positionEstimator(
        focal_length=902.55,
        known_width=24.0,
        frame_width=display_size[0],
        frame_height=display_size[1],
        window_size=1,
    )#window_size=1 means no rolling average — immediate distance estimate, lower latency

    controller = FlightController(
        target_distance=100.0,
        frame_width=display_size[0],
        frame_height=display_size[1],
    )

#load the detection model
    print("Loading YOLOv8 model...")
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
    )

    tello = Tello() #initialize Tello object
    tello.connect() #connect calls the drone and checks battery
    print(f"BATTERY LEVEL ==> {tello.get_battery()}%")
    tello.streamon() #start video stream

    frame_read = tello.get_frame_read()
    # Allow stream to start (first frame can be delayed on some systems)
    time.sleep(1.0)
    frame = None

    for _ in range(50):#get frame loop
        f = frame_read.frame#check if frame is valid
        if f is not None and f.size > 0: #if valid, break loop
            frame = f
            break
        time.sleep(0.1)#wait for 100ms
    if frame is None: # if no valid, return error and stop
        print("Error: No frame01")
        tello.streamoff()
        return

    print("Stream ready.")

    last_rc = (0, 0, 0, 0)  # last RC command sent; replayed during grace period
    no_det_frames = 0        # consecutive frames without a detection
    GRACE_FRAMES = 10        # frames to coast on last RC before hovering

    try:
        tello.takeoff() 

        #Loop to capture frames, run detection, and output results in real-time.
        while True:
            frame = frame_read.frame #get the frame
            if frame is None or frame.size == 0: #check if the frame is valid
                continue

            frame_display = cv2.resize(frame, display_size)
            # PyAV returns RGB; convert to BGR for OpenCV display and YOLO inference
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

            detections = detector.detect(frame_display)#run detection on the current frame
            detector.draw(frame_display, detections)#draw

            if detections:
                no_det_frames = 0
                d = detections[0]  # highest-confidence detection
                x1, y1, x2, y2 = d["bbox"]
                pos = estimator.estimate_ps(d["bbox"])#pos stores the position results from estimation

                if pos:
                    rc = controller.compute(d, pos)
                    last_rc = rc
                    tello.send_rc_control(*rc)

                    print(
                        f"bbox=[{x1},{y1},{x2},{y2}]  "
                        f"conf={d['confidence']:.3f}  "
                        f"class={d['class_name']}  "
                        f"dist={pos['distance']:.1f}cm  "
                        f"rc={rc}"
                    )
                    # Overlay distance and RC below the bounding box
                    cv2.putText(
                        frame_display,
                        f"d={pos['distance']:.0f}cm  rc={rc}",
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
                    )
                else:
                    no_det_frames += 1
                    if no_det_frames > GRACE_FRAMES:
                        last_rc = controller.compute_no_target()
                    tello.send_rc_control(*last_rc)
            else:
                no_det_frames += 1
                if no_det_frames > GRACE_FRAMES:
                    last_rc = controller.compute_no_target()
                tello.send_rc_control(*last_rc)

            cv2.imshow("Tello YOLOv8", frame_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
