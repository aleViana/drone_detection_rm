from collections import deque
import numpy as np

class positionEstimator:
    def __init__(self, focal_length: float = 902.55, known_width : int = 24, frame_width: int = 960, frame_height: int = 720, window_size: int = 5):
        self.focal_length = focal_length
        self.known_width = known_width
        self.cx = frame_width // 2 #center x coordinate of the frame
        self.cy = frame_height // 2 #center y coordinate of the frame

        #buffer
        self.distance_buffer = deque(maxlen=window_size)
    

    def estimate_ps(self, box: list) -> dict | None:
        #Called for each bounding box. 
        x1, y1, x2, y2 = box
        bbox_width_px = x2 - x1

        if bbox_width_px <= 0:
            return None

        # estimate distance.
        distance = (self.known_width * self.focal_length) / bbox_width_px

        # Push raw values into smoothing buffers
        self.distance_buffer.append(distance)

        # Return rolling-average estimates
        return {
            "distance": float(np.mean(self.distance_buffer))
        }