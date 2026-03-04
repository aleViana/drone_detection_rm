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
        self.x_offset_buffer = deque(maxlen=window_size)
        self.y_offset_buffer = deque(maxlen=window_size)

    def estimate_ps(self, box: list) -> dict | None:
        """Called by the main loop to estimate the position of the detected object.

        Parameters
        ----------
        box : [x1, y1, x2, y2]
            Pixel coordinates of the detection bounding box.

        Returns
        -------
        dict with smoothed 'distance', 'x_offset', 'y_offset', or None if the
        bounding box is degenerate.
        """
        x1, y1, x2, y2 = box
        bbox_width_px = x2 - x1

        if bbox_width_px <= 0:
            return None

        # Pinhole model: distance = (real_width * focal_length) / bbox_width_px
        distance = (self.known_width * self.focal_length) / bbox_width_px

        # Bounding-box centre
        bbox_cx = (x1 + x2) / 2
        bbox_cy = (y1 + y2) / 2

        # Lateral offsets from frame centre (positive = right / down)
        x_offset = (bbox_cx - self.cx) * distance / self.focal_length
        y_offset = (bbox_cy - self.cy) * distance / self.focal_length

        # Push raw values into smoothing buffers
        self.distance_buffer.append(distance)
        self.x_offset_buffer.append(x_offset)
        self.y_offset_buffer.append(y_offset)

        # Return rolling-average estimates
        return {
            "distance": float(np.mean(self.distance_buffer)),
            "x_offset": float(np.mean(self.x_offset_buffer)),
            "y_offset": float(np.mean(self.y_offset_buffer)),
        }