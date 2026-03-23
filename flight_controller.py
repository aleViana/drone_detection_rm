

import time


class PIDController:
    """Generic discrete PID controller."""

    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error: float, timestamp: float | None = None) -> float:
        now = timestamp if timestamp is not None else time.monotonic()

        if self._prev_time is None:
            dt = 0.0
        else:
            dt = now - self._prev_time
            dt = max(dt, 1e-6)  # guard against zero division

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, output))

        self._prev_error = error
        self._prev_time = now

        return output


class FlightController:
    """
    Translates position estimates from positionEstimator + bounding-box geometry
    into Tello RC commands via three PID controllers.

    Coordinate conventions (Tello send_rc_control):
        left_right       : negative = left,    positive = right
        forward_backward : negative = backward, positive = forward
        up_down          : negative = down,     positive = up
        yaw              : negative = CCW,      positive = CW

    """

    def __init__(
        self,
        # Desired standoff distance from the target (cm)
        target_distance: float = 150.0,
        # Frame dimensions — must match display_size used in tello_yolo_detect.py
        frame_width: int = 960,
        frame_height: int = 720,
        # Vertical target position as a fraction of frame height (0=top, 1=bottom).
        # Use ~0.70 for a fixed forward-facing camera: keeps the ground target in
        # the lower part of the frame so the drone doesn't descend to center it.
        target_vertical_ratio: float = 0.60,
        # PID gains  (kp, ki, kd)  — tune these for your environment
        yaw_gains: tuple[float, float, float] = (0.40, 0.002, 0.12),
        vertical_gains: tuple[float, float, float] = (0.35, 0.002, 0.10),
        distance_gains: tuple[float, float, float] = (0.50, 0.001, 0.12),
        # Per-axis output clamps (mapped to Tello's -100..100 range)
        yaw_limit: float = 65.0,
        vertical_limit: float = 50.0,
        distance_limit: float = 85.0,
    ):
        self.target_distance = target_distance
        self.cx = frame_width // 2                          # frame horizontal centre
        self.cy = int(frame_height * target_vertical_ratio) # vertical target (lower = higher drone)

        self.pid_yaw = PIDController(*yaw_gains, output_limit=yaw_limit)
        self.pid_vertical = PIDController(*vertical_gains, output_limit=vertical_limit)
        self.pid_distance = PIDController(*distance_gains, output_limit=distance_limit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all PID internal state (call when tracking is lost/reacquired)."""
        self.pid_yaw.reset()
        self.pid_vertical.reset()
        self.pid_distance.reset()

    def compute(
        self,
        detection: dict,
        position: dict,
        timestamp: float | None = None,
    ) -> tuple[int, int, int, int]:
        """
        Compute RC commands for one control tick.

        Parameters
        ----------
        detection : dict
            One entry from YOLODetector.detect(), must contain key 'bbox'
            with [x1, y1, x2, y2] pixel coordinates.
        position : dict
            Output of positionEstimator.estimate_ps(), must contain key
            'distance' in centimetres.
        timestamp : float | None
            Monotonic time in seconds. Uses time.monotonic() when None.

        Returns
        -------
        (left_right, forward_backward, up_down, yaw) : tuple[int, int, int, int]
            Integer RC values in [-100, 100].
        """
        now = timestamp if timestamp is not None else time.monotonic()

        x1, y1, x2, y2 = detection["bbox"]
        # Compute pixel error between target centre and frame centre
        bbox_cx = (x1 + x2) // 2
        bbox_cy = (y1 + y2) // 2

        # Positive error → target is to the right → rotate CW (positive yaw)
        yaw_error = float(bbox_cx - self.cx)

        # Positive error → target is below centre → fly down (negative up_down)
        # We invert so that positive error produces a positive up correction.
        vertical_error = float(self.cy - bbox_cy)

        # Positive error → too far → fly forward (positive forward_backward)
        distance_error = float(position["distance"] - self.target_distance)

        yaw_cmd = self.pid_yaw.compute(yaw_error, now)
        vertical_cmd = self.pid_vertical.compute(vertical_error, now)
        forward_cmd = self.pid_distance.compute(distance_error, now)

        return (
            0,                   # left_right  (not used — yaw handles lateral alignment)
            int(forward_cmd),    # forward_backward
            int(vertical_cmd),   # up_down
            int(yaw_cmd),        # yaw
        )

    def compute_no_target(self) -> tuple[int, int, int, int]:
        """
        RC command to issue when no target is detected (hover in place).
        Also resets PID state so there is no integral windup on reacquisition.
        """
        self.reset()
        return (0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Gain tuning helpers
    # ------------------------------------------------------------------

    def set_yaw_gains(self, kp: float, ki: float, kd: float):
        self.pid_yaw.kp, self.pid_yaw.ki, self.pid_yaw.kd = kp, ki, kd

    def set_vertical_gains(self, kp: float, ki: float, kd: float):
        self.pid_vertical.kp, self.pid_vertical.ki, self.pid_vertical.kd = kp, ki, kd

    def set_distance_gains(self, kp: float, ki: float, kd: float):
        self.pid_distance.kp, self.pid_distance.ki, self.pid_distance.kd = kp, ki, kd

    def set_target_distance(self, distance_cm: float):
        self.target_distance = distance_cm


# ------------------------------------------------------------------
# Quick sanity-check (no drone required)
# ------------------------------------------------------------------
if __name__ == "__main__":
    fc = FlightController(target_distance=150.0, frame_width=960, frame_height=720)

    fake_detection = {"bbox": [600, 280, 560, 440]}  # target right of centre, below centre
    fake_position = {"distance": 200.0}              # 50 cm farther than desired

    for i in range(5):
        rc = fc.compute(fake_detection, fake_position, timestamp=i * 0.1)
        print(f"tick {i}: left_right={rc[0]:4d}  forward={rc[1]:4d}  up_down={rc[2]:4d}  yaw={rc[3]:4d}")
