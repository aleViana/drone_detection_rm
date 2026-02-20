import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """Generic YOLOv8 detector for any model; outputs bbox coordinates and optional class filter."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        target_classes: list[str | int] | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes  # e.g. ["person"] or [0]; None = all classes
        self.names = self.model.names  # class_id -> name

    def _keep_class(self, class_id: int) -> bool:
        if self.target_classes is None:
            return True
        name = self.names.get(class_id, "")
        for t in self.target_classes:
            if isinstance(t, int) and t == class_id:
                return True
            if isinstance(t, str) and t == name:
                return True
        return False

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of detections:
        [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int, 'class_name': str}]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue
            cid = int(box.cls)
            if not self._keep_class(cid):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cid,
                "class_name": self.names.get(cid, "?"),
            })
        return detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f"{d.get('class_name', d.get('class_id', ''))} {d['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        return frame


class ArduinoCarDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)          # load yolov8n custom weights
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of detections:
        [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int}]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': int(box.cls)
            })

        return detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            label = f"car {d['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
