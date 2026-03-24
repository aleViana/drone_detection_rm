import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    """use the path to the trined model to start the detection process:
    - detect() returns list of detections with bounding box, confidence, class_id, and class_name
    - draw() draws the bounding boxes and labels on the frame"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        target_classes: list[str | int] | None = None,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes  # Arduino Car target 
        self.names = self.model.names  # class_id

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

        results = self.model(frame, verbose=False)[0] #results is a list of detections. 
        detections = [] 
        #loop through the detected boxes
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
        #output a list of dictionaries
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

#train the model 
def train_model(
    data: str = "DatasetTest1/drone_vision.v3i.yolov8/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "arduino_car",
) -> str:
    
    model = YOLO(base_model)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    return str(best.resolve())


