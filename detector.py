import cv2
import numpy as np
from pathlib import Path
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


def train_model(
    data: str = "arduino_car.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "arduino_car",
) -> str:
    """
    Fine-tune YOLOv8 on the ArduinoCar dataset.

    Args:
        data       : path to the dataset YAML (arduino_car.yaml)
        base_model : pretrained weights to start from (yolov8n.pt auto-downloads)
        epochs     : number of training epochs
        imgsz      : input image size (square)
        batch      : batch size (-1 = auto)
        project    : output directory
        name       : run sub-folder name

    Returns:
        Absolute path to the best checkpoint (best.pt)
    """
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 on ArduinoCar dataset")
    parser.add_argument("--data",       default="arduino_car.yaml", help="dataset YAML")
    parser.add_argument("--model",      default="yolov8n.pt",       help="base weights")
    parser.add_argument("--epochs",     default=100,  type=int)
    parser.add_argument("--imgsz",      default=640,  type=int)
    parser.add_argument("--batch",      default=16,   type=int)
    parser.add_argument("--project",    default="runs/detect")
    parser.add_argument("--name",       default="arduino_car")
    args = parser.parse_args()

    best_weights = train_model(
        data=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(f"\nTraining complete.")
    print(f"Best weights : {best_weights}")
    print(f"\nTo use in tello_yolo_detect.py:")
    print(f"  model_path     = '{best_weights}'")
    print(f"  target_classes = ['ArduinoCar']")
