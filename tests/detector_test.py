# Unit tests for the detector module
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

# Allow running this test file directly (python tests/detector_test.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import YOLODetector


class FakeBox:
    def __init__(self, conf, cls_id, xyxy):
        self.conf = conf
        self.cls = cls_id
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    def __init__(self, _model_path):
        self.names = {0: "ArduinoCar3", 1: "person"}
        self._results_sequence = [[FakeResult([])]]
        self._call_idx = 0

    def __call__(self, frame, verbose=False):
        if frame is None:
            raise ValueError("Invalid frame: None")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Invalid frame: expected HxWx3 ndarray")
        if self._call_idx >= len(self._results_sequence):
            return [FakeResult([])]
        out = self._results_sequence[self._call_idx]
        self._call_idx += 1
        return out


class TestYOLODetectorDetect(unittest.TestCase):
    @patch("detector.YOLO", FakeYOLO)
    def test_detect_per_frame_outputs_conf_cid_and_bbox_mapping(self):
        detector = YOLODetector(
            model_path="dummy.pt",
            conf_threshold=0.50,
            target_classes=None,
        )

        # Simulate sequential inference results for 2 frames.
        # Frame 1: one valid + one below threshold.
        # Frame 2: one valid detection of cid 0.
        detector.model._results_sequence = [
            [
                FakeResult(
                    boxes=[
                        FakeBox(conf=0.50, cls_id=1.0, xyxy=[10.9, 20.1, 110.8, 220.7]),
                        FakeBox(conf=0.49, cls_id=0.0, xyxy=[1.0, 2.0, 3.0, 4.0]),
                    ]
                )
            ],
            [
                FakeResult(
                    boxes=[
                        FakeBox(conf=0.92, cls_id=0.0, xyxy=[30.4, 40.6, 130.2, 240.9]),
                    ]
                )
            ],
        ]

        frame1 = np.zeros((720, 960, 3), dtype=np.uint8)
        frame2 = np.ones((720, 960, 3), dtype=np.uint8)

        detections1 = detector.detect(frame1)
        print(f"frame1_detections={detections1}")
        self.assertEqual(len(detections1), 1)
        self.assertEqual(detections1[0]["class_id"], 1)
        self.assertEqual(detections1[0]["class_name"], "person")
        self.assertEqual(detections1[0]["bbox"], [10, 20, 110, 220])
        self.assertTrue(all(isinstance(v, int) for v in detections1[0]["bbox"]))
        self.assertAlmostEqual(detections1[0]["confidence"], 0.50, places=6)

        detections2 = detector.detect(frame2)
        print(f"frame2_detections={detections2}")
        self.assertEqual(len(detections2), 1)
        self.assertEqual(detections2[0]["class_id"], 0)
        self.assertEqual(detections2[0]["class_name"], "ArduinoCar3")
        self.assertEqual(detections2[0]["bbox"], [30, 40, 130, 240])
        self.assertTrue(all(isinstance(v, int) for v in detections2[0]["bbox"]))
        self.assertAlmostEqual(detections2[0]["confidence"], 0.92, places=6)

    @patch("detector.YOLO", FakeYOLO)
    def test_detect_invalid_frame_raises(self):
        detector = YOLODetector(
            model_path="dummy.pt",
            conf_threshold=0.50,
            target_classes=None,
        )

        with self.assertRaises(ValueError):
            detector.detect(None)
        print("invalid_frame_none -> ValueError")

        invalid_gray = np.zeros((720, 960), dtype=np.uint8)  # not HxWx3
        with self.assertRaises(ValueError):
            detector.detect(invalid_gray)
        print("invalid_frame_grayscale -> ValueError")


class TestYOLODetectorDraw(unittest.TestCase):
    @patch("detector.YOLO", FakeYOLO)
    def test_draw_changes_frame_with_detection(self):
        detector = YOLODetector(
            model_path="dummy.pt",
            conf_threshold=0.50,
            target_classes=None,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame_before = frame.copy()
        detections = [
            {
                "bbox": [20, 20, 100, 90],
                "confidence": 0.87,
                "class_id": 0,
                "class_name": "ArduinoCar3",
            }
        ]

        out = detector.draw(frame, detections)
        changed = not np.array_equal(frame_before, out)
        print(f"draw_with_detection_changed={changed}")

        self.assertEqual(out.shape, frame_before.shape)
        self.assertTrue(changed)

    @patch("detector.YOLO", FakeYOLO)
    def test_draw_empty_detections_keeps_frame(self):
        detector = YOLODetector(
            model_path="dummy.pt",
            conf_threshold=0.50,
            target_classes=None,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame_before = frame.copy()

        out = detector.draw(frame, [])
        unchanged = np.array_equal(frame_before, out)
        print(f"draw_empty_unchanged={unchanged}")

        self.assertEqual(out.shape, frame_before.shape)
        self.assertTrue(unchanged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
