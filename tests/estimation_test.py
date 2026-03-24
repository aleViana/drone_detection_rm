# Unit tests for the estimation module
# estimate the distance of the target from the camera
import unittest
import sys
from pathlib import Path

# Allow running this test file directly (python tests/estimation_test.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimation import positionEstimator


class TestPositionEstimatorDistance(unittest.TestCase):
    """Test the distance estimation from the bounding box width"""
    def test_estimate_distance_from_bbox_width(self):
        estimator = positionEstimator(
            focal_length=902.55,
            known_width=24,
            frame_width=960,
            frame_height=720,
            window_size=5,
        )

        # bbox x1 = 100, x2 = 220, y1 = 200, y2 = 300
        box = [100, 200, 220, 300]
        result = estimator.estimate_ps(box)
        
        #assertIsNotNone and assertAlmostEqual are methods of the unittest.TestCase class
        self.assertIsNotNone(result)
        expected_distance = (24 * 902.55) / 120
        print(f"result_distance={result['distance']:.6f}")
        print(f"expected_distance={expected_distance:.6f}")
        self.assertAlmostEqual(result["distance"], expected_distance, places=6)


if __name__ == "__main__":
    unittest.main()
