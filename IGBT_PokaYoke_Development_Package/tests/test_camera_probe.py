import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools import camera_probe


class FakeCapture:
    def isOpened(self):
        return False

    def release(self):
        pass


class CameraProbeTests(unittest.TestCase):
    def test_no_camera_writes_json_and_markdown_without_control_claims(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                output_dir=temp_dir, max_index=0, backends=["DSHOW"], duration_s=0.01,
                sample_interval_s=0.01, show=False,
            )
            fake_cv = SimpleNamespace(CAP_DSHOW=700, CAP_MSMF=1400)
            result = camera_probe.run_probe(
                args, cv2_module=fake_cv, capture_factory=lambda _i, _b: FakeCapture()
            )
            self.assertEqual(result["status"], "NO_CAMERA")
            report = json.loads((Path(temp_dir) / "camera_probe_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NO_CAMERA")
            md = (Path(temp_dir) / "camera_probe_report.md").read_text(encoding="utf-8")
            self.assertIn("IAMCameraControl", md)
            self.assertFalse((Path(temp_dir) / "camera_stability.csv").exists())

    def test_opencv_property_observation_never_claims_capability_or_sets(self):
        class Cap:
            def get(self, _prop):
                return 0.0

        fake_cv = SimpleNamespace(
            CAP_PROP_FOCUS=1, CAP_PROP_AUTOFOCUS=2, CAP_PROP_EXPOSURE=3,
            CAP_PROP_AUTO_EXPOSURE=4, CAP_PROP_WB_TEMPERATURE=5, CAP_PROP_GAIN=6,
        )
        controls = camera_probe._opencv_control_observation(Cap(), fake_cv)
        self.assertEqual(set(controls), set(camera_probe.CONTROL_NAMES))
        for value in controls.values():
            self.assertEqual(value["capability_query"], "NOT_AVAILABLE_IN_OPENCV")
            self.assertFalse(value["set_attempted"])
            self.assertEqual(value["status"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
