"""End-to-end tests for StationRuntime, ViewValidEvaluator, ContactCandidateTracker, and FastAPI UI endpoints."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from igbt_pokayoke import ui_app
from igbt_pokayoke.live_camera import LiveCameraEngine
from igbt_pokayoke.logic.fsm import FSMConfig
from igbt_pokayoke.runtime import (
    SCENARIO_NAMES,
    STANDARD_ROIS,
    YELLOW_REF_IMAGE,
    StationRuntime,
    build_scenario_run,
    default_view_config,
    load_image_unicode,
)
from igbt_pokayoke.vision.view_valid import FrameObservation


class StationRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 500.0
        self.frame = load_image_unicode(YELLOW_REF_IMAGE)
        h, w = self.frame.shape[:2]
        self.view_cfg = default_view_config(width=w, height=h)
        self.runtime = StationRuntime(
            view_config=self.view_cfg,
            fsm_config=FSMConfig(
                initial_complete_ms=400,
                stable_state_ms=400,
                min_stable_frames=5,
                c_absent_duration_ms=1200,
                max_frame_gap_ms=200,
                hand_leave_stable_ms=200,
            ),
            clock=lambda: self.now,
        )
        self.frame_id = 0
        self.grips = [
            {"class_name": "hand_glove", "score": 0.96, "box": (0.18, 0.42, 0.29, 0.74)},
            {"class_name": "hand_glove", "score": 0.95, "box": (0.71, 0.42, 0.82, 0.74)},
        ]

    def _obs(self, dt: float = 0.1, label_occluded: bool = False) -> FrameObservation:
        self.now += dt
        self.frame_id += 1
        return FrameObservation(
            frame_id=self.frame_id,
            captured_monotonic_s=self.now,
            camera_device_id=self.view_cfg.camera_device_id,
            roi_config_version=self.view_cfg.roi_config_version,
            camera_healthy=True,
            tube_present=True,
            label_window_present=True,
            label_occluded=label_occluded,
            front_face_confirmed=True,
            orientation_reference_visible=True,
            orientation_evidence_source="machine_reference",
            tube_center_norm=(0.50, 0.50),
            label_center_norm=(0.50, 0.50),
        )

    def test_runtime_pipeline_c_removal_end_to_end(self) -> None:
        self.runtime.start_cycle("CYC_TEST_C", pipe_color="yellow", now_monotonic=self.now)
        for _ in range(5):
            snap = self.runtime.step(
                self.frame,
                self._obs(),
                detections=self.grips,
                label_status="complete",
                label_confidence=0.98,
                now_monotonic=self.now,
            )
        self.assertEqual(snap["fsm_state"], "READY")
        self.assertTrue(snap["view_valid"])
        self.assertGreater(snap["laplacian_variance"], 120.0)
        self.assertTrue(snap["left_grip_active"])
        self.assertTrue(snap["right_grip_active"])

        self.runtime.start_inspection(now_monotonic=self.now)

        # Central hand contact for 2 frames triggers rising-edge ContactCandidateTracker
        central = [
            *self.grips,
            {"class_name": "hand_glove", "score": 0.94, "box": (0.43, 0.42, 0.57, 0.58)},
        ]
        for _ in range(2):
            snap = self.runtime.step(
                self.frame,
                self._obs(label_occluded=True),
                detections=central,
                label_status="complete",
                label_confidence=0.75,
                now_monotonic=self.now,
            )
        self.assertTrue(snap["hand_contact_candidate"])
        self.assertFalse(snap["proves_physical_contact"])

        # Hand leaves and label stays absent for >= 1200 ms
        for _ in range(16):
            snap = self.runtime.step(
                self.frame,
                self._obs(label_occluded=False),
                detections=self.grips,
                label_status="absent",
                label_confidence=0.99,
                now_monotonic=self.now,
            )
            if snap["outcome"] == "PASS":
                break

        self.assertEqual(snap["fsm_state"], "PASS")
        self.assertEqual(snap["outcome"], "PASS")
        self.assertEqual(snap["result_reason"], "C_REMOVED")
        self.assertEqual(snap["c_timer_ms"], 1200)
        self.assertTrue(snap["view_valid"])

    def test_build_scenario_run_all_six_scenarios(self) -> None:
        expected = {
            "a_pass": ("PASS", "PASS", "A_CROSSED_OUT", "yellow"),
            "b_pass": ("PASS", "PASS", "B_COVERED", "blue"),
            "c_pass": ("PASS", "PASS", "C_REMOVED", "yellow"),
            "alarm": ("ALARM", "ALARM", "INITIAL_LABEL_ABSENT", "yellow"),
            "glare_invalid": ("VIEW_INVALID", "VIEW_INVALID", "VIEW_INVALID", "blue"),
            "reset": ("WAIT_PART", "WAITING", "AUTHORIZED_RESET", "yellow"),
        }
        for name in SCENARIO_NAMES:
            run = build_scenario_run(name)
            snap = run["snapshot"]
            exp_state, exp_outcome, exp_reason, exp_color = expected[name]
            self.assertEqual(snap["fsm_state"], exp_state, f"Scenario {name} fsm_state mismatch")
            self.assertEqual(snap["outcome"], exp_outcome, f"Scenario {name} outcome mismatch")
            self.assertEqual(snap["result_reason"], exp_reason, f"Scenario {name} reason mismatch")
            self.assertEqual(run["pipe_color"], exp_color)
            self.assertIsNotNone(snap["laplacian_variance"])
            self.assertIsNotNone(snap["brightness"])
            self.assertIsNotNone(snap["overexposed_fraction"])
            self.assertTrue(Path(run["reference_image_path"]).is_file())
            self.assertGreater(len(run["transitions"]), 0)
        c_run = build_scenario_run("c_pass")
        self.assertEqual(c_run["snapshot"]["c_timer_ms"], 1200)
        glare_run = build_scenario_run("glare_invalid")
        self.assertGreater(glare_run["snapshot"]["overexposed_fraction"], 0.08)
        self.assertIn("LABEL_ROI_OVEREXPOSED", glare_run["snapshot"]["view_valid_reasons"])


class FastAPIStationEndpointsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_data_dir = os.environ.get("IGBT_DATA_DIR")
        self._old_admin_token = os.environ.get("IGBT_ADMIN_TOKEN")
        self._old_station_token = os.environ.get("IGBT_STATION_TOKEN")
        self._old_engineer_pin = os.environ.get("IGBT_ENGINEER_PIN")
        os.environ["IGBT_DATA_DIR"] = self._temp_dir.name
        os.environ["IGBT_ADMIN_TOKEN"] = "test-admin-token"
        os.environ["IGBT_STATION_TOKEN"] = "test-station-token"
        os.environ.pop("IGBT_ENGINEER_PIN", None)
        ui_app.initialize_database()
        self.client = TestClient(ui_app.app, base_url="http://127.0.0.1:8765")

    def tearDown(self) -> None:
        self.client.close()
        if self._old_data_dir is None:
            os.environ.pop("IGBT_DATA_DIR", None)
        else:
            os.environ["IGBT_DATA_DIR"] = self._old_data_dir
        if self._old_admin_token is None:
            os.environ.pop("IGBT_ADMIN_TOKEN", None)
        else:
            os.environ["IGBT_ADMIN_TOKEN"] = self._old_admin_token
        if self._old_station_token is None:
            os.environ.pop("IGBT_STATION_TOKEN", None)
        else:
            os.environ["IGBT_STATION_TOKEN"] = self._old_station_token
        if self._old_engineer_pin is None:
            os.environ.pop("IGBT_ENGINEER_PIN", None)
        else:
            os.environ["IGBT_ENGINEER_PIN"] = self._old_engineer_pin
        self._temp_dir.cleanup()

    def test_fail_closed_pass_rejection_on_runtime_snapshot(self) -> None:
        # Attempt to push outcome="PASS" while view_valid is False -> must be rejected to VIEW_INVALID
        res = self.client.post(
            "/api/runtime/snapshot",
            headers={"X-Station-Token": "test-station-token"},
            json={
                "camera_status": "ONLINE",
                "label_status": "crossed_out",
                "view_valid": False,
                "quality_ok": True,
                "fsm_state": "PASS",
                "outcome": "PASS",
                "cycle_id": "CYC_FAIL_CLOSED_1",
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["outcome"], "VIEW_INVALID")
        self.assertEqual(body["result_reason"], "SHADOW_PASS_REJECTED_MISSING_VALID_EVIDENCE")
        self.assertEqual(body["mode"], "shadow")
        self.assertFalse(body["interlock_output_enabled"])

        # Attempt to push outcome="PASS" while quality_ok is False -> must be rejected to ALARM
        res2 = self.client.post(
            "/api/runtime/snapshot",
            headers={"X-Station-Token": "test-station-token"},
            json={
                "camera_status": "ONLINE",
                "label_status": "crossed_out",
                "view_valid": True,
                "quality_ok": False,
                "fsm_state": "PASS",
                "outcome": "PASS",
                "cycle_id": "CYC_FAIL_CLOSED_2",
            },
        )
        self.assertEqual(res2.status_code, 200)
        body2 = res2.json()
        self.assertEqual(body2["outcome"], "ALARM")
        self.assertEqual(body2["result_reason"], "SHADOW_PASS_REJECTED_MISSING_VALID_EVIDENCE")

    def test_roi_overlap_validation_on_config_endpoint(self) -> None:
        # Overlapping label_contact_roi and left_grip_zone must fail with 422
        bad_payload = {
            "camera_device_id": "CAM_STATION_01",
            "camera_width": 1408,
            "camera_height": 768,
            "camera_covers_both_ends": True,
            "rois": {
                "tube_roi": [0.16, 0.42, 0.84, 0.58],
                "label_roi": [0.43, 0.45, 0.57, 0.55],
                "label_contact_roi": [0.25, 0.38, 0.60, 0.62],
                "left_grip_zone": [0.16, 0.34, 0.31, 0.82],
                "right_grip_zone": [0.69, 0.34, 0.84, 0.82],
            },
            "thresholds": {
                "hand_glove_confidence": 0.70,
                "label_min_confidence": 0.85,
                "label_min_margin": 0.15,
                "stable_state_ms": 500,
                "min_stable_frames": 5,
                "min_laplacian_variance": 120.0,
                "min_brightness": 55.0,
                "max_brightness": 225.0,
                "max_overexposed_fraction": 0.08,
            },
        }
        bad_res = self.client.put(
            "/api/config",
            headers={"X-Admin-Token": "test-admin-token"},
            json=bad_payload,
        )
        self.assertEqual(bad_res.status_code, 422)
        self.assertIn("CONTACT_ROI_OVERLAPS_GRIP_ZONE", str(bad_res.json()))

        # Valid separated 5-zone configuration must succeed
        good_payload = {
            **bad_payload,
            "rois": {
                **bad_payload["rois"],
                "label_contact_roi": [0.40, 0.38, 0.60, 0.62],
            },
        }
        good_res = self.client.put(
            "/api/config",
            headers={"X-Admin-Token": "test-admin-token"},
            json=good_payload,
        )
        self.assertEqual(good_res.status_code, 200)
        self.assertEqual(good_res.json()["roi_config_version"], "ROI_V001")

    def test_simulate_all_six_scenarios_populates_evidence_cycles_and_events(self) -> None:
        for scenario in SCENARIO_NAMES:
            res = self.client.post(f"/api/runtime/simulate/{scenario}")
            self.assertEqual(res.status_code, 200, f"Simulate {scenario} failed: {res.text}")
            data = res.json()
            self.assertEqual(data["mode"], "shadow")
            self.assertFalse(data["interlock_output_enabled"])
            self.assertIsNotNone(data["laplacian_variance"])
            self.assertIsNotNone(data["brightness"])
            self.assertIsNotNone(data["overexposed_fraction"])
            self.assertIsNotNone(data["underexposed_fraction"])
            self.assertTrue(str(data.get("latest_evidence_url", "")).startswith("/api/evidence/"))
            self.assertIn(data.get("latest_evidence_kind"), {"initial_label", "A", "B", "C", "anomaly"})

            # Verify GET /api/state also includes latest_evidence_url and optical metrics
            state_res = self.client.get("/api/state")
            self.assertEqual(state_res.status_code, 200)
            state_body = state_res.json()
            self.assertEqual(state_body["latest_evidence_url"], data["latest_evidence_url"])
            self.assertEqual(state_body["latest_evidence_kind"], data["latest_evidence_kind"])

            # Verify evidence image file endpoint serves valid JPEG
            img_res = self.client.get(data["latest_evidence_url"])
            self.assertEqual(img_res.status_code, 200)
            self.assertEqual(img_res.headers["content-type"], "image/jpeg")
            self.assertTrue(img_res.content.startswith(b"\xff\xd8"))

        cycles = self.client.get("/api/cycles").json()
        self.assertGreaterEqual(len(cycles), 6)
        first_cycle_id = cycles[0]["cycle_id"]
        events = self.client.get(f"/api/cycles/{first_cycle_id}/events").json()
        self.assertGreater(len(events), 0)

    def test_dataset_bootstrap_and_auxiliary_roi_toggle(self) -> None:
        aux_res = self.client.post("/api/camera/auxiliary_rois", json={"show": True})
        self.assertEqual(aux_res.status_code, 200)
        self.assertTrue(aux_res.json()["show_auxiliary_rois"])
        aux_off = self.client.post("/api/camera/auxiliary_rois", json={"show": False})
        self.assertEqual(aux_off.status_code, 200)
        self.assertFalse(aux_off.json()["show_auxiliary_rois"])

        boot_res = self.client.post("/api/dataset/bootstrap")
        self.assertEqual(boot_res.status_code, 200)
        boot_data = boot_res.json()
        self.assertGreaterEqual(boot_data["total_samples"], 52)
        self.assertTrue(boot_data["ready_to_train"])
        self.assertGreaterEqual(boot_data["independent_tube_groups"], 4)

        sample_url = boot_data["recent_samples"][0]["image_url"]
        img_res = self.client.get(sample_url)
        self.assertEqual(img_res.status_code, 200)
        self.assertTrue(img_res.content.startswith(b"\xff\xd8"))

    def test_engineer_auth_verify_endpoint(self) -> None:
        status_res = self.client.get("/api/auth/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertEqual(
            status_res.json(),
            {"role_gate_enabled": True, "default_pin_active": True},
        )

        # 1. Default demo PIN "888888" returns 200 with authenticated=True and configured admin_token
        pin_res = self.client.post("/api/auth/verify", json={"passcode": "888888"})
        self.assertEqual(pin_res.status_code, 200)
        pin_body = pin_res.json()
        self.assertTrue(pin_body["authenticated"])
        self.assertEqual(pin_body["role"], "engineer")
        self.assertEqual(pin_body["admin_token"], "test-admin-token")
        self.assertEqual(pin_body["expires_in_seconds"], 1800)

        # 2. Matching IGBT_ADMIN_TOKEN returns 200
        token_res = self.client.post("/api/auth/verify", json={"passcode": "test-admin-token"})
        self.assertEqual(token_res.status_code, 200)
        token_body = token_res.json()
        self.assertTrue(token_body["authenticated"])
        self.assertEqual(token_body["role"], "engineer")
        self.assertEqual(token_body["admin_token"], "test-admin-token")
        self.assertEqual(token_body["expires_in_seconds"], 1800)

        # 3. Wrong passcode "000000" returns 401
        bad_res = self.client.post("/api/auth/verify", json={"passcode": "000000"})
        self.assertEqual(bad_res.status_code, 401)
        self.assertEqual(bad_res.json()["detail"], "工程师口令错误，请重试")

        # Non-ASCII wrong passcode also cleanly returns 401; extra fields forbidden (422)
        non_ascii_res = self.client.post("/api/auth/verify", json={"passcode": "错误口令"})
        self.assertEqual(non_ascii_res.status_code, 401)
        extra_res = self.client.post("/api/auth/verify", json={"passcode": "888888", "role": "engineer"})
        self.assertEqual(extra_res.status_code, 422)

        # 4. When IGBT_ADMIN_TOKEN is unset, verify falls back to "local-admin-token"
        # while admin_or_station_token still returns 503 on protected endpoints
        os.environ.pop("IGBT_ADMIN_TOKEN", None)
        try:
            fallback_res = self.client.post("/api/auth/verify", json={"passcode": "888888"})
            self.assertEqual(fallback_res.status_code, 200)
            self.assertEqual(fallback_res.json()["admin_token"], "local-admin-token")

            cfg_res = self.client.put(
                "/api/config",
                headers={"X-Admin-Token": "local-admin-token"},
                json={
                    "camera_device_id": "CAM_STATION_01",
                    "camera_width": 1408,
                    "camera_height": 768,
                    "camera_covers_both_ends": True,
                    "rois": {
                        "tube_roi": [0.16, 0.42, 0.84, 0.58],
                        "label_roi": [0.43, 0.45, 0.57, 0.55],
                        "label_contact_roi": [0.40, 0.38, 0.60, 0.62],
                        "left_grip_zone": [0.16, 0.34, 0.31, 0.82],
                        "right_grip_zone": [0.69, 0.34, 0.84, 0.82],
                    },
                    "thresholds": {},
                },
            )
            self.assertEqual(cfg_res.status_code, 503)
            self.assertIn("IGBT_ADMIN_TOKEN_NOT_CONFIGURED", str(cfg_res.json()))
        finally:
            os.environ["IGBT_ADMIN_TOKEN"] = "test-admin-token"

        # 5. Custom IGBT_ENGINEER_PIN overrides default PIN
        os.environ["IGBT_ENGINEER_PIN"] = "246810"
        try:
            custom_status = self.client.get("/api/auth/status")
            self.assertEqual(custom_status.status_code, 200)
            self.assertFalse(custom_status.json()["default_pin_active"])
            self.assertEqual(self.client.post("/api/auth/verify", json={"passcode": "888888"}).status_code, 401)
            self.assertEqual(self.client.post("/api/auth/verify", json={"passcode": "246810"}).status_code, 200)
        finally:
            os.environ.pop("IGBT_ENGINEER_PIN", None)

    def test_camera_mirror_toggle_endpoint(self) -> None:
        self.assertTrue(ui_app.LIVE_CAMERA.mirror_horizontal)
        self.assertTrue(ui_app.LIVE_CAMERA.get_state()["mirror_horizontal"])
        init_state = self.client.get("/api/state")
        self.assertEqual(init_state.status_code, 200)
        self.assertTrue(init_state.json()["mirror_horizontal"])

        try:
            off_res = self.client.post("/api/camera/mirror", json={"mirror": False})
            self.assertEqual(off_res.status_code, 200)
            self.assertEqual(off_res.json(), {"mirror_horizontal": False})
            self.assertFalse(ui_app.LIVE_CAMERA.mirror_horizontal)
            self.assertFalse(ui_app.LIVE_CAMERA.get_state()["mirror_horizontal"])
            self.assertFalse(self.client.get("/api/state").json()["mirror_horizontal"])

            extra_res = self.client.post("/api/camera/mirror", json={"mirror": True, "extra": "forbid"})
            self.assertEqual(extra_res.status_code, 422)

            on_res = self.client.post("/api/camera/mirror", json={"mirror": True})
            self.assertEqual(on_res.status_code, 200)
            self.assertEqual(on_res.json(), {"mirror_horizontal": True})
            self.assertTrue(ui_app.LIVE_CAMERA.mirror_horizontal)
            self.assertTrue(ui_app.LIVE_CAMERA.get_state()["mirror_horizontal"])
            self.assertTrue(self.client.get("/api/state").json()["mirror_horizontal"])
        finally:
            ui_app.LIVE_CAMERA.set_mirror_horizontal(True)

    def test_live_camera_clean_hud_and_mirror_default(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        self.assertTrue(engine.mirror_horizontal)
        self.assertTrue(engine.get_state()["mirror_horizontal"])

        frame = np.full((720, 1280, 3), (80, 90, 100), dtype=np.uint8)
        for alarm_latched in (False, True):
            hud_bytes, crop_bytes, _ = engine._draw_hud(
                frame=frame,
                rois=STANDARD_ROIS,
                tube_box=None,
                label_box=None,
                left_glove=None,
                right_glove=None,
                bare_hand=None,
                contact_active=False,
                bare_hand_violation=False,
                alarm_latched=alarm_latched,
                target_locked=False,
                view_valid=False,
                label_status="unknown",
                label_conf=None,
            )
            hud_img = cv2.imdecode(np.frombuffer(hud_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            crop_img = cv2.imdecode(np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(hud_img)
            self.assertIsNotNone(crop_img)

            # Top-left HUD area must stay clean (no hard-drawn global banner at (20, 20))
            top_left_hud = hud_img[0:80, 0:500]
            self.assertLess(float(np.max(np.std(top_left_hud, axis=(0, 1)))), 2.0)

            # Top-left 4x crop area must stay clean (no hard-drawn banner at (10, 10)),
            # while the center green crosshair remains present.
            top_left_crop = crop_img[0:50, 0:320]
            self.assertLess(float(np.max(np.std(top_left_crop, axis=(0, 1)))), 2.0)
            ch, cw = crop_img.shape[:2]
            center_bgr = crop_img[ch // 2, cw // 2]
            self.assertGreater(int(center_bgr[1]) - int(center_bgr[0]), 30)

    @staticmethod
    def _make_image_bytes(ext: str = ".jpg") -> bytes:
        img = np.full((120, 180, 3), (60, 180, 210), dtype=np.uint8)
        cv2.rectangle(img, (30, 25), (150, 95), (240, 240, 240), -1)
        ok, buf = cv2.imencode(ext, img)
        assert ok
        return bytes(buf)

    def test_storage_overview_endpoint(self) -> None:
        empty_res = self.client.get("/api/storage/overview")
        self.assertEqual(empty_res.status_code, 200)
        ov = empty_res.json()
        self.assertIn("business_db", ov)
        self.assertIn("training_dataset", ov)
        self.assertIn("model_artifacts", ov)
        self.assertEqual(ov["business_db"]["tables"]["cycles"], 0)
        self.assertEqual(ov["business_db"]["tables"]["events"], 0)
        self.assertEqual(ov["business_db"]["tables"]["evidence"], 0)
        self.assertGreaterEqual(ov["business_db"]["tables"]["configuration_history"], 1)
        self.assertEqual(ov["business_db"]["tables"]["model_registry"], 0)
        self.assertEqual(ov["training_dataset"]["total_samples"], 0)
        self.assertEqual(ov["training_dataset"]["coco_annotations"], 0)

        # Populate business DB, uploaded training sample (COCO), and a run artifact
        self.assertEqual(self.client.post("/api/runtime/simulate/a_pass").status_code, 200)
        jpg = self._make_image_bytes(".jpg")
        up_res = self.client.post(
            "/api/dataset/upload",
            content=jpg,
            headers={
                "Content-Type": "image/jpeg",
                "X-Sample-Label": "complete",
                "X-Pipe-Color": "yellow",
                "X-Tube-Id": "TUBE_OV_01",
                "X-Batch-Id": "BATCH_OV_01",
                "X-Admin-Token": "test-admin-token",
            },
        )
        self.assertEqual(up_res.status_code, 200)

        run_dir = Path(self._temp_dir.name) / "runs" / "label_classifier" / "mnetv3_ov_test"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "label_classifier.onnx").write_bytes(b"0" * 4096)

        filled = self.client.get("/api/storage/overview").json()
        self.assertEqual(filled["business_db"]["tables"]["cycles"], 1)
        self.assertGreater(filled["business_db"]["tables"]["events"], 0)
        self.assertEqual(filled["business_db"]["tables"]["evidence"], 1)
        self.assertEqual(filled["business_db"]["evidence_files"], 1)
        self.assertGreater(filled["business_db"]["db_size_kb"], 0)
        self.assertEqual(filled["training_dataset"]["total_samples"], 1)
        self.assertEqual(filled["training_dataset"]["class_counts"]["complete"], 1)
        self.assertEqual(filled["training_dataset"]["coco_annotations"], 1)
        self.assertEqual(filled["model_artifacts"]["run_count"], 1)
        self.assertTrue(bool(filled["model_artifacts"]["active_model_version"]))

    def test_cycle_and_evidence_delete_and_purge_crud(self) -> None:
        # Non-ASCII secret check must return 401 (never 500 TypeError)
        with self.assertRaises(ui_app.HTTPException) as ctx:
            ui_app.admin_or_station_token("错误令牌", "IGBT_ADMIN_TOKEN")
        self.assertEqual(ctx.exception.status_code, 401)

        # Create simulated cycles (CYC_*) plus a non-simulated LIVE_001 cycle
        sim_a = self.client.post("/api/runtime/simulate/a_pass")
        sim_b = self.client.post("/api/runtime/simulate/b_pass")
        self.assertEqual(sim_a.status_code, 200)
        self.assertEqual(sim_b.status_code, 200)
        cyc_a_id = sim_a.json()["cycle_id"]
        cyc_b_id = sim_b.json()["cycle_id"]
        self.assertTrue(cyc_a_id.startswith("CYC_"))
        self.assertTrue(cyc_b_id.startswith("CYC_"))
        jpg = self._make_image_bytes(".jpg")
        ev_upload = self.client.post(
            "/api/cycles/LIVE_001/evidence/initial_label",
            content=jpg,
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(len(jpg)),
                "X-Frame-Id": "10",
                "X-Pipe-Color": "yellow",
                "X-Operation-Event": "LIVE_TEST",
                "X-Operator-Anon-Id": "OP_01",
                "X-Camera-Metadata": '{"camera":"c920"}',
                "X-Label-Status": "complete",
                "X-Captured-At-Utc": "2026-11-22T12:00:00.000Z",
            },
        )
        self.assertEqual(ev_upload.status_code, 200)
        live_ev_id = ev_upload.json()["evidence_id"]
        live_ev_file = Path(self._temp_dir.name) / "evidence" / "LIVE_001" / f"{live_ev_id}.jpg"
        self.assertTrue(live_ev_file.is_file())

        # Auth checks
        self.assertEqual(self.client.delete(f"/api/evidence/{live_ev_id}").status_code, 401)
        self.assertEqual(self.client.delete(f"/api/cycles/{cyc_a_id}").status_code, 401)
        self.assertEqual(
            self.client.post("/api/cycles/purge", json={"scope": "simulated_only", "confirm": True}).status_code,
            401,
        )

        # Delete single evidence record and verify disk cleanup
        del_ev = self.client.delete(
            f"/api/evidence/{live_ev_id}",
            headers={"X-Admin-Token": "test-admin-token"},
        )
        self.assertEqual(del_ev.status_code, 200)
        self.assertEqual(del_ev.json()["deleted_files"], 1)
        self.assertFalse(live_ev_file.exists())
        self.assertEqual(
            self.client.delete(f"/api/evidence/{live_ev_id}", headers={"X-Admin-Token": "test-admin-token"}).status_code,
            404,
        )

        # Cascade delete simulated cycle A
        del_cyc = self.client.delete(
            f"/api/cycles/{cyc_a_id}",
            headers={"X-Admin-Token": "test-admin-token"},
        )
        self.assertEqual(del_cyc.status_code, 200)
        del_cyc_body = del_cyc.json()
        self.assertEqual(del_cyc_body["deleted_cycles"], 1)
        self.assertGreater(del_cyc_body["deleted_events"], 0)
        self.assertEqual(del_cyc_body["deleted_evidence"], 1)
        self.assertEqual(del_cyc_body["deleted_files"], 1)
        self.assertEqual(del_cyc_body["freed_files"], 1)
        self.assertEqual(self.client.get(f"/api/cycles/{cyc_a_id}/events").json(), [])
        self.assertEqual(
            self.client.delete(f"/api/cycles/{cyc_a_id}", headers={"X-Admin-Token": "test-admin-token"}).status_code,
            404,
        )

        # Purge validation rejects confirm=False and extra fields
        self.assertEqual(
            self.client.post(
                "/api/cycles/purge",
                headers={"X-Admin-Token": "test-admin-token"},
                json={"scope": "simulated_only", "confirm": False},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/cycles/purge",
                headers={"X-Admin-Token": "test-admin-token"},
                json={"scope": "simulated_only", "confirm": True, "extra": 1},
            ).status_code,
            422,
        )

        # Purge simulated_only removes CYC_B but preserves LIVE_001
        purge_sim = self.client.post(
            "/api/cycles/purge",
            headers={"X-Admin-Token": "test-admin-token"},
            json={"scope": "simulated_only", "confirm": True},
        )
        self.assertEqual(purge_sim.status_code, 200)
        self.assertEqual(purge_sim.json()["deleted_cycles"], 1)
        self.assertFalse(purge_sim.json()["vacuumed"])
        remaining_ids = [c["cycle_id"] for c in self.client.get("/api/cycles").json()]
        self.assertEqual(remaining_ids, ["LIVE_001"])

        # Purge all wipes LIVE_001 and runs SQLite VACUUM
        purge_all = self.client.post(
            "/api/cycles/purge",
            headers={"X-Admin-Token": "test-admin-token"},
            json={"scope": "all", "confirm": True},
        )
        self.assertEqual(purge_all.status_code, 200)
        self.assertTrue(purge_all.json()["vacuumed"])
        self.assertEqual(purge_all.json()["deleted_cycles"], 1)
        ov_after = self.client.get("/api/storage/overview").json()
        self.assertEqual(ov_after["business_db"]["tables"]["cycles"], 0)
        self.assertEqual(ov_after["business_db"]["tables"]["events"], 0)
        self.assertEqual(ov_after["business_db"]["tables"]["evidence"], 0)
        self.assertEqual(ov_after["business_db"]["evidence_files"], 0)

    def test_model_delete_active_protection_and_path_traversal_guard(self) -> None:
        with ui_app.LIVE_CAMERA._lock:
            prev_active_version = ui_app.LIVE_CAMERA._onnx_model_version
        try:
            # Path traversal in model_version is rejected at registration
            for bad_ver in ("../outside", "v1..2"):
                bad_reg = self.client.post(
                    "/api/models",
                    headers={"X-Admin-Token": "test-admin-token"},
                    json={
                        "model_name": "evil",
                        "model_version": bad_ver,
                        "sha256": "a" * 64,
                        "license_status": "unknown",
                        "notes": "",
                    },
                )
                self.assertEqual(bad_reg.status_code, 422)

            # Register two models with disk artifact folders
            r1 = self.client.post(
                "/api/models",
                headers={"X-Admin-Token": "test-admin-token"},
                json={
                    "model_name": "mobilenetv3-small-igbt-roi",
                    "model_version": "mnetv3_active_01",
                    "sha256": "b" * 64,
                    "license_status": "reviewed",
                    "notes": "active model",
                },
            )
            r2 = self.client.post(
                "/api/models",
                headers={"X-Admin-Token": "test-admin-token"},
                json={
                    "model_name": "mobilenetv3-small-igbt-roi",
                    "model_version": "mnetv3_old_02",
                    "sha256": "c" * 64,
                    "license_status": "reviewed",
                    "notes": "old model",
                },
            )
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            active_id = r1.json()["model_id"]
            old_id = r2.json()["model_id"]

            runs_root = Path(self._temp_dir.name) / "runs" / "label_classifier"
            active_dir = runs_root / "mnetv3_active_01"
            old_dir = runs_root / "mnetv3_old_02"
            active_dir.mkdir(parents=True, exist_ok=True)
            old_dir.mkdir(parents=True, exist_ok=True)
            (active_dir / "metrics.json").write_text("{}", encoding="utf-8")
            (old_dir / "metrics.json").write_text("{}", encoding="utf-8")

            # Activate model 1
            act = self.client.post(f"/api/models/{active_id}/activate")
            self.assertEqual(act.status_code, 200)

            # Deleting active model must return 409 Conflict and preserve folder
            del_active = self.client.delete(
                f"/api/models/{active_id}",
                headers={"X-Admin-Token": "test-admin-token"},
            )
            self.assertEqual(del_active.status_code, 409)
            self.assertEqual(del_active.json()["detail"], "不能删除当前正在实时运行的模型，请先切换其他模型")
            self.assertTrue(active_dir.is_dir())

            # Deleting inactive model requires token
            self.assertEqual(self.client.delete(f"/api/models/{old_id}").status_code, 401)

            # Deleting inactive model succeeds and cleans disk artifact directory
            del_old = self.client.delete(
                f"/api/models/{old_id}",
                headers={"X-Admin-Token": "test-admin-token"},
            )
            self.assertEqual(del_old.status_code, 200)
            self.assertTrue(del_old.json()["deleted"])
            self.assertTrue(del_old.json()["deleted_artifact_dir"])
            self.assertFalse(old_dir.exists())
            remaining_models = [m["model_id"] for m in self.client.get("/api/models").json()]
            self.assertEqual(remaining_models, [active_id])
        finally:
            with ui_app.LIVE_CAMERA._lock:
                ui_app.LIVE_CAMERA._onnx_model_version = prev_active_version
                ui_app.LIVE_CAMERA._state["model_version"] = prev_active_version

    def test_dataset_upload_list_update_batch_delete_and_clear_auth(self) -> None:
        jpg = self._make_image_bytes(".jpg")
        png = self._make_image_bytes(".png")

        # Auth enforced on upload when IGBT_ADMIN_TOKEN is set
        self.assertEqual(
            self.client.post(
                "/api/dataset/upload",
                content=jpg,
                headers={"Content-Type": "image/jpeg", "X-Sample-Label": "complete"},
            ).status_code,
            401,
        )

        # Reject invalid media type and corrupted bytes
        self.assertEqual(
            self.client.post(
                "/api/dataset/upload",
                content=b"plain text",
                headers={
                    "Content-Type": "text/plain",
                    "X-Sample-Label": "complete",
                    "X-Admin-Token": "test-admin-token",
                },
            ).status_code,
            415,
        )
        self.assertEqual(
            self.client.post(
                "/api/dataset/upload",
                content=b"\xff\xd8corrupted_not_real_jpeg",
                headers={
                    "Content-Type": "image/jpeg",
                    "X-Sample-Label": "complete",
                    "X-Admin-Token": "test-admin-token",
                },
            ).status_code,
            422,
        )

        # Upload valid JPEG and PNG samples
        u1 = self.client.post(
            "/api/dataset/upload",
            content=jpg,
            headers={
                "Content-Type": "image/jpeg",
                "X-Sample-Label": "complete",
                "X-Pipe-Color": "yellow",
                "X-Tube-Id": "TUBE_CUSTOM_01",
                "X-Batch-Id": "BATCH_CUSTOM_01",
                "X-Admin-Token": "test-admin-token",
            },
        )
        u2 = self.client.post(
            "/api/dataset/upload",
            content=png,
            headers={
                "Content-Type": "image/png",
                "X-Sample-Label": "crossed_out",
                "X-Pipe-Color": "blue",
                "X-Tube-Id": "TUBE_CUSTOM_02",
                "X-Batch-Id": "BATCH_CUSTOM_02",
                "X-Admin-Token": "test-admin-token",
            },
        )
        u3 = self.client.post(
            "/api/dataset/upload",
            content=jpg,
            headers={
                "Content-Type": "image/jpeg",
                "X-Sample-Label": "crossed_out",
                "X-Pipe-Color": "blue",
                "X-Admin-Token": "test-admin-token",
            },
        )
        self.assertEqual(u1.status_code, 200)
        self.assertEqual(u2.status_code, 200)
        self.assertEqual(u3.status_code, 200)
        s1_id = u1.json()["sample_id"]
        s2_id = u2.json()["sample_id"]

        # List with filters and query
        filtered = self.client.get("/api/dataset/samples?label=complete&q=TUBE_CUSTOM_01&limit=50")
        self.assertEqual(filtered.status_code, 200)
        items = filtered.json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["sample_id"], s1_id)
        self.assertEqual(items[0]["box_xyxy"], "0.0000,0.0000,1.0000,1.0000")
        self.assertEqual(items[0]["image_url"], f"/api/dataset/sample/{s1_id}/image")
        self.assertEqual(self.client.get(items[0]["image_url"]).status_code, 200)

        # Re-annotate s1 and ensure box_xyxy is preserved after manifest rewrite
        patch_res = self.client.patch(
            f"/api/dataset/sample/{s1_id}",
            json={"label": "covered", "physical_tube_id": "TUBE_PATCHED"},
        )
        self.assertEqual(patch_res.status_code, 200)
        patched_items = self.client.get("/api/dataset/samples?label=covered").json()
        self.assertEqual(len(patched_items), 1)
        self.assertEqual(patched_items[0]["box_xyxy"], "0.0000,0.0000,1.0000,1.0000")
        self.assertEqual(patched_items[0]["physical_tube_id"], "TUBE_PATCHED")

        # Batch delete auth & validation
        self.assertEqual(
            self.client.post("/api/dataset/batch_delete", json={"sample_ids": [s2_id]}).status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                "/api/dataset/batch_delete",
                headers={"X-Admin-Token": "test-admin-token"},
                json={},
            ).status_code,
            422,
        )

        # Batch delete by sample_ids
        bd1 = self.client.post(
            "/api/dataset/batch_delete",
            headers={"X-Admin-Token": "test-admin-token"},
            json={"sample_ids": [s2_id]},
        )
        self.assertEqual(bd1.status_code, 200)
        self.assertEqual(bd1.json()["deleted_count"], 1)
        self.assertEqual(bd1.json()["deleted_files"], 1)
        self.assertEqual(bd1.json()["remaining_samples"], 2)

        # Batch delete by label class
        bd2 = self.client.post(
            "/api/dataset/batch_delete",
            headers={"X-Admin-Token": "test-admin-token"},
            json={"label": "crossed_out"},
        )
        self.assertEqual(bd2.status_code, 200)
        self.assertEqual(bd2.json()["deleted_count"], 1)
        self.assertEqual(bd2.json()["remaining_samples"], 1)

        # Clear enforces X-Admin-Token when IGBT_ADMIN_TOKEN is configured
        self.assertEqual(self.client.post("/api/dataset/clear").status_code, 401)
        clr_ok = self.client.post(
            "/api/dataset/clear",
            headers={"X-Admin-Token": "test-admin-token"},
        )
        self.assertEqual(clr_ok.status_code, 200)
        self.assertEqual(clr_ok.json()["total_samples"], 0)

        # When IGBT_ADMIN_TOKEN is unset, clear succeeds without token
        os.environ.pop("IGBT_ADMIN_TOKEN", None)
        try:
            self.assertEqual(self.client.post("/api/dataset/clear").status_code, 200)
        finally:
            os.environ["IGBT_ADMIN_TOKEN"] = "test-admin-token"


if __name__ == "__main__":
    unittest.main()
