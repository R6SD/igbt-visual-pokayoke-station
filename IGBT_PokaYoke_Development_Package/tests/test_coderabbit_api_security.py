"""Comprehensive regression tests for CodeRabbit Wave 1 API & Security fixes (BE-30..BE-45, BE-54..BE-57)."""
from __future__ import annotations

import asyncio
import collections
import contextlib
import hashlib
import inspect
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from igbt_pokayoke import ui_app
from igbt_pokayoke.runtime import BLUE_REF_IMAGE, STANDARD_ROIS
from igbt_pokayoke.vision.view_valid import measure_label_quality


def _make_jpeg_bytes(color_bgr: tuple[int, int, int] = (60, 180, 210), width: int = 180, height: int = 120) -> bytes:
    img = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return bytes(buf)


def _make_png_bytes(color_bgr: tuple[int, int, int] = (40, 160, 90), width: int = 180, height: int = 120) -> bytes:
    img = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


class CodeRabbitApiSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        # R2-25: Register addCleanup immediately as each resource/state changes so
        # partial setup failures still restore every acquired resource cleanly.
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._temp_dir.cleanup)

        env_backup = {
            key: os.environ.get(key)
            for key in (
                "IGBT_DATA_DIR",
                "IGBT_ADMIN_TOKEN",
                "IGBT_STATION_TOKEN",
                "IGBT_ENGINEER_PIN",
                "IGBT_ENABLE_LIVE_CAMERA",
            )
        }

        def _restore_env() -> None:
            for key, value in env_backup.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(_restore_env)
        os.environ["IGBT_DATA_DIR"] = self._temp_dir.name
        os.environ["IGBT_ADMIN_TOKEN"] = "static-admin-secret-xyz"
        os.environ["IGBT_STATION_TOKEN"] = "static-station-secret-xyz"
        os.environ.pop("IGBT_ENGINEER_PIN", None)
        os.environ["IGBT_ENABLE_LIVE_CAMERA"] = "0"

        ui_app._clear_engineer_sessions()
        self.addCleanup(ui_app._clear_engineer_sessions)

        def _clear_purge_caches() -> None:
            with ui_app._EVIDENCE_IO_LOCK:
                db_key = (Path(self._temp_dir.name) / "station.sqlite3").resolve()
                ui_app._PURGE_CUTOFF_MONOTONIC_BY_DB.pop(db_key, None)
                ui_app._DELETED_CYCLES_BY_DB.pop(db_key, None)

        self.addCleanup(_clear_purge_caches)

        with ui_app.LIVE_CAMERA._lock:
            cam_backup = {
                "camera_online": ui_app.LIVE_CAMERA.camera_online,
                "_running": ui_app.LIVE_CAMERA._running,
                "simulation_override_until": ui_app.LIVE_CAMERA.simulation_override_until,
                "work_mode": ui_app.LIVE_CAMERA.work_mode,
                "mirror_horizontal": ui_app.LIVE_CAMERA.mirror_horizontal,
                "_alarm_latched": ui_app.LIVE_CAMERA._alarm_latched,
                "_latched_reason": ui_app.LIVE_CAMERA._latched_reason,
                "_raw_frame": ui_app.LIVE_CAMERA._raw_frame,
                "_raw_frame_state": getattr(ui_app.LIVE_CAMERA, "_raw_frame_state", None),
                "_raw_frame_label_box": getattr(ui_app.LIVE_CAMERA, "_raw_frame_label_box", None),
                "_frame_seq": getattr(ui_app.LIVE_CAMERA, "_frame_seq", 0),
                "_hud_jpeg": ui_app.LIVE_CAMERA._hud_jpeg,
                "_crop_jpeg": ui_app.LIVE_CAMERA._crop_jpeg,
                "_tracked_label_box": ui_app.LIVE_CAMERA._tracked_label_box,
                "_last_tracked_crop": ui_app.LIVE_CAMERA._last_tracked_crop,
                "_onnx_model_version": ui_app.LIVE_CAMERA._onnx_model_version,
                "_ort_session": ui_app.LIVE_CAMERA._ort_session,
                "_state": dict(ui_app.LIVE_CAMERA._state),
            }

            def _restore_camera() -> None:
                with ui_app.LIVE_CAMERA._lock:
                    ui_app.LIVE_CAMERA.camera_online = cam_backup["camera_online"]
                    ui_app.LIVE_CAMERA._running = cam_backup["_running"]
                    ui_app.LIVE_CAMERA.simulation_override_until = cam_backup["simulation_override_until"]
                    ui_app.LIVE_CAMERA.work_mode = cam_backup["work_mode"]
                    ui_app.LIVE_CAMERA.mirror_horizontal = cam_backup["mirror_horizontal"]
                    ui_app.LIVE_CAMERA._alarm_latched = cam_backup["_alarm_latched"]
                    ui_app.LIVE_CAMERA._latched_reason = cam_backup["_latched_reason"]
                    ui_app.LIVE_CAMERA._raw_frame = cam_backup["_raw_frame"]
                    if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_state"):
                        ui_app.LIVE_CAMERA._raw_frame_state = cam_backup["_raw_frame_state"]
                    if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_label_box"):
                        ui_app.LIVE_CAMERA._raw_frame_label_box = cam_backup["_raw_frame_label_box"]
                    if hasattr(ui_app.LIVE_CAMERA, "_frame_seq"):
                        ui_app.LIVE_CAMERA._frame_seq = cam_backup["_frame_seq"]
                    ui_app.LIVE_CAMERA._hud_jpeg = cam_backup["_hud_jpeg"]
                    ui_app.LIVE_CAMERA._crop_jpeg = cam_backup["_crop_jpeg"]
                    ui_app.LIVE_CAMERA._tracked_label_box = cam_backup["_tracked_label_box"]
                    ui_app.LIVE_CAMERA._last_tracked_crop = cam_backup["_last_tracked_crop"]
                    ui_app.LIVE_CAMERA._onnx_model_version = cam_backup["_onnx_model_version"]
                    ui_app.LIVE_CAMERA._ort_session = cam_backup["_ort_session"]
                    ui_app.LIVE_CAMERA._state = dict(cam_backup["_state"])

            self.addCleanup(_restore_camera)
            ui_app.LIVE_CAMERA.camera_online = False
            ui_app.LIVE_CAMERA._running = False
            ui_app.LIVE_CAMERA.simulation_override_until = 0.0
            ui_app.LIVE_CAMERA._raw_frame = None
            if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_state"):
                ui_app.LIVE_CAMERA._raw_frame_state = None
            if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_label_box"):
                ui_app.LIVE_CAMERA._raw_frame_label_box = None
            ui_app.LIVE_CAMERA._hud_jpeg = None
            ui_app.LIVE_CAMERA._crop_jpeg = None
            ui_app.LIVE_CAMERA._tracked_label_box = None
            ui_app.LIVE_CAMERA._last_tracked_crop = None
            ui_app.LIVE_CAMERA._alarm_latched = False
            ui_app.LIVE_CAMERA._latched_reason = None

        ui_app.initialize_database()
        self.client = TestClient(ui_app.app, base_url="http://127.0.0.1:8765")
        self.addCleanup(self.client.close)

    def _mint_via_api(self, passcode: str = "static-admin-secret-xyz") -> str:
        res = self.client.post("/api/auth/verify", json={"passcode": passcode})
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()["admin_token"]

    # ------------------------------------------------------------------
    # BE-30: Startup ONNX autoload falls back to newest verified model
    # ------------------------------------------------------------------
    def test_be30_startup_autoload_falls_back_to_previous_verified_model(self) -> None:
        runs_root = Path(self._temp_dir.name) / "runs" / "label_classifier"
        now = time.time()

        # 1. Oldest but verified and loadable
        valid_dir = runs_root / "mnetv3_v1_good"
        valid_dir.mkdir(parents=True)
        valid_bytes = b"VALID_ONNX_BYTES_V1"
        valid_onnx = valid_dir / "label_classifier.onnx"
        valid_onnx.write_bytes(valid_bytes)
        (valid_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "input_size": 128,
                    "onnx_parity_ok": True,
                    "onnx_sha256": hashlib.sha256(valid_bytes).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        os.utime(valid_onnx, (now - 500, now - 500))

        # 2. Newer with valid metrics, but ONNX session load fails
        bad_load_dir = runs_root / "mnetv3_v2_corrupt_onnx"
        bad_load_dir.mkdir(parents=True)
        bad_load_bytes = b"CORRUPT_ONNX_PAYLOAD"
        bad_load_onnx = bad_load_dir / "label_classifier.onnx"
        bad_load_onnx.write_bytes(bad_load_bytes)
        (bad_load_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "input_size": 96,
                    "onnx_parity_ok": True,
                    "onnx_sha256": hashlib.sha256(bad_load_bytes).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        os.utime(bad_load_onnx, (now - 400, now - 400))

        # 3. Newer with SHA-256 mismatch against metrics.json
        sha_bad_dir = runs_root / "mnetv3_v3_sha_mismatch"
        sha_bad_dir.mkdir(parents=True)
        sha_bad_onnx = sha_bad_dir / "label_classifier.onnx"
        sha_bad_onnx.write_bytes(b"TAMPERED_BYTES")
        (sha_bad_dir / "metrics.json").write_text(
            json.dumps({"input_size": 96, "onnx_parity_ok": True, "onnx_sha256": "a" * 64}),
            encoding="utf-8",
        )
        os.utime(sha_bad_onnx, (now - 300, now - 300))

        # 4. Newer with onnx_parity_ok == False
        unverified_dir = runs_root / "mnetv3_v4_unverified"
        unverified_dir.mkdir(parents=True)
        unverified_onnx = unverified_dir / "label_classifier.onnx"
        unverified_onnx.write_bytes(b"UNVERIFIED_ONNX")
        (unverified_dir / "metrics.json").write_text(
            json.dumps({"input_size": 96, "onnx_parity_ok": False}),
            encoding="utf-8",
        )
        os.utime(unverified_onnx, (now - 200, now - 200))

        # 5. Newest with invalid JSON metrics.json
        bad_json_dir = runs_root / "mnetv3_v5_bad_metrics"
        bad_json_dir.mkdir(parents=True)
        bad_json_onnx = bad_json_dir / "label_classifier.onnx"
        bad_json_onnx.write_bytes(b"LATEST_BAD_METRICS")
        (bad_json_dir / "metrics.json").write_text("{not-valid-json", encoding="utf-8")
        os.utime(bad_json_onnx, (now - 100, now - 100))

        # 6. R2-19: Missing metrics.json entirely (must NOT self-verify using own digest)
        no_meta_dir = runs_root / "mnetv3_v6_no_metrics"
        no_meta_dir.mkdir(parents=True)
        no_meta_onnx = no_meta_dir / "label_classifier.onnx"
        no_meta_onnx.write_bytes(b"NO_METRICS_FILE")
        os.utime(no_meta_onnx, (now - 80, now - 80))

        # 7. R2-19: metrics.json present but missing affirmative parity key
        no_parity_dir = runs_root / "mnetv3_v7_missing_parity"
        no_parity_dir.mkdir(parents=True)
        no_parity_bytes = b"MISSING_PARITY_FLAG"
        no_parity_onnx = no_parity_dir / "label_classifier.onnx"
        no_parity_onnx.write_bytes(no_parity_bytes)
        (no_parity_dir / "metrics.json").write_text(
            json.dumps({"input_size": 96, "onnx_sha256": hashlib.sha256(no_parity_bytes).hexdigest()}),
            encoding="utf-8",
        )
        os.utime(no_parity_onnx, (now - 60, now - 60))

        # 8. R2-19: metrics.json has onnx_parity_ok=True but missing recorded onnx_sha256
        no_sha_dir = runs_root / "mnetv3_v8_missing_sha"
        no_sha_dir.mkdir(parents=True)
        no_sha_onnx = no_sha_dir / "label_classifier.onnx"
        no_sha_onnx.write_bytes(b"MISSING_SHA_RECORD")
        (no_sha_dir / "metrics.json").write_text(
            json.dumps({"input_size": 96, "onnx_parity_ok": True}),
            encoding="utf-8",
        )
        os.utime(no_sha_onnx, (now - 40, now - 40))

        attempted: list[str] = []

        def fake_load(onnx_path, *, model_version, sha256=None, input_size=96, expected_sha256=None):
            attempted.append(model_version)
            if model_version == "mnetv3_v2_corrupt_onnx":
                raise RuntimeError("Invalid protobuf")
            return {"model_version": model_version, "input_size": input_size, "sha256": sha256}

        with mock.patch.object(ui_app.LIVE_CAMERA, "load_onnx_model", side_effect=fake_load):
            ui_app._autoload_latest_onnx_model()

        # Only v2 (which passed affirmative parity + sha checks then failed load) and v1 (loaded!) should be attempted
        self.assertEqual(attempted, ["mnetv3_v2_corrupt_onnx", "mnetv3_v1_good"])

    # ------------------------------------------------------------------
    # BE-31: Remove default PIN; independent expiring session token
    # ------------------------------------------------------------------
    def test_be31_zero_default_pin_and_expiring_engineer_session(self) -> None:
        status = self.client.get("/api/auth/status")
        self.assertEqual(status.status_code, 200)
        status_body = status.json()
        self.assertFalse(status_body["default_pin_active"])
        self.assertFalse(status_body["pin_configured"])
        self.assertTrue(status_body["role_gate_enabled"])

        # Built-in "888888" must NOT work when IGBT_ENGINEER_PIN is unset
        self.assertEqual(self.client.post("/api/auth/verify", json={"passcode": "888888"}).status_code, 401)

        # Static admin token authenticates, but response NEVER returns static secret
        res1 = self.client.post("/api/auth/verify", json={"passcode": "static-admin-secret-xyz"})
        self.assertEqual(res1.status_code, 200)
        body1 = res1.json()
        self.assertTrue(body1["authenticated"])
        self.assertEqual(body1["role"], "engineer")
        self.assertEqual(body1["expires_in_seconds"], ui_app.ENGINEER_SESSION_TTL_SECONDS)
        session_token = body1["admin_token"]
        self.assertNotEqual(session_token, "static-admin-secret-xyz")
        self.assertGreaterEqual(len(session_token), 32)

        # Second verification mints a distinct token
        session_token_2 = self._mint_via_api("static-admin-secret-xyz")
        self.assertNotEqual(session_token, session_token_2)

        # Session token works on protected endpoints
        ok_mode = self.client.post(
            "/api/camera/work_mode",
            headers={"X-Admin-Token": session_token},
            json={"work_mode": "strict"},
        )
        self.assertEqual(ok_mode.status_code, 200)

        # Expiring the session token invalidates it (fail closed 401)
        digest = ui_app._session_token_digest(session_token)
        with ui_app._ENGINEER_SESSIONS_LOCK:
            ui_app._ENGINEER_SESSIONS[digest] = time.monotonic() - 1.0
        expired_mode = self.client.post(
            "/api/camera/work_mode",
            headers={"X-Admin-Token": session_token},
            json={"work_mode": "debug"},
        )
        self.assertEqual(expired_mode.status_code, 401)

        # Explicit IGBT_ENGINEER_PIN works even when IGBT_ADMIN_TOKEN is unset
        os.environ.pop("IGBT_ADMIN_TOKEN", None)
        os.environ["IGBT_ENGINEER_PIN"] = "246810"
        st2 = self.client.get("/api/auth/status").json()
        self.assertTrue(st2["pin_configured"])
        self.assertFalse(st2["default_pin_active"])
        self.assertEqual(self.client.post("/api/auth/verify", json={"passcode": "888888"}).status_code, 401)
        pin_token = self._mint_via_api("246810")
        self.assertNotEqual(pin_token, "246810")
        self.assertEqual(
            self.client.post(
                "/api/camera/work_mode",
                headers={"X-Admin-Token": pin_token},
                json={"work_mode": "debug"},
            ).status_code,
            200,
        )

        # When BOTH IGBT_ENGINEER_PIN and IGBT_ADMIN_TOKEN are unset, auth fails closed (401)
        os.environ.pop("IGBT_ENGINEER_PIN", None)
        ui_app._clear_engineer_sessions()
        self.assertEqual(self.client.post("/api/auth/verify", json={"passcode": "888888"}).status_code, 401)
        self.assertEqual(self.client.post("/api/camera/work_mode", json={"work_mode": "debug"}).status_code, 401)

    # ------------------------------------------------------------------
    # BE-32 & FE-04: Disconnected live camera never returns persisted PASS
    # ------------------------------------------------------------------
    def test_be32_and_fe04_disconnected_live_camera_and_full_cycle_counts(self) -> None:
        # Persist a live PASS cycle via _on_live_cycle_update
        live_pass_state = {
            **ui_app.default_station_state(),
            "camera_status": "ONLINE",
            "view_valid": True,
            "quality_ok": True,
            "label_status": "crossed_out",
            "fsm_state": "PASS",
            "outcome": "PASS",
            "result_reason": "A_CROSSED_OUT",
            "cycle_id": "LIVE_PASS_HISTORY_01",
            "live_stream_url": "/api/camera/stream",
            "frame_id": 42,
        }
        self.assertTrue(ui_app._on_live_cycle_update(live_pass_state, "INSPECTING"))

        # Camera disconnects; in-memory state may still hold stale PASS until cleared
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA.camera_online = False
            ui_app.LIVE_CAMERA._state["camera_status"] = "NOT_CONNECTED"
            ui_app.LIVE_CAMERA._state["view_valid"] = False
            ui_app.LIVE_CAMERA._state["outcome"] = "PASS"
            ui_app.LIVE_CAMERA._state["fsm_state"] = "PASS"

        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertFalse(body["online"])
        self.assertFalse(body["view_valid"])
        self.assertNotEqual(body["camera_status"], "ONLINE")
        self.assertNotEqual(body["outcome"], "PASS")
        self.assertNotIn("live_stream_url", body)

        # Completed cycle remains preserved in cycles history
        cycles = self.client.get("/api/cycles").json()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0]["cycle_id"], "LIVE_PASS_HISTORY_01")
        self.assertEqual(cycles[0]["final_outcome"], "PASS")
        self.assertEqual(body["cycle_summary_all"]["total"], 1)
        self.assertEqual(body["cycle_summary_all"]["pass"], 1)

        # FE-04: Insert >30 cycles and verify cycle_summary_all reflects full DB total (not 30-row cap)
        now = ui_app.utc_now()
        with ui_app.connect_db() as db:
            for i in range(34):
                outcome = "ALARM" if i % 5 == 0 else "PASS"
                db.execute(
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state,source) VALUES(?,?,?,?,?,?)",
                    (f"CYC_BULK_{i:03d}", now, now, outcome, outcome, "station"),
                )
        st_full = self.client.get("/api/state").json()
        self.assertEqual(st_full["cycle_summary_all"]["total"], 35)
        self.assertEqual(st_full["cycle_summary_all"]["alarm"], 7)
        self.assertEqual(st_full["cycle_summary_all"]["pass"], 28)

    # ------------------------------------------------------------------
    # BE-33: Live dashboard applies validity gate (view_valid=False != PASS)
    # ------------------------------------------------------------------
    def test_be33_live_dashboard_validity_gate_suppresses_pass(self) -> None:
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA.camera_online = True
            ui_app.LIVE_CAMERA.simulation_override_until = 0.0
            ui_app.LIVE_CAMERA._state.update(
                {
                    "camera_status": "ONLINE",
                    "view_valid": False,
                    "quality_ok": True,
                    "label_status": "crossed_out",
                    "fsm_state": "PASS",
                    "outcome": "PASS",
                    "result_reason": "A_CROSSED_OUT",
                }
            )
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["outcome"], "VIEW_INVALID")
        self.assertEqual(body["result_reason"], "SHADOW_PASS_REJECTED_MISSING_VALID_EVIDENCE")

        # Quality invalid also suppresses PASS to ALARM
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA._state["view_valid"] = True
            ui_app.LIVE_CAMERA._state["quality_ok"] = False
        res2 = self.client.get("/api/state")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["outcome"], "ALARM")

        # Valid live PASS passes through cleanly
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA._state["quality_ok"] = True
        res3 = self.client.get("/api/state")
        self.assertEqual(res3.status_code, 200)
        self.assertEqual(res3.json()["outcome"], "PASS")

    # ------------------------------------------------------------------
    # BE-34 / R2-26: Async disconnect-aware bounded MJPEG video generators
    # ------------------------------------------------------------------
    def test_be34_mjpeg_streams_are_async_disconnect_aware_and_bounded(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(ui_app.camera_mjpeg_stream))
        self.assertTrue(inspect.iscoroutinefunction(ui_app.camera_crop_mjpeg_stream))

        async def _consume_http_stream_with_deadline(path: str, deadline_s: float = 1.5) -> tuple[int, bytes]:
            transport = httpx.ASGITransport(app=ui_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as ac:
                async def _read_all() -> tuple[int, bytes]:
                    async with ac.stream("GET", path) as resp:
                        chunks: list[bytes] = []
                        async for chunk in resp.aiter_bytes():
                            chunks.append(chunk)
                        return resp.status_code, b"".join(chunks)

                return await asyncio.wait_for(_read_all(), timeout=deadline_s)

        # 1. With no frame cached and camera offline, both endpoints exit cleanly before independent deadline
        t0 = time.monotonic()
        with mock.patch.object(ui_app, "STREAM_OFFLINE_NO_FRAME_TIMEOUT_S", 0.08):
            s1, _ = asyncio.run(_consume_http_stream_with_deadline("/api/camera/stream", deadline_s=1.5))
            s2, _ = asyncio.run(_consume_http_stream_with_deadline("/api/camera/stream_crop", deadline_s=1.5))
        elapsed = time.monotonic() - t0
        self.assertEqual(s1, 200)
        self.assertEqual(s2, 200)
        self.assertLess(elapsed, 1.5)

        # 2. Disconnect exits immediately under independent deadline even while camera is marked active
        class _FakeDisconnectRequest:
            def __init__(self) -> None:
                self.checks = 0

            async def is_disconnected(self) -> bool:
                self.checks += 1
                return self.checks >= 2

        async def _consume_until_disconnect() -> list[bytes]:
            req = _FakeDisconnectRequest()
            gen = ui_app._mjpeg_stream_generator(req, lambda: None, 0.01, "hud")
            chunks: list[bytes] = []
            try:
                async for chunk in gen:
                    chunks.append(chunk)
            finally:
                await gen.aclose()
            return chunks

        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA.camera_online = True
        t1 = time.monotonic()
        chunks = asyncio.run(asyncio.wait_for(_consume_until_disconnect(), timeout=1.0))
        self.assertEqual(chunks, [])
        self.assertLess(time.monotonic() - t1, 0.5)

        # 3. Cached frame yields valid multipart JPEG frame within independent deadline
        sample_jpg = _make_jpeg_bytes()
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA.camera_online = False
            ui_app.LIVE_CAMERA._running = False
            ui_app.LIVE_CAMERA._hud_jpeg = sample_jpg
        s3, body3 = asyncio.run(_consume_http_stream_with_deadline("/api/camera/stream", deadline_s=1.5))
        self.assertEqual(s3, 200)
        self.assertIn(b"Content-Type: image/jpeg", body3)
        self.assertIn(sample_jpg, body3)

    # ------------------------------------------------------------------
    # BE-35 / R2-37: Atomic capture pairs image + crop + metadata from ONE frame
    # ------------------------------------------------------------------
    def test_be35_atomic_capture_uses_single_frame_for_image_crop_and_metadata(self) -> None:
        # Frame 101: green background with a distinctive bright BLUE marker ONLY in top-left box (0.05..0.45)
        frame1 = np.full((180, 320, 3), (20, 210, 30), dtype=np.uint8)
        frame1[9:81, 16:144] = (235, 30, 20)  # bright blue marker exclusively inside Frame 101's tracked box
        # Frame 202: red background with a completely different tracked box (0.55..0.95)
        frame2 = np.full((180, 320, 3), (20, 25, 220), dtype=np.uint8)

        frame1_box = (0.05, 0.05, 0.45, 0.45)
        frame2_box = (0.55, 0.55, 0.95, 0.95)

        def reset_to_frame1() -> None:
            with ui_app.LIVE_CAMERA._lock:
                ui_app.LIVE_CAMERA._raw_frame = frame1.copy()
                ui_app.LIVE_CAMERA._tracked_label_box = frame1_box
                if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_label_box"):
                    ui_app.LIVE_CAMERA._raw_frame_label_box = frame1_box
                ui_app.LIVE_CAMERA._state["frame_id"] = 101
                ui_app.LIVE_CAMERA._state["capture_monotonic_s"] = 111.5
                ui_app.LIVE_CAMERA._state["cycle_id"] = "LIVE_ATOMIC_101"
                if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_state"):
                    ui_app.LIVE_CAMERA._raw_frame_state = dict(ui_app.LIVE_CAMERA._state)

        orig_encode = ui_app._encode_bgr_jpeg

        def racing_encode(frame_bgr: np.ndarray, quality: int = 92) -> bytes | None:
            # Simulate camera thread publishing Frame #2 AND a different tracked box right while JPEG encodes
            with ui_app.LIVE_CAMERA._lock:
                ui_app.LIVE_CAMERA._raw_frame = frame2.copy()
                ui_app.LIVE_CAMERA._tracked_label_box = frame2_box
                if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_label_box"):
                    ui_app.LIVE_CAMERA._raw_frame_label_box = frame2_box
                ui_app.LIVE_CAMERA._state["frame_id"] = 202
                ui_app.LIVE_CAMERA._state["capture_monotonic_s"] = 222.5
                ui_app.LIVE_CAMERA._state["cycle_id"] = "LIVE_ATOMIC_202"
                if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_state"):
                    ui_app.LIVE_CAMERA._raw_frame_state = dict(ui_app.LIVE_CAMERA._state)
            return orig_encode(frame_bgr, quality=quality)

        # A. /api/camera/capture
        reset_to_frame1()
        with mock.patch.object(ui_app, "_encode_bgr_jpeg", side_effect=racing_encode):
            cap_res = self.client.post("/api/camera/capture")
        self.assertEqual(cap_res.status_code, 200, cap_res.text)
        cap_body = cap_res.json()
        self.assertEqual(cap_body["cycle_id"], "LIVE_ATOMIC_101")
        self.assertEqual(cap_body["frame_id"], 101)
        self.assertEqual(cap_body["capture_monotonic_s"], 111.5)
        img_bytes = self.client.get(cap_body["url"]).content
        decoded = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        # Bottom-right region of archived full-frame is green (Frame 101), not red (Frame 202)
        self.assertGreater(float(np.mean(decoded[100:170, 180:300, 1])), 180.0)
        self.assertLess(float(np.mean(decoded[100:170, 180:300, 2])), 60.0)

        # B. /api/dataset/capture (R2-37 tracked-crop case: box omitted -> MUST use Frame 101's tracked box with blue marker!)
        reset_to_frame1()
        with mock.patch.object(ui_app, "_encode_bgr_jpeg", side_effect=racing_encode):
            ds_tracked = self.client.post(
                "/api/dataset/capture",
                json={"label": "complete", "pipe_color": "yellow"},
            )
        self.assertEqual(ds_tracked.status_code, 200, ds_tracked.text)
        self.assertEqual(ds_tracked.json()["box_xyxy"], "0.0500,0.0500,0.4500,0.4500")
        ev_list = self.client.get("/api/evidence?limit=10").json()
        tracked_ev = next(e for e in ev_list if e["metadata"].get("sample_id") == ds_tracked.json()["sample_id"])
        self.assertEqual(tracked_ev["frame_id"], 101)
        self.assertEqual(tracked_ev["capture_monotonic_s"], 111.5)
        tracked_crop_bytes = self.client.get(ds_tracked.json()["image_url"]).content
        tracked_crop = cv2.imdecode(np.frombuffer(tracked_crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(tracked_crop)
        # Blue channel dominates ONLY in Frame 101's tracked box (0.05..0.45)
        self.assertGreater(float(np.mean(tracked_crop[:, :, 0])), 190.0)
        self.assertLess(float(np.mean(tracked_crop[:, :, 1])), 70.0)
        self.assertLess(float(np.mean(tracked_crop[:, :, 2])), 60.0)

        # C. /api/dataset/capture explicit-box control (bottom-right of Frame 101 -> green, NOT Frame 202 red or blue marker)
        reset_to_frame1()
        with mock.patch.object(ui_app, "_encode_bgr_jpeg", side_effect=racing_encode):
            ds_explicit = self.client.post(
                "/api/dataset/capture",
                json={"label": "complete", "pipe_color": "yellow", "box": [0.55, 0.55, 0.95, 0.95]},
            )
        self.assertEqual(ds_explicit.status_code, 200, ds_explicit.text)
        self.assertEqual(ds_explicit.json()["box_xyxy"], "0.5500,0.5500,0.9500,0.9500")
        explicit_crop_bytes = self.client.get(ds_explicit.json()["image_url"]).content
        explicit_crop = cv2.imdecode(np.frombuffer(explicit_crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(explicit_crop)
        self.assertGreater(float(np.mean(explicit_crop[:, :, 1])), 180.0)
        self.assertLess(float(np.mean(explicit_crop[:, :, 0])), 60.0)
        self.assertLess(float(np.mean(explicit_crop[:, :, 2])), 60.0)

    # ------------------------------------------------------------------
    # BE-36: Live alarm reset through simulation endpoint requires auth
    # ------------------------------------------------------------------
    def test_be36_live_alarm_reset_requires_authorization(self) -> None:
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA.camera_online = True
            ui_app.LIVE_CAMERA._alarm_latched = True
            ui_app.LIVE_CAMERA._latched_reason = "BARE_HAND_VIOLATION"
            ui_app.LIVE_CAMERA._state["alarm_latched"] = True
            ui_app.LIVE_CAMERA._state["outcome"] = "ALARM"

        # Unauthorized live reset must be rejected and preserve alarm latch
        unauth = self.client.post("/api/runtime/simulate/reset")
        self.assertEqual(unauth.status_code, 401)
        with ui_app.LIVE_CAMERA._lock:
            self.assertTrue(ui_app.LIVE_CAMERA._alarm_latched)

        # Authorized live reset clears the latch
        token = self._mint_via_api()
        auth_res = self.client.post("/api/runtime/simulate/reset", headers={"X-Admin-Token": token})
        self.assertEqual(auth_res.status_code, 200)
        with ui_app.LIVE_CAMERA._lock:
            self.assertFalse(ui_app.LIVE_CAMERA._alarm_latched)

        # Offline simulation reset stays separate from live control
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA.camera_online = False
        offline_res = self.client.post("/api/runtime/simulate/reset")
        self.assertEqual(offline_res.status_code, 200)

    # ------------------------------------------------------------------
    # BE-37: Simulated-only purge uses cycle provenance, NOT CYC_ prefix
    # ------------------------------------------------------------------
    def test_be37_simulated_only_purge_preserves_real_cyc_prefix_cycles(self) -> None:
        # 1. Real station cycle with CYC_ prefix via /api/runtime/snapshot + evidence upload
        snap_res = self.client.post(
            "/api/runtime/snapshot",
            headers={"X-Station-Token": "static-station-secret-xyz"},
            json={
                "camera_status": "ONLINE",
                "label_status": "crossed_out",
                "view_valid": True,
                "quality_ok": True,
                "fsm_state": "PASS",
                "outcome": "PASS",
                "cycle_id": "CYC_REAL_STATION_001",
            },
        )
        self.assertEqual(snap_res.status_code, 200)
        jpg = _make_jpeg_bytes()
        ev_res = self.client.post(
            "/api/cycles/CYC_REAL_STATION_001/evidence/A",
            content=jpg,
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(len(jpg)),
                "X-Station-Token": "static-station-secret-xyz",
                "X-Frame-Id": "7",
                "X-Camera-Metadata": '{"camera":"basler"}',
                "X-Captured-At-Utc": "2026-11-23T10:00:00.000Z",
            },
        )
        self.assertEqual(ev_res.status_code, 200)
        real_ev_id = ev_res.json()["evidence_id"]

        # 2. Simulated cycle via /api/runtime/simulate/a_pass
        sim_res = self.client.post("/api/runtime/simulate/a_pass")
        self.assertEqual(sim_res.status_code, 200)
        sim_cycle_id = sim_res.json()["cycle_id"]

        # 3. Purge simulated_only -> deletes sim_cycle_id, preserves CYC_REAL_STATION_001
        token = self._mint_via_api()
        purge = self.client.post(
            "/api/cycles/purge",
            headers={"X-Admin-Token": token},
            json={"scope": "simulated_only", "confirm": True},
        )
        self.assertEqual(purge.status_code, 200)
        self.assertEqual(purge.json()["deleted_cycles"], 1)

        remaining = [c["cycle_id"] for c in self.client.get("/api/cycles").json()]
        self.assertIn("CYC_REAL_STATION_001", remaining)
        self.assertNotIn(sim_cycle_id, remaining)
        self.assertEqual(self.client.get(f"/api/evidence/{real_ev_id}/file").status_code, 200)

        # 4. R2-20 & R2-38: Schema migration & backfill on legacy DB lacking cycles.source column
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as legacy_dir:
            legacy_db = Path(legacy_dir) / "station.sqlite3"
            with contextlib.closing(sqlite3.connect(legacy_db)) as conn, conn:
                conn.executescript(
                    """
                    CREATE TABLE cycles (
                        cycle_id TEXT PRIMARY KEY,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL,
                        initial_label TEXT,
                        final_outcome TEXT NOT NULL DEFAULT 'WAITING',
                        result_reason TEXT,
                        pipe_color TEXT,
                        operator_anon_id TEXT,
                        model_version TEXT,
                        config_version TEXT,
                        fsm_state TEXT NOT NULL DEFAULT 'WAIT_PART'
                    );
                    CREATE TABLE events (
                        event_id TEXT PRIMARY KEY,
                        cycle_id TEXT,
                        created_at_utc TEXT NOT NULL,
                        capture_monotonic_s REAL,
                        frame_id INTEGER,
                        event_type TEXT NOT NULL,
                        state_before TEXT,
                        state_after TEXT,
                        outcome TEXT,
                        reason TEXT,
                        payload_json TEXT NOT NULL
                    );
                    CREATE TABLE evidence (
                        evidence_id TEXT PRIMARY KEY,
                        cycle_id TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        file_relpath TEXT NOT NULL UNIQUE,
                        mime_type TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        frame_id INTEGER,
                        capture_monotonic_s REAL,
                        metadata_json TEXT NOT NULL
                    );
                    """
                )
                ts = "2026-01-01T00:00:00.000Z"
                # (a) Ambiguous legacy row with no events/evidence -> must stay 'station'
                conn.execute(
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state) VALUES('CYC_LEGACY_AMBIG',?,?,'PASS','PASS')",
                    (ts, ts),
                )
                # (b) Legacy A-pass simulation identified by SCENARIO_A_PASS evidence metadata -> 'simulated'
                conn.execute(
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state) VALUES('CYC_A_LEGACY',?,?,'PASS','PASS')",
                    (ts, ts),
                )
                conn.execute(
                    "INSERT INTO events VALUES('ev_sim_a','CYC_A_LEGACY',?,1000.5,5,'A_CROSSED_OUT','INSPECTING','PASS','PASS','A_CROSSED_OUT',?)",
                    (ts, json.dumps({"record_type": "STATE_TRANSITION", "reason": "A_CROSSED_OUT"})),
                )
                conn.execute(
                    "INSERT INTO evidence VALUES('img_sim_a','CYC_A_LEGACY','ev_sim_a',?,'A','evidence/CYC_A_LEGACY/a.jpg','image/jpeg','0',5,1000.5,?)",
                    (ts, json.dumps({"operation_event": "SCENARIO_A_PASS"})),
                )
                # (c) Legacy C-pass simulation with STATE_TRANSITION-only events (evidence already removed) -> 'simulated'
                conn.execute(
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state) VALUES('CYC_C_LEGACY',?,?,'PASS','PASS')",
                    (ts, ts),
                )
                conn.execute(
                    "INSERT INTO events VALUES('ev_sim_c','CYC_C_LEGACY',?,1001.2,12,'C_REMOVED','VERIFY_C_ABSENT','PASS','PASS','C_REMOVED',?)",
                    (ts, json.dumps({"record_type": "STATE_TRANSITION", "reason": "C_REMOVED"})),
                )
                # (d) Legacy real station cycle with CYC_ prefix and RuntimeSnapshot event -> must stay 'station'
                conn.execute(
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state) VALUES('CYC_STATION_LEGACY',?,?,'PASS','PASS')",
                    (ts, ts),
                )
                conn.execute(
                    "INSERT INTO events VALUES('ev_st','CYC_STATION_LEGACY',?,50.0,3,'STATE_CHANGE','INSPECTING','PASS','PASS','A_CROSSED_OUT',?)",
                    (ts, json.dumps({"mode": "shadow", "camera_status": "ONLINE", "outcome": "PASS"})),
                )
                # (e) Legacy live cycle -> 'live'
                conn.execute(
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state) VALUES('LIVE_LEGACY_01',?,?,'PASS','PASS')",
                    (ts, ts),
                )
                conn.execute(
                    "INSERT INTO events VALUES('ev_live','LIVE_LEGACY_01',?,10.0,1,'LIVE_CAPTURE','INSPECTING','INSPECTING','WAITING','manual',?)",
                    (ts, json.dumps({"operation_event": "LIVE_CAMERA_MANUAL_CAPTURE"})),
                )
            old_dir = os.environ["IGBT_DATA_DIR"]
            os.environ["IGBT_DATA_DIR"] = legacy_dir
            try:
                ui_app.initialize_database()
                with ui_app.connect_db() as db:
                    sources = {
                        r["cycle_id"]: r["source"]
                        for r in db.execute("SELECT cycle_id, source FROM cycles").fetchall()
                    }
                self.assertEqual(sources["CYC_A_LEGACY"], "simulated")
                self.assertEqual(sources["CYC_C_LEGACY"], "simulated")
                self.assertEqual(sources["CYC_STATION_LEGACY"], "station")
                self.assertEqual(sources["CYC_LEGACY_AMBIG"], "station")
                self.assertEqual(sources["LIVE_LEGACY_01"], "live")
            finally:
                os.environ["IGBT_DATA_DIR"] = old_dir

    # ------------------------------------------------------------------
    # BE-38 / R2-39: Purge all does not delete files created after purge transaction
    # ------------------------------------------------------------------
    def test_be38_purge_all_preserves_files_created_after_transaction(self) -> None:
        self.assertEqual(self.client.post("/api/runtime/simulate/a_pass").status_code, 200)
        evidence_root = Path(self._temp_dir.name) / "evidence"
        orphan_file = evidence_root / "ORPHAN_DIR" / "pre_existing_orphan.jpg"
        orphan_file.parent.mkdir(parents=True, exist_ok=True)
        orphan_file.write_bytes(_make_jpeg_bytes())

        post_tx_file = evidence_root / "NEW_AFTER_TX" / "fresh_capture.jpg"
        orig_unlink = ui_app._safe_unlink_evidence
        created_once = False
        targeted_relpaths: list[str] = []

        def unlink_then_simulate_concurrent_new_file(data_root, ev_root, relpath):
            nonlocal created_once
            targeted_relpaths.append(str(relpath))
            result = orig_unlink(data_root, ev_root, relpath)
            if not created_once:
                created_once = True
                post_tx_file.parent.mkdir(parents=True, exist_ok=True)
                post_tx_file.write_bytes(_make_jpeg_bytes())
            return result

        token = self._mint_via_api()
        with mock.patch.object(ui_app, "_safe_unlink_evidence", side_effect=unlink_then_simulate_concurrent_new_file):
            purge = self.client.post(
                "/api/cycles/purge",
                headers={"X-Admin-Token": token},
                json={"scope": "all", "confirm": True},
            )
        self.assertEqual(purge.status_code, 200)
        self.assertTrue(created_once)
        self.assertFalse(any("fresh_capture.jpg" in p for p in targeted_relpaths))
        self.assertFalse(orphan_file.exists())
        self.assertTrue(post_tx_file.is_file(), "Post-transaction evidence file must NEVER be swept")

    # ------------------------------------------------------------------
    # BE-39: Authenticate external evidence writes
    # ------------------------------------------------------------------
    def test_be39_upload_evidence_requires_station_or_admin_token(self) -> None:
        jpg = _make_jpeg_bytes()
        base_headers = {
            "Content-Type": "image/jpeg",
            "Content-Length": str(len(jpg)),
            "X-Frame-Id": "1",
            "X-Camera-Metadata": '{"camera":"c920"}',
            "X-Captured-At-Utc": "2026-11-23T10:00:00.000Z",
        }
        # Missing token -> 401 and no cycle created
        r_unauth = self.client.post("/api/cycles/CYC_UNAUTH_1/evidence/initial_label", content=jpg, headers=base_headers)
        self.assertEqual(r_unauth.status_code, 401)
        self.assertEqual(self.client.get("/api/cycles").json(), [])

        # Wrong token -> 401
        r_bad = self.client.post(
            "/api/cycles/CYC_UNAUTH_1/evidence/initial_label",
            content=jpg,
            headers={**base_headers, "X-Station-Token": "wrong"},
        )
        self.assertEqual(r_bad.status_code, 401)

        # Valid station token -> 200
        r_station = self.client.post(
            "/api/cycles/CYC_AUTH_STATION/evidence/initial_label",
            content=jpg,
            headers={**base_headers, "X-Station-Token": "static-station-secret-xyz"},
        )
        self.assertEqual(r_station.status_code, 200)

        # Valid engineer session token -> 200
        eng_token = self._mint_via_api()
        r_eng = self.client.post(
            "/api/cycles/CYC_AUTH_ENG/evidence/A",
            content=jpg,
            headers={**base_headers, "X-Admin-Token": eng_token},
        )
        self.assertEqual(r_eng.status_code, 200)

    # ------------------------------------------------------------------
    # BE-40: Protect work-mode changes with server-side role check
    # ------------------------------------------------------------------
    def test_be40_work_mode_requires_engineer_role(self) -> None:
        ui_app.LIVE_CAMERA.set_work_mode("strict")
        self.assertEqual(self.client.post("/api/camera/work_mode", json={"work_mode": "debug"}).status_code, 401)
        self.assertEqual(ui_app.LIVE_CAMERA.work_mode, "strict")

        token = self._mint_via_api()
        ok = self.client.post(
            "/api/camera/work_mode",
            headers={"X-Admin-Token": token},
            json={"work_mode": "debug"},
        )
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["work_mode"], "debug")
        self.assertEqual(ui_app.LIVE_CAMERA.work_mode, "debug")

    # ------------------------------------------------------------------
    # BE-41: Enforce upload size limit incrementally without Content-Length
    # ------------------------------------------------------------------
    def test_be41_dataset_upload_bounded_stream_without_content_length(self) -> None:
        # 1. Incremental stop check on _read_bounded_body
        chunks_pulled = 0

        class _FakeStreamingRequest:
            async def stream(self):
                nonlocal chunks_pulled
                for _ in range(100):
                    chunks_pulled += 1
                    yield b"x" * 1024

        with self.assertRaises(ui_app.HTTPException) as ctx:
            asyncio.run(ui_app._read_bounded_body(_FakeStreamingRequest(), 3000, "TOO_LARGE"))
        self.assertEqual(ctx.exception.status_code, 413)
        # Stopped on the 3rd chunk (3072 > 3000), never pulled chunks 4..100!
        self.assertEqual(chunks_pulled, 3)

        # 2. HTTP chunked upload (no Content-Length) exceeding cap returns 413
        token = self._mint_via_api()

        def oversized_chunks():
            for _ in range(10):
                yield b"\xff\xd8" + b"A" * 1024

        with mock.patch.object(ui_app, "MAX_EVIDENCE_BYTES", 2048):
            r_big = self.client.post(
                "/api/dataset/upload",
                content=oversized_chunks(),
                headers={
                    "Content-Type": "image/jpeg",
                    "X-Sample-Label": "complete",
                    "X-Admin-Token": token,
                },
            )
        self.assertEqual(r_big.status_code, 413)

        # 3. Valid chunked JPEG upload (no Content-Length) succeeds with 200
        jpg = _make_jpeg_bytes()

        def valid_chunks():
            mid = len(jpg) // 2
            yield jpg[:mid]
            yield jpg[mid:]

        r_ok = self.client.post(
            "/api/dataset/upload",
            content=valid_chunks(),
            headers={
                "Content-Type": "image/jpeg",
                "X-Sample-Label": "complete",
                "X-Admin-Token": token,
            },
        )
        self.assertEqual(r_ok.status_code, 200)

        # 4. Empty chunked body returns 400
        def empty_chunks():
            if False:
                yield b""

        r_empty = self.client.post(
            "/api/dataset/upload",
            content=empty_chunks(),
            headers={
                "Content-Type": "image/jpeg",
                "X-Sample-Label": "complete",
                "X-Admin-Token": token,
            },
        )
        self.assertEqual(r_empty.status_code, 400)

    # ------------------------------------------------------------------
    # BE-42: Dataset mutation routes enforce admin auth fail-closed
    # ------------------------------------------------------------------
    def test_be42_dataset_mutation_routes_require_admin_fail_closed(self) -> None:
        token = self._mint_via_api()
        jpg = _make_jpeg_bytes()
        up = self.client.post(
            "/api/dataset/upload",
            content=jpg,
            headers={
                "Content-Type": "image/jpeg",
                "X-Sample-Label": "complete",
                "X-Admin-Token": token,
            },
        )
        self.assertEqual(up.status_code, 200)
        sample_id = up.json()["sample_id"]

        # Unauthenticated PATCH / DELETE / clear must fail with 401 and leave stored sample unchanged (R2-40)
        self.assertEqual(
            self.client.patch(f"/api/dataset/sample/{sample_id}", json={"label": "covered"}).status_code,
            401,
        )
        samples_after_denied = self.client.get("/api/dataset/samples").json()
        denied_sample = next(s for s in samples_after_denied if s["sample_id"] == sample_id)
        self.assertEqual(denied_sample["label"], "complete")

        self.assertEqual(self.client.delete(f"/api/dataset/sample/{sample_id}").status_code, 401)
        self.assertEqual(self.client.post("/api/dataset/clear").status_code, 401)

        # Authorized PATCH and DELETE succeed
        self.assertEqual(
            self.client.patch(
                f"/api/dataset/sample/{sample_id}",
                headers={"X-Admin-Token": token},
                json={"label": "covered"},
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.delete(f"/api/dataset/sample/{sample_id}", headers={"X-Admin-Token": token}).status_code,
            200,
        )

        # Fail closed even when IGBT_ADMIN_TOKEN & IGBT_ENGINEER_PIN are BOTH unconfigured
        os.environ.pop("IGBT_ADMIN_TOKEN", None)
        os.environ.pop("IGBT_ENGINEER_PIN", None)
        ui_app._clear_engineer_sessions()
        self.assertEqual(self.client.post("/api/dataset/clear").status_code, 401)
        self.assertEqual(
            self.client.post(
                "/api/dataset/upload",
                content=jpg,
                headers={"Content-Type": "image/jpeg", "X-Sample-Label": "complete"},
            ).status_code,
            401,
        )

    # ------------------------------------------------------------------
    # BE-43 / BE-44 / BE-45: Model activation auth, SHA-256, missing artifact
    # ------------------------------------------------------------------
    def test_be43_be44_be45_model_activation_auth_digest_and_missing_artifact(self) -> None:
        token = self._mint_via_api()
        runs_root = Path(self._temp_dir.name) / "runs" / "label_classifier"

        # Model A: valid artifact with matching SHA-256
        good_bytes = b"ONNX_MODEL_A_VERIFIED_BYTES"
        good_sha = hashlib.sha256(good_bytes).hexdigest()
        dir_a = runs_root / "mnetv3_model_a"
        dir_a.mkdir(parents=True)
        (dir_a / "label_classifier.onnx").write_bytes(good_bytes)

        # Model B: missing ONNX artifact on disk
        dir_b = runs_root / "mnetv3_model_b_missing"
        dir_b.mkdir(parents=True)

        # Model C: tampered ONNX bytes on disk (digest mismatch)
        dir_c = runs_root / "mnetv3_model_c_tampered"
        dir_c.mkdir(parents=True)
        (dir_c / "label_classifier.onnx").write_bytes(b"REPLACED_MALICIOUS_BYTES")

        def reg(version: str, sha: str) -> str:
            r = self.client.post(
                "/api/models",
                headers={"X-Admin-Token": token},
                json={
                    "model_name": "mobilenetv3-small-igbt-roi",
                    "model_version": version,
                    "sha256": sha,
                    "license_status": "reviewed",
                    "notes": "test",
                },
            )
            self.assertEqual(r.status_code, 200)
            return r.json()["model_id"]

        id_a = reg("mnetv3_model_a", good_sha)
        id_b = reg("mnetv3_model_b_missing", "b" * 64)
        id_c = reg("mnetv3_model_c_tampered", "c" * 64)

        # Set known active model baseline
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA._onnx_model_version = "baseline-active-v0"
            ui_app.LIVE_CAMERA._state["model_version"] = "baseline-active-v0"

        # BE-43: Unauthorized activate -> 401, active model unchanged
        self.assertEqual(self.client.post(f"/api/models/{id_a}/activate").status_code, 401)
        self.assertEqual(ui_app.LIVE_CAMERA._onnx_model_version, "baseline-active-v0")

        # BE-45: Missing artifact -> 404, active model unchanged
        res_missing = self.client.post(f"/api/models/{id_b}/activate", headers={"X-Admin-Token": token})
        self.assertEqual(res_missing.status_code, 404)
        self.assertEqual(res_missing.json()["detail"], "MODEL_ARTIFACT_UNAVAILABLE")
        self.assertEqual(ui_app.LIVE_CAMERA._onnx_model_version, "baseline-active-v0")
        self.assertEqual(ui_app.LIVE_CAMERA._state["model_version"], "baseline-active-v0")

        # BE-44: SHA-256 mismatch -> 409, active model unchanged
        res_mismatch = self.client.post(f"/api/models/{id_c}/activate", headers={"X-Admin-Token": token})
        self.assertEqual(res_mismatch.status_code, 409)
        self.assertEqual(res_mismatch.json()["detail"], "MODEL_SHA256_MISMATCH")
        self.assertEqual(ui_app.LIVE_CAMERA._onnx_model_version, "baseline-active-v0")

        # Valid digest & artifact -> succeeds and updates active model
        def fake_load(onnx_path, *, model_version, sha256=None, input_size=96, expected_sha256=None):
            with ui_app.LIVE_CAMERA._lock:
                ui_app.LIVE_CAMERA._onnx_model_version = model_version
                ui_app.LIVE_CAMERA._state["model_version"] = model_version
            return {"model_version": model_version, "sha256": sha256, "input_size": input_size}

        with mock.patch.object(ui_app.LIVE_CAMERA, "load_onnx_model", side_effect=fake_load):
            res_ok = self.client.post(f"/api/models/{id_a}/activate", headers={"X-Admin-Token": token})
        self.assertEqual(res_ok.status_code, 200)
        self.assertEqual(res_ok.json()["mode"], "onnx_runtime")
        self.assertEqual(ui_app.LIVE_CAMERA._onnx_model_version, "mnetv3_model_a")

    # ------------------------------------------------------------------
    # BE-54: Simulated glare_invalid archives the evaluated glare frame
    # ------------------------------------------------------------------
    def test_be54_glare_invalid_archives_evaluated_glare_frame(self) -> None:
        res = self.client.post("/api/runtime/simulate/glare_invalid")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        ev_url = body["latest_evidence_url"]
        archived_bytes = self.client.get(ev_url).content

        raw_blue_bytes = BLUE_REF_IMAGE.read_bytes()
        self.assertNotEqual(
            hashlib.sha256(archived_bytes).hexdigest(),
            hashlib.sha256(raw_blue_bytes).hexdigest(),
            "glare_invalid must archive the synthetic glare frame, not the unmodified blue reference image",
        )
        decoded = cv2.imdecode(np.frombuffer(archived_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        quality = measure_label_quality(decoded, STANDARD_ROIS.label_roi)
        self.assertIsNotNone(quality)
        self.assertGreater(quality.overexposed_fraction, 0.08)

    # ------------------------------------------------------------------
    # BE-55: Simulated transitions store per-transition triggering frame_id
    # ------------------------------------------------------------------
    def test_be55_simulated_transitions_store_per_transition_frame_id(self) -> None:
        res = self.client.post("/api/runtime/simulate/a_pass")
        self.assertEqual(res.status_code, 200)
        cycle_id = res.json()["cycle_id"]
        final_frame_id = res.json()["frame_id"]

        events = self.client.get(f"/api/cycles/{cycle_id}/events").json()
        # R2-41: Assert each expected transition occurs exactly once before collapsing by event_type
        type_counts = collections.Counter(ev["event_type"] for ev in events)
        for expected_type in ("NEW_CYCLE", "INITIAL_COMPLETE_CONFIRMED", "INSPECTION_STARTED", "A_CROSSED_OUT"):
            self.assertEqual(
                type_counts[expected_type],
                1,
                f"Expected exactly 1 {expected_type} transition, got {type_counts[expected_type]}",
            )
        by_type = {ev["event_type"]: ev for ev in events}

        # Non-frame transitions must leave frame_id unset (None)
        self.assertIn("NEW_CYCLE", by_type)
        self.assertIsNone(by_type["NEW_CYCLE"]["frame_id"])
        self.assertIn("INSPECTION_STARTED", by_type)
        self.assertIsNone(by_type["INSPECTION_STARTED"]["frame_id"])

        # Earlier frame-triggered transition must record its own earlier frame_id (< final_frame_id)
        self.assertIn("INITIAL_COMPLETE_CONFIRMED", by_type)
        self.assertIn("A_CROSSED_OUT", by_type)
        init_fid = by_type["INITIAL_COMPLETE_CONFIRMED"]["frame_id"]
        pass_fid = by_type["A_CROSSED_OUT"]["frame_id"]
        self.assertIsInstance(init_fid, int)
        self.assertIsInstance(pass_fid, int)
        self.assertLess(init_fid, pass_fid)
        self.assertEqual(pass_fid, final_frame_id)

    # ------------------------------------------------------------------
    # BE-56: Decode & bound evidence images before storing
    # ------------------------------------------------------------------
    def test_be56_upload_evidence_decodes_and_bounds_dimensions(self) -> None:
        base_headers = {
            "X-Station-Token": "static-station-secret-xyz",
            "X-Frame-Id": "5",
            "X-Camera-Metadata": '{"camera":"basler"}',
            "X-Captured-At-Utc": "2026-11-23T10:00:00.000Z",
        }

        # 1. Corrupt JPEG with valid SOI/EOI markers -> 422, nothing written
        corrupt_jpg = b"\xff\xd8corrupted_not_valid_jpeg_body\xff\xd9"
        r_bad_jpg = self.client.post(
            "/api/cycles/CYC_CORRUPT_1/evidence/initial_label",
            content=corrupt_jpg,
            headers={**base_headers, "Content-Type": "image/jpeg", "Content-Length": str(len(corrupt_jpg))},
        )
        self.assertEqual(r_bad_jpg.status_code, 422)

        # 2. Corrupt PNG with valid magic header -> 422, nothing written
        corrupt_png = b"\x89PNG\r\n\x1a\ncorrupted_png_payload"
        r_bad_png = self.client.post(
            "/api/cycles/CYC_CORRUPT_2/evidence/initial_label",
            content=corrupt_png,
            headers={**base_headers, "Content-Type": "image/png", "Content-Length": str(len(corrupt_png))},
        )
        self.assertEqual(r_bad_png.status_code, 422)

        # 3. Oversized dimensions -> 422, nothing written
        valid_jpg = _make_jpeg_bytes(width=180, height=120)
        with mock.patch.object(ui_app, "MAX_EVIDENCE_IMAGE_WIDTH", 64):
            r_oversized = self.client.post(
                "/api/cycles/CYC_OVERSIZED_1/evidence/initial_label",
                content=valid_jpg,
                headers={**base_headers, "Content-Type": "image/jpeg", "Content-Length": str(len(valid_jpg))},
            )
        self.assertEqual(r_oversized.status_code, 422)
        self.assertEqual(self.client.get("/api/evidence").json(), [])
        self.assertEqual(self.client.get("/api/cycles").json(), [])

        # 4. Valid JPEG & valid PNG succeed
        r_ok_jpg = self.client.post(
            "/api/cycles/CYC_GOOD_1/evidence/initial_label",
            content=valid_jpg,
            headers={**base_headers, "Content-Type": "image/jpeg", "Content-Length": str(len(valid_jpg))},
        )
        self.assertEqual(r_ok_jpg.status_code, 200)
        valid_png = _make_png_bytes(width=180, height=120)
        r_ok_png = self.client.post(
            "/api/cycles/CYC_GOOD_2/evidence/A",
            content=valid_png,
            headers={**base_headers, "Content-Type": "image/png", "Content-Length": str(len(valid_png))},
        )
        self.assertEqual(r_ok_png.status_code, 200)

    # ------------------------------------------------------------------
    # BE-57: Reject invalid/inverted explicit crop box before saving sample
    # ------------------------------------------------------------------
    def test_be57_dataset_capture_rejects_invalid_explicit_crop_box(self) -> None:
        frame = np.full((180, 320, 3), (80, 140, 200), dtype=np.uint8)
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA._raw_frame = frame
            ui_app.LIVE_CAMERA._tracked_label_box = (0.3, 0.3, 0.7, 0.7)
            if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_label_box"):
                ui_app.LIVE_CAMERA._raw_frame_label_box = (0.3, 0.3, 0.7, 0.7)
            ui_app.LIVE_CAMERA._state["frame_id"] = 12
            ui_app.LIVE_CAMERA._state["capture_monotonic_s"] = 50.0
            if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_state"):
                ui_app.LIVE_CAMERA._raw_frame_state = dict(ui_app.LIVE_CAMERA._state)

        invalid_boxes = [
            [0.8, 0.2, 0.3, 0.9],    # inverted x1 > x2
            [0.2, 0.8, 0.6, 0.2],    # inverted y1 > y2
            [0.4, 0.4, 0.4, 0.8],    # zero width
            [-0.1, 0.2, 0.5, 0.8],   # out of [0, 1]
            [0.1, 0.2, 1.2, 0.8],    # out of [0, 1]
            [0.1, 0.2, 0.5],         # wrong length
        ]
        for bad_box in invalid_boxes:
            r = self.client.post(
                "/api/dataset/capture",
                json={"label": "complete", "pipe_color": "yellow", "box": bad_box},
            )
            self.assertEqual(r.status_code, 422, f"Expected 422 for invalid box {bad_box}, got {r.status_code}")

        # No samples should have been created by rejected boxes
        self.assertEqual(self.client.get("/api/dataset/samples").json(), [])

        # Valid explicit box succeeds with exact coordinates
        r_good = self.client.post(
            "/api/dataset/capture",
            json={"label": "complete", "pipe_color": "yellow", "box": [0.1, 0.2, 0.6, 0.7]},
        )
        self.assertEqual(r_good.status_code, 200, r_good.text)
        self.assertEqual(r_good.json()["box_xyxy"], "0.1000,0.2000,0.6000,0.7000")

        # Omitted box uses tracked fallback cleanly
        r_omit = self.client.post(
            "/api/dataset/capture",
            json={"label": "crossed_out", "pipe_color": "blue"},
        )
        self.assertEqual(r_omit.status_code, 200, r_omit.text)
        self.assertEqual(r_omit.json()["box_xyxy"], "0.3000,0.3000,0.7000,0.7000")

    # ------------------------------------------------------------------
    # R2-21: Queued live transitions cannot recreate purged/deleted cycles
    # ------------------------------------------------------------------
    def test_r2_21_purge_cutoff_blocks_stale_live_retries_and_allows_new_cycles(self) -> None:
        old_mono = time.monotonic() - 2.0
        old_utc = "2026-11-23T09:00:00.000+00:00"
        stale_state = {
            **ui_app.default_station_state(),
            "camera_status": "ONLINE",
            "view_valid": True,
            "quality_ok": True,
            "label_status": "crossed_out",
            "fsm_state": "PASS",
            "outcome": "PASS",
            "result_reason": "A_CROSSED_OUT",
            "cycle_id": "LIVE_STALE_PRE_PURGE",
            "frame_id": 10,
            "capture_monotonic_s": old_mono,
            "updated_at_utc": old_utc,
        }
        self.assertTrue(ui_app._on_live_cycle_update(stale_state, "INSPECTING"))
        self.assertEqual([c["cycle_id"] for c in self.client.get("/api/cycles").json()], ["LIVE_STALE_PRE_PURGE"])

        # All-scope purge invokes discard_pending_transitions(None) and records purge cutoff
        token = self._mint_via_api()
        with mock.patch.object(
            ui_app.LIVE_CAMERA,
            "discard_pending_transitions",
            wraps=getattr(ui_app.LIVE_CAMERA, "discard_pending_transitions", None),
        ) as mock_discard:
            purge = self.client.post(
                "/api/cycles/purge",
                headers={"X-Admin-Token": token},
                json={"scope": "all", "confirm": True},
            )
        self.assertEqual(purge.status_code, 200)
        if hasattr(ui_app.LIVE_CAMERA, "discard_pending_transitions"):
            mock_discard.assert_called_with(None)
        self.assertEqual(self.client.get("/api/cycles").json(), [])

        # Retrying the pre-purge transition returns True (dropped) and MUST NOT recreate the cycle
        self.assertTrue(ui_app._on_live_cycle_update(stale_state, "INSPECTING"))
        self.assertEqual(self.client.get("/api/cycles").json(), [])

        # Genuinely new cycle created after purge IS accepted and persisted
        fresh_state = {
            **ui_app.default_station_state(),
            "camera_status": "ONLINE",
            "view_valid": True,
            "quality_ok": True,
            "label_status": "covered",
            "fsm_state": "PASS",
            "outcome": "PASS",
            "result_reason": "B_COVERED",
            "cycle_id": "LIVE_FRESH_POST_PURGE",
            "frame_id": 11,
            "capture_monotonic_s": time.monotonic() + 0.05,
            "updated_at_utc": ui_app.utc_now(),
        }
        self.assertTrue(ui_app._on_live_cycle_update(fresh_state, "INSPECTING"))
        self.assertEqual([c["cycle_id"] for c in self.client.get("/api/cycles").json()], ["LIVE_FRESH_POST_PURGE"])

        # Single-cycle DELETE also prevents an in-flight retry for that cycle from resurrecting it
        del_res = self.client.delete("/api/cycles/LIVE_FRESH_POST_PURGE", headers={"X-Admin-Token": token})
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(ui_app._on_live_cycle_update(fresh_state, "INSPECTING"))
        self.assertEqual(self.client.get("/api/cycles").json(), [])

    # ------------------------------------------------------------------
    # R2-22: Blocking evidence decode/persistence runs off the event loop
    # ------------------------------------------------------------------
    def test_r2_22_upload_evidence_runs_decode_and_persistence_in_worker_thread(self) -> None:
        jpg = _make_jpeg_bytes()
        worker_threads: list[int] = []
        orig_persist = ui_app._decode_and_persist_evidence

        def spy_persist(**kwargs):
            worker_threads.append(threading.get_ident())
            return orig_persist(**kwargs)

        async def _call_async() -> tuple[int, int]:
            loop_tid = threading.get_ident()
            transport = httpx.ASGITransport(app=ui_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as ac:
                resp = await ac.post(
                    "/api/cycles/CYC_OFFLOOP_01/evidence/initial_label",
                    content=jpg,
                    headers={
                        "Content-Type": "image/jpeg",
                        "Content-Length": str(len(jpg)),
                        "X-Station-Token": "static-station-secret-xyz",
                        "X-Frame-Id": "1",
                        "X-Camera-Metadata": '{"camera":"basler"}',
                        "X-Captured-At-Utc": "2026-11-23T10:00:00.000Z",
                    },
                )
            return loop_tid, resp.status_code

        with mock.patch.object(ui_app, "_decode_and_persist_evidence", side_effect=spy_persist):
            loop_tid, status = asyncio.run(_call_async())
        self.assertEqual(status, 200)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], loop_tid, "Blocking evidence I/O must run off the asyncio event loop")

    # ------------------------------------------------------------------
    # R2-23: Evidence failure after dataset commit returns committed sample
    # ------------------------------------------------------------------
    def test_r2_23_dataset_capture_returns_committed_sample_when_evidence_fails(self) -> None:
        frame = np.full((180, 320, 3), (80, 140, 200), dtype=np.uint8)
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA._raw_frame = frame
            ui_app.LIVE_CAMERA._tracked_label_box = (0.2, 0.2, 0.6, 0.6)
            if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_label_box"):
                ui_app.LIVE_CAMERA._raw_frame_label_box = (0.2, 0.2, 0.6, 0.6)
            ui_app.LIVE_CAMERA._state["frame_id"] = 21
            ui_app.LIVE_CAMERA._state["capture_monotonic_s"] = 75.0
            if hasattr(ui_app.LIVE_CAMERA, "_raw_frame_state"):
                ui_app.LIVE_CAMERA._raw_frame_state = dict(ui_app.LIVE_CAMERA._state)

        real_connect_db = ui_app.connect_db
        call_count = 0

        def fail_on_evidence_tx():
            nonlocal call_count
            call_count += 1
            # First call is initialize_database(); fail the second call (evidence transaction after capture_sample commits).
            if call_count >= 2:
                raise sqlite3.OperationalError("database is locked")
            return real_connect_db()

        with mock.patch.object(ui_app, "connect_db", side_effect=fail_on_evidence_tx):
            res = self.client.post(
                "/api/dataset/capture",
                json={"label": "complete", "pipe_color": "yellow", "box": [0.1, 0.1, 0.5, 0.5]},
            )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertTrue(body["sample_id"])
        self.assertFalse(body["evidence_archived"])
        self.assertIn("EVIDENCE_ARCHIVE_FAILED", body["evidence_error"])
        # Exactly one training sample exists and no partial evidence file was left behind
        self.assertEqual(len(self.client.get("/api/dataset/samples").json()), 1)
        evidence_files = list((Path(self._temp_dir.name) / "evidence").rglob("*.jpg"))
        self.assertEqual(evidence_files, [])

    # ------------------------------------------------------------------
    # R2-24: Registration failure leaves previous active model unchanged
    # ------------------------------------------------------------------
    def test_on_training_completed_refuses_failed_parity_and_preserves_active_on_db_failure(self) -> None:
        onnx_file = Path(self._temp_dir.name) / "runs" / "label_classifier" / "mnet_cb" / "label_classifier.onnx"
        onnx_file.parent.mkdir(parents=True, exist_ok=True)
        raw = b"DUMMY_ONNX_MODEL"
        onnx_file.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()

        base_summary = {
            "model_name": "mobilenetv3-small-igbt-roi",
            "model_version": "mnet_cb",
            "onnx_path": str(onnx_file),
            "onnx_sha256": sha,
            "input_size": 96,
            "onnx_parity_ok": False,
            "test_accuracy": 0.95,
            "complete_false_accept_total": 0,
            "onnx_max_abs_diff": 0.5,
            "train_samples": 20,
            "val_samples": 10,
            "test_samples": 10,
        }
        self.assertFalse(ui_app._on_training_completed(base_summary))
        self.assertEqual(self.client.get("/api/models").json(), [])

        # R2-24: If DB registration fails AFTER prepare_onnx_model succeeds, active model stays untouched
        with ui_app.LIVE_CAMERA._lock:
            ui_app.LIVE_CAMERA._onnx_model_version = "prev-active-v0"
            ui_app.LIVE_CAMERA._state["model_version"] = "prev-active-v0"

        good_summary = {**base_summary, "onnx_parity_ok": True, "onnx_max_abs_diff": 1e-5}
        fake_prepared = {"model_version": "mnet_cb", "sha256": sha, "input_size": 96, "session": object()}

        def fake_commit(prepared):
            with ui_app.LIVE_CAMERA._lock:
                ui_app.LIVE_CAMERA._onnx_model_version = prepared["model_version"]
                ui_app.LIVE_CAMERA._state["model_version"] = prepared["model_version"]
            return {"model_version": prepared["model_version"]}

        with (
            mock.patch.object(ui_app.LIVE_CAMERA, "prepare_onnx_model", return_value=fake_prepared) as m_prep,
            mock.patch.object(ui_app.LIVE_CAMERA, "commit_prepared_onnx_model", side_effect=fake_commit) as m_commit,
            mock.patch.object(ui_app, "connect_db", side_effect=sqlite3.OperationalError("disk full")),
        ):
            self.assertFalse(ui_app._on_training_completed(good_summary))
        m_prep.assert_called_once()
        m_commit.assert_not_called()
        self.assertEqual(ui_app.LIVE_CAMERA._onnx_model_version, "prev-active-v0")
        self.assertEqual(ui_app.LIVE_CAMERA._state["model_version"], "prev-active-v0")
        self.assertEqual(self.client.get("/api/models").json(), [])

        # When both prepare and DB registration succeed -> commits session swap and returns True
        with (
            mock.patch.object(ui_app.LIVE_CAMERA, "prepare_onnx_model", return_value=fake_prepared),
            mock.patch.object(ui_app.LIVE_CAMERA, "commit_prepared_onnx_model", side_effect=fake_commit),
        ):
            self.assertTrue(ui_app._on_training_completed(good_summary))
        models = self.client.get("/api/models").json()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["model_version"], "mnet_cb")
        self.assertEqual(ui_app.LIVE_CAMERA._onnx_model_version, "mnet_cb")

    # ------------------------------------------------------------------
    # R2-35: Partial evidence files removed on write/close failure; pre-existing untouched
    # ------------------------------------------------------------------
    def test_r2_35_exclusive_evidence_file_cleans_partial_write_and_preserves_existing(self) -> None:
        ev_dir = Path(self._temp_dir.name) / "evidence" / "R235"
        existing = ev_dir / "existing.jpg"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"ORIGINAL_DATA")

        # 1. Pre-existing file raises FileExistsError and remains untouched
        with self.assertRaises(FileExistsError):
            with ui_app._exclusive_evidence_file(existing, b"OVERWRITE_ATTEMPT"):
                pass
        self.assertEqual(existing.read_bytes(), b"ORIGINAL_DATA")

        # 2. Write failure after exclusive open removes partial file
        partial = ev_dir / "partial_write.jpg"
        real_open = Path.open

        class _FailingWriteStream:
            def __init__(self, inner):
                self._inner = inner

            def write(self, data):
                self._inner.write(data[:4])
                raise OSError("simulated disk full during write")

            def flush(self):
                self._inner.flush()

            def close(self):
                self._inner.close()

        def failing_open(self_path, mode="r", *args, **kwargs):
            handle = real_open(self_path, mode, *args, **kwargs)
            if self_path == partial and mode == "xb":
                return _FailingWriteStream(handle)
            return handle

        with mock.patch.object(Path, "open", failing_open):
            with self.assertRaises(OSError):
                with ui_app._exclusive_evidence_file(partial, b"PARTIAL_JPEG_DATA"):
                    pass
        self.assertFalse(partial.exists(), "Partial evidence file must be removed on write failure")

    # ------------------------------------------------------------------
    # R2-36: Non-scalar input_size in optional metrics.json keeps default 96
    # ------------------------------------------------------------------
    def test_r2_36_model_activate_handles_non_scalar_metrics_input_size(self) -> None:
        token = self._mint_via_api()
        runs_root = Path(self._temp_dir.name) / "runs" / "label_classifier"
        good_bytes = b"ONNX_R236_BYTES"
        good_sha = hashlib.sha256(good_bytes).hexdigest()
        run_dir = runs_root / "mnetv3_r236"
        run_dir.mkdir(parents=True)
        (run_dir / "label_classifier.onnx").write_bytes(good_bytes)
        (run_dir / "metrics.json").write_text(json.dumps({"input_size": [96]}), encoding="utf-8")

        reg = self.client.post(
            "/api/models",
            headers={"X-Admin-Token": token},
            json={
                "model_name": "mobilenetv3-small-igbt-roi",
                "model_version": "mnetv3_r236",
                "sha256": good_sha,
                "license_status": "reviewed",
                "notes": "non-scalar input_size test",
            },
        )
        self.assertEqual(reg.status_code, 200)
        model_id = reg.json()["model_id"]

        seen_sizes: list[int] = []

        def fake_load(onnx_path, *, model_version, sha256=None, input_size=96, expected_sha256=None):
            seen_sizes.append(input_size)
            return {"model_version": model_version, "sha256": sha256, "input_size": input_size}

        with mock.patch.object(ui_app.LIVE_CAMERA, "load_onnx_model", side_effect=fake_load):
            act = self.client.post(f"/api/models/{model_id}/activate", headers={"X-Admin-Token": token})
        self.assertEqual(act.status_code, 200, act.text)
        self.assertEqual(seen_sizes, [96])


if __name__ == "__main__":
    unittest.main()
