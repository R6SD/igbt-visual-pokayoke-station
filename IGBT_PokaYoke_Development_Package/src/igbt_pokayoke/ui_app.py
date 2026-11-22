"""Loopback-only FastAPI shadow dashboard and SQLite evidence store."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from igbt_pokayoke.live_camera import LiveCameraEngine
from igbt_pokayoke.runtime import SCENARIO_NAMES, ScenarioName, build_scenario_run
from igbt_pokayoke.training_manager import ALL_LABELS, TrainingManager

logger = logging.getLogger(__name__)



PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web" / "dist"
STATIC_ROOT = WEB_ROOT / "assets"
INDEX_HTML = WEB_ROOT / "index.html"
DATA_ROOT = Path(os.environ.get("IGBT_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
DB_PATH = DATA_ROOT / "station.sqlite3"
EVIDENCE_ROOT = DATA_ROOT / "evidence"
MAX_EVIDENCE_BYTES = 10 * 1024 * 1024
LABEL_STATES = {"complete", "crossed_out", "covered", "absent", "unknown"}
OUTCOMES = {"WAITING", "PASS", "ALARM", "FAULT", "VIEW_INVALID"}
FSM_STATES = {"BOOT", "WAIT_PART", "VERIFY_INITIAL", "READY", "INSPECTING", "PROCESSING",
              "VERIFY_C_ABSENT", "VIEW_INVALID", "PASS", "ALARM", "FAULT", "REVIEW_REQUIRED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def get_data_paths() -> tuple[Path, Path, Path]:
    env_dir = os.environ.get("IGBT_DATA_DIR")
    root = Path(env_dir).resolve() if env_dir else DATA_ROOT
    return root, root / "station.sqlite3", root / "evidence"


def default_config() -> dict:
    return {
        "config_version": 0,
        "roi_config_version": "ROI_UNCALIBRATED",
        "camera_device_id": "CAM_STATION_01",
        "camera_width": 1408,
        "camera_height": 768,
        "calibrated_at_utc": None,
        "camera_covers_both_ends": True,
        "rois": {},
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
        "mode": "shadow",
        "interlock_output_enabled": False,
    }


def _load_latest_config_safe() -> dict:
    try:
        with connect_db() as db:
            return _latest_config(db)
    except Exception:
        logger.warning("Failed to load latest station config, using default", exc_info=True)
        return default_config()


def _on_live_cycle_update(state_data: dict, state_before: str | None) -> None:
    try:
        initialize_database()
        cycle_id = state_data.get("cycle_id")
        now = utc_now()
        with connect_db() as db:
            db.execute(
                "UPDATE station_state SET updated_at_utc=?, state_json=? WHERE singleton_id=1",
                (now, json.dumps(state_data, ensure_ascii=False)),
            )
            if cycle_id:
                exists = db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
                if not exists:
                    db.execute(
                        "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            cycle_id,
                            now,
                            now,
                            state_data.get("label_status"),
                            state_data.get("outcome", "WAITING"),
                            state_data.get("result_reason"),
                            state_data.get("pipe_color"),
                            state_data.get("operator_anon_id"),
                            state_data.get("model_version"),
                            state_data.get("config_version"),
                            state_data.get("fsm_state", "WAIT_PART"),
                        ),
                    )
                else:
                    db.execute(
                        "UPDATE cycles SET updated_at_utc=?,final_outcome=?,result_reason=?,pipe_color=?,operator_anon_id=?,model_version=?,config_version=?,fsm_state=? WHERE cycle_id=?",
                        (
                            now,
                            state_data.get("outcome", "WAITING"),
                            state_data.get("result_reason"),
                            state_data.get("pipe_color"),
                            state_data.get("operator_anon_id"),
                            state_data.get("model_version"),
                            state_data.get("config_version"),
                            state_data.get("fsm_state", "WAIT_PART"),
                            cycle_id,
                        ),
                    )
                ev_id = _event(
                    db,
                    cycle_id,
                    state_data.get("result_reason") or "STATE_CHANGE",
                    state_data,
                    state_before,
                    state_data.get("fsm_state"),
                    state_data.get("outcome"),
                    state_data.get("result_reason"),
                    state_data.get("capture_monotonic_s"),
                    state_data.get("frame_id"),
                )
                if state_data.get("outcome") == "ALARM":
                    jpg = LIVE_CAMERA.get_hud_jpeg() or LIVE_CAMERA.get_raw_frame_jpeg()
                    if jpg:
                        evidence_id = uuid.uuid4().hex
                        sha = hashlib.sha256(jpg).hexdigest()
                        data_root, _, _ = get_data_paths()
                        relpath = Path("evidence") / cycle_id / f"{evidence_id}.jpg"
                        absolute = data_root / relpath
                        absolute.parent.mkdir(parents=True, exist_ok=True)
                        if not absolute.exists():
                            with absolute.open("xb") as stream:
                                stream.write(jpg)
                            db.execute(
                                "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                (
                                    evidence_id,
                                    cycle_id,
                                    ev_id,
                                    now,
                                    "anomaly",
                                    relpath.as_posix(),
                                    "image/jpeg",
                                    sha,
                                    state_data.get("frame_id") or 1,
                                    state_data.get("capture_monotonic_s"),
                                    json.dumps(
                                        {
                                            "pipe_color": state_data.get("pipe_color") or "unknown",
                                            "operation_event": state_data.get("result_reason") or "ALARM_AUTO_CAPTURE",
                                            "operator_anon_id": state_data.get("operator_anon_id") or "OP_STATION_01",
                                            "label_status": state_data.get("label_status"),
                                            "captured_at_utc": now,
                                        },
                                        ensure_ascii=False,
                                    ),
                                ),
                            )
    except Exception:
        logger.warning("Failed to persist live camera cycle update", exc_info=True)


LIVE_CAMERA = LiveCameraEngine(config_loader=_load_latest_config_safe, on_cycle_update=_on_live_cycle_update)
TRAINING_MANAGER = TrainingManager(data_root_getter=lambda: get_data_paths()[0])


def _autoload_latest_onnx_model() -> None:
    try:
        data_root, _, _ = get_data_paths()
        runs_dir = data_root / "runs" / "label_classifier"
        if not runs_dir.is_dir():
            return
        candidates = sorted(runs_dir.glob("*/label_classifier.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return
        latest_onnx = candidates[0]
        metrics_file = latest_onnx.parent / "metrics.json"
        version = latest_onnx.parent.name
        sha = hashlib.sha256(latest_onnx.read_bytes()).hexdigest()
        input_size = 96
        if metrics_file.is_file():
            meta = json.loads(metrics_file.read_text(encoding="utf-8"))
            input_size = int(meta.get("input_size") or 96)
        LIVE_CAMERA.load_onnx_model(latest_onnx, model_version=version, sha256=sha, input_size=input_size)
    except Exception:
        logger.warning("Failed to autoload latest ONNX model", exc_info=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    _autoload_latest_onnx_model()
    if os.environ.get("IGBT_ENABLE_LIVE_CAMERA", "0") == "1":
        LIVE_CAMERA.start()
    yield
    LIVE_CAMERA.stop()


app = FastAPI(title="IGBT Visual Poka-Yoke Shadow Station", version="0.1.0", lifespan=lifespan,
              docs_url="/api/docs", redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
app.mount("/assets", StaticFiles(directory=STATIC_ROOT, check_dir=False), name="assets")


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


def connect_db() -> sqlite3.Connection:
    data_root, db_path, _ = get_data_paths()
    data_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10, factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=10000")
    return connection


def initialize_database() -> None:
    data_root, _, evidence_root = get_data_paths()
    data_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    with connect_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS cycles (
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
        CREATE TABLE IF NOT EXISTS events (
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
            payload_json TEXT NOT NULL,
            FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id)
        );
        CREATE INDEX IF NOT EXISTS events_cycle_idx ON events(cycle_id, created_at_utc);
        CREATE TABLE IF NOT EXISTS evidence (
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
            metadata_json TEXT NOT NULL,
            FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id),
            FOREIGN KEY(event_id) REFERENCES events(event_id)
        );
        CREATE TABLE IF NOT EXISTS configuration_history (
            config_version INTEGER PRIMARY KEY,
            roi_config_version TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            changed_by TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            license_status TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            registered_by TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS station_state (
            singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
            updated_at_utc TEXT NOT NULL,
            state_json TEXT NOT NULL
        );
        """)
        if not db.execute("SELECT 1 FROM configuration_history LIMIT 1").fetchone():
            config = default_config()
            db.execute("INSERT INTO configuration_history VALUES(?,?,?,?,?)",
                       (0, config["roi_config_version"], utc_now(), "system", json.dumps(config)))
        if not db.execute("SELECT 1 FROM station_state WHERE singleton_id=1").fetchone():
            initial = default_station_state()
            db.execute("INSERT INTO station_state VALUES(1,?,?)", (utc_now(), json.dumps(initial)))


def default_station_state() -> dict:
    return {
        "mode": "shadow",
        "interlock_output_enabled": False,
        "camera_status": "NOT_CONNECTED",
        "camera_name": None,
        "camera_device_id": None,
        "backend": None,
        "resolution": None,
        "fps": None,
        "camera_settings_locked": False,
        "label_status": "unknown",
        "label_confidence": None,
        "view_valid": False,
        "view_valid_reasons": ["RUNTIME_NOT_CONNECTED"],
        "quality_ok": False,
        "laplacian_variance": None,
        "brightness": None,
        "overexposed_fraction": None,
        "underexposed_fraction": None,
        "left_grip_active": None,
        "right_grip_active": None,
        "label_contact_active": None,
        "hand_contact_candidate": False,
        "contact_evidence_type": "2d_visual_proximity_proxy",
        "proves_physical_contact": False,
        "fsm_state": "WAIT_PART",
        "outcome": "WAITING",
        "result_reason": "RUNTIME_NOT_CONNECTED",
        "c_timer_ms": 0,
        "cycle_id": None,
        "pipe_color": None,
        "operator_anon_id": None,
        "frame_id": None,
        "capture_monotonic_s": None,
        "model_version": None,
        "config_version": None,
        "mirror_horizontal": True,
        "updated_at_utc": utc_now(),
    }


class RuntimeSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["shadow"] = "shadow"
    camera_status: Literal["ONLINE", "OFFLINE", "FAULT", "UNVERIFIED"]
    camera_name: str | None = None
    camera_device_id: str | None = None
    backend: str | None = None
    camera_width: int | None = Field(default=None, ge=1, le=16384)
    camera_height: int | None = Field(default=None, ge=1, le=16384)
    camera_settings_locked: bool | None = None
    fps: float | None = Field(default=None, ge=0, le=1000)
    label_status: Literal["complete", "crossed_out", "covered", "absent", "unknown"] = "unknown"
    label_confidence: float | None = Field(default=None, ge=0, le=1)
    view_valid: bool | None = None
    view_valid_reasons: list[str] = Field(default_factory=list)
    quality_ok: bool | None = None
    laplacian_variance: float | None = Field(default=None, ge=0)
    brightness: float | None = Field(default=None, ge=0, le=255)
    overexposed_fraction: float | None = Field(default=None, ge=0, le=1)
    underexposed_fraction: float | None = Field(default=None, ge=0, le=1)
    left_grip_active: bool | None = None
    right_grip_active: bool | None = None
    label_contact_active: bool | None = None
    hand_contact_candidate: bool = False
    contact_evidence_type: Literal["2d_visual_proximity_proxy"] = "2d_visual_proximity_proxy"
    proves_physical_contact: Literal[False] = False
    fsm_state: Literal["BOOT", "WAIT_PART", "VERIFY_INITIAL", "READY", "INSPECTING", "PROCESSING",
                       "VERIFY_C_ABSENT", "VIEW_INVALID", "PASS", "ALARM", "FAULT", "REVIEW_REQUIRED"] = "WAIT_PART"
    outcome: Literal["WAITING", "PASS", "ALARM", "FAULT", "VIEW_INVALID"] = "WAITING"
    result_reason: str | None = None
    c_timer_ms: int = Field(default=0, ge=0, le=1200)
    cycle_id: str | None = None
    pipe_color: Literal["yellow", "blue", "other", "unknown"] | None = None
    operator_anon_id: str | None = None
    event_type: str | None = None
    frame_id: int | None = Field(default=None, ge=0)
    capture_monotonic_s: float | None = Field(default=None, ge=0)
    model_version: str | None = None
    config_version: str | None = None


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_device_id: str = Field(min_length=1, max_length=256)
    camera_width: int = Field(ge=1, le=16384)
    camera_height: int = Field(ge=1, le=16384)
    camera_covers_both_ends: bool = True
    rois: dict[str, list[float]]
    thresholds: dict[str, float | None]


class ModelRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]{1,64}$")
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    license_status: Literal["reviewed", "needs_review", "unknown"] = "unknown"
    notes: str = Field(default="", max_length=2000)

    @field_validator("model_version")
    @classmethod
    def _reject_dotdot(cls, value: str) -> str:
        if ".." in value or value in {".", ".."}:
            raise ValueError("model_version must not contain path traversal segments")
        return value


class AuthVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passcode: str = Field(max_length=256)


DEFAULT_ENGINEER_PIN = "888888"
ENGINEER_SESSION_TTL_SECONDS = 1800


def _compare_secret(supplied: str, expected: str) -> bool:
    if not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def admin_or_station_token(supplied: str | None, env_key: str) -> str:
    expected = os.environ.get(env_key)
    if not expected:
        raise HTTPException(503, f"{env_key}_NOT_CONFIGURED")
    if supplied is None or not _compare_secret(supplied, expected):
        raise HTTPException(401, "INVALID_TOKEN")
    return "local-admin" if env_key == "IGBT_ADMIN_TOKEN" else "station-runtime"


def _latest_config(db: sqlite3.Connection) -> dict:
    row = db.execute("SELECT config_json FROM configuration_history ORDER BY config_version DESC LIMIT 1").fetchone()
    return json.loads(row[0]) if row else default_config()


def _safe_cycle_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise HTTPException(400, "INVALID_CYCLE_ID")
    return value


def _event(db: sqlite3.Connection, cycle_id: str | None, event_type: str, payload: dict,
           state_before: str | None = None, state_after: str | None = None,
           outcome: str | None = None, reason: str | None = None,
           capture_monotonic_s: float | None = None, frame_id: int | None = None) -> str:
    event_id = uuid.uuid4().hex
    db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
        event_id, cycle_id, utc_now(), capture_monotonic_s, frame_id, event_type,
        state_before, state_after, outcome, reason, json.dumps(payload, ensure_ascii=False),
    ))
    return event_id


def _validate_config(payload: ConfigUpdate) -> dict:
    required = {"tube_roi", "label_roi", "label_contact_roi"}
    if payload.camera_covers_both_ends:
        required |= {"left_grip_zone", "right_grip_zone"}
    if not required.issubset(payload.rois):
        raise HTTPException(422, {"missing_rois": sorted(required - set(payload.rois))})
    for name, box in payload.rois.items():
        if name not in required | {"quality_roi"} or len(box) != 4:
            raise HTTPException(422, f"INVALID_ROI:{name}")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise HTTPException(422, f"ROI_OUT_OF_NORMALIZED_BOUNDS:{name}")
    if payload.camera_covers_both_ends:
        c = payload.rois["label_contact_roi"]
        for name in ("left_grip_zone", "right_grip_zone"):
            g = payload.rois[name]
            iw, ih = max(0, min(c[2], g[2]) - max(c[0], g[0])), max(0, min(c[3], g[3]) - max(c[1], g[1]))
            area = (c[2] - c[0]) * (c[3] - c[1])
            if iw * ih / area > 0.01:
                raise HTTPException(422, "CONTACT_ROI_OVERLAPS_GRIP_ZONE")
    thresholds = dict(payload.thresholds)
    for key in ("hand_glove_confidence", "label_min_confidence", "label_min_margin"):
        value = thresholds.get(key)
        if value is not None and not 0 <= value <= 1:
            raise HTTPException(422, f"THRESHOLD_OUT_OF_RANGE:{key}")
    for key in ("stable_state_ms", "min_stable_frames", "min_laplacian_variance", "min_brightness", "max_brightness", "max_overexposed_fraction"):
        value = thresholds.get(key)
        if value is not None and value < 0:
            raise HTTPException(422, f"THRESHOLD_MUST_BE_NONNEGATIVE:{key}")
    for key in ("min_brightness", "max_brightness"):
        value = thresholds.get(key)
        if value is not None and value > 255:
            raise HTTPException(422, f"BRIGHTNESS_THRESHOLD_OUT_OF_RANGE:{key}")
    if thresholds.get("max_overexposed_fraction") is not None and thresholds["max_overexposed_fraction"] > 1:
        raise HTTPException(422, "OVEREXPOSED_FRACTION_MUST_BE_AT_MOST_ONE")
    if (
        thresholds.get("min_brightness") is not None
        and thresholds.get("max_brightness") is not None
        and thresholds["min_brightness"] >= thresholds["max_brightness"]
    ):
        raise HTTPException(422, "MIN_BRIGHTNESS_MUST_BE_LOWER_THAN_MAX")
    return {
        "camera_device_id": payload.camera_device_id,
        "camera_width": payload.camera_width,
        "camera_height": payload.camera_height,
        "camera_covers_both_ends": payload.camera_covers_both_ends,
        "rois": payload.rois,
        "thresholds": thresholds,
        "mode": "shadow",
        "interlock_output_enabled": False,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    if not INDEX_HTML.is_file():
        raise HTTPException(503, "FRONTEND_BUILD_MISSING: run npm ci and npm run build in frontend/igbt-station-ui")
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "host_policy": "loopback_only", "mode": "shadow", "interlock_output_enabled": False}


@app.get("/api/auth/status")
def auth_status():
    configured_pin = os.environ.get("IGBT_ENGINEER_PIN")
    default_pin_active = (not configured_pin) or _compare_secret(configured_pin, DEFAULT_ENGINEER_PIN)
    return {
        "role_gate_enabled": True,
        "default_pin_active": default_pin_active,
    }


@app.post("/api/auth/verify")
def verify_engineer_auth(payload: AuthVerifyRequest):
    supplied = payload.passcode
    configured_admin_token = os.environ.get("IGBT_ADMIN_TOKEN") or ""
    configured_pin = os.environ.get("IGBT_ENGINEER_PIN")
    engineer_pin = configured_pin if configured_pin else DEFAULT_ENGINEER_PIN

    matches_admin = _compare_secret(supplied, configured_admin_token)
    matches_pin = _compare_secret(supplied, engineer_pin)
    if not (matches_admin or matches_pin):
        raise HTTPException(401, "工程师口令错误，请重试")

    expected_admin_token = configured_admin_token or "local-admin-token"
    return {
        "authenticated": True,
        "role": "engineer",
        "admin_token": expected_admin_token,
        "expires_in_seconds": ENGINEER_SESSION_TTL_SECONDS,
    }


def _attach_latest_evidence(db: sqlite3.Connection, state: dict) -> dict:
    ev = db.execute("SELECT evidence_id, kind FROM evidence ORDER BY created_at_utc DESC LIMIT 1").fetchone()
    if ev:
        state["latest_evidence_url"] = f"/api/evidence/{ev['evidence_id']}/file"
        state["latest_evidence_kind"] = ev["kind"]
    else:
        state.pop("latest_evidence_url", None)
        state.pop("latest_evidence_kind", None)
    return state


@app.get("/api/state")
def get_state():
    initialize_database()
    with connect_db() as db:
        if LIVE_CAMERA.camera_online and time.monotonic() >= LIVE_CAMERA.simulation_override_until:
            state = LIVE_CAMERA.get_state()
            return _attach_latest_evidence(db, state)
        row = db.execute("SELECT state_json FROM station_state WHERE singleton_id=1").fetchone()
        state = json.loads(row[0]) if row else default_station_state()
        state["mirror_horizontal"] = LIVE_CAMERA.mirror_horizontal
        if LIVE_CAMERA.camera_online:
            state["live_stream_url"] = "/api/camera/stream"
            state["live_crop_url"] = "/api/camera/stream_crop"
        return _attach_latest_evidence(db, state)


@app.get("/api/camera/stream")
def camera_mjpeg_stream():
    def gen():
        while True:
            jpg = LIVE_CAMERA.get_hud_jpeg()
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.033)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/stream_crop")
def camera_crop_mjpeg_stream():
    def gen():
        while True:
            jpg = LIVE_CAMERA.get_crop_jpeg()
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.04)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/frame.jpg")
def camera_raw_frame():
    jpg = LIVE_CAMERA.get_raw_frame_jpeg()
    if not jpg:
        raise HTTPException(503, "LIVE_CAMERA_FRAME_UNAVAILABLE")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/camera/capture")
def capture_live_camera_frame():
    initialize_database()
    jpg = LIVE_CAMERA.get_raw_frame_jpeg()
    if not jpg:
        raise HTTPException(503, "当前未获取到实时摄像头画面")
    live_state = LIVE_CAMERA.get_state()
    cycle_id = _safe_cycle_id(live_state.get("cycle_id") or f"LIVE_{datetime.now().strftime('%H%M%S')}")
    now = utc_now()
    evidence_id = uuid.uuid4().hex
    sha = hashlib.sha256(jpg).hexdigest()
    data_root, _, _ = get_data_paths()
    relpath = Path("evidence") / cycle_id / f"{evidence_id}.jpg"
    absolute = data_root / relpath
    absolute.parent.mkdir(parents=True, exist_ok=True)
    with absolute.open("xb") as stream:
        stream.write(jpg)

    metadata = {
        "pipe_color": live_state.get("pipe_color") or "yellow",
        "operation_event": "LIVE_CAMERA_MANUAL_CAPTURE",
        "operator_anon_id": live_state.get("operator_anon_id") or "OP_STATION_01",
        "camera_metadata": {
            "camera_name": live_state.get("camera_name"),
            "camera_device_id": live_state.get("camera_device_id"),
            "resolution": f"{live_state.get('camera_width')}x{live_state.get('camera_height')}",
            "fps": live_state.get("fps"),
            "laplacian_variance": live_state.get("laplacian_variance"),
            "brightness": live_state.get("brightness"),
        },
        "label_status": live_state.get("label_status"),
        "captured_at_utc": now,
    }
    with connect_db() as db:
        if not db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone():
            db.execute(
                "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle_id,
                    now,
                    now,
                    live_state.get("label_status"),
                    live_state.get("outcome") or "WAITING",
                    live_state.get("result_reason"),
                    live_state.get("pipe_color"),
                    live_state.get("operator_anon_id"),
                    live_state.get("model_version"),
                    live_state.get("config_version"),
                    live_state.get("fsm_state") or "INSPECTING",
                ),
            )
        ev_id = _event(
            db,
            cycle_id,
            "LIVE_CAPTURE",
            metadata,
            state_before=live_state.get("fsm_state"),
            state_after=live_state.get("fsm_state"),
            outcome=live_state.get("outcome"),
            reason="手动抓拍实时相机画面",
            capture_monotonic_s=live_state.get("capture_monotonic_s"),
            frame_id=live_state.get("frame_id"),
        )
        db.execute(
            "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                evidence_id,
                cycle_id,
                ev_id,
                now,
                "initial_label",
                relpath.as_posix(),
                "image/jpeg",
                sha,
                live_state.get("frame_id") or 1,
                live_state.get("capture_monotonic_s"),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
    return {
        "evidence_id": evidence_id,
        "cycle_id": cycle_id,
        "sha256": sha,
        "url": f"/api/evidence/{evidence_id}/file",
        "captured_at_utc": now,
    }


@app.post("/api/runtime/simulate/{scenario}")
def simulate_runtime_scenario(scenario: ScenarioName):
    if scenario not in SCENARIO_NAMES:
        raise HTTPException(422, f"UNSUPPORTED_SCENARIO:{scenario}")
    initialize_database()
    if LIVE_CAMERA.camera_online:
        if scenario == "reset":
            st = LIVE_CAMERA.trigger_new_cycle()
            with connect_db() as db:
                return _attach_latest_evidence(db, st)
        LIVE_CAMERA.pause_for_simulation(10.0)
    run = build_scenario_run(scenario)
    validated = RuntimeSnapshot(**run["snapshot"])
    data = validated.model_dump()
    data["mode"] = "shadow"
    data["interlock_output_enabled"] = False
    if data["outcome"] == "PASS" and (
        data["view_valid"] is not True or data["quality_ok"] is not True or
        data["camera_status"] != "ONLINE" or data["label_status"] == "unknown" or data["fsm_state"] != "PASS"
    ):
        data["outcome"] = "VIEW_INVALID" if data["view_valid"] is not True else "ALARM"
        data["result_reason"] = "SHADOW_PASS_REJECTED_MISSING_VALID_EVIDENCE"

    transitions = run["transitions"]
    cycle_id = data.get("cycle_id")
    if not cycle_id:
        for tr in transitions:
            if tr.get("cycle_id"):
                cycle_id = tr["cycle_id"]
                break
    cycle_id = _safe_cycle_id(cycle_id or f"CYC_SIM_{uuid.uuid4().hex[:6].upper()}")

    now_dt = datetime.now(timezone.utc)
    elapsed_s = 1.8
    if len(transitions) >= 2:
        t0 = transitions[0].get("now_monotonic_s")
        t1 = transitions[-1].get("now_monotonic_s")
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and t1 > t0:
            elapsed_s = max(0.8, min(30.0, float(t1 - t0)))
    created_utc = (now_dt - timedelta(seconds=elapsed_s)).isoformat(timespec="milliseconds")
    updated_utc = now_dt.isoformat(timespec="milliseconds")
    data["updated_at_utc"] = updated_utc

    ref_path = Path(run["reference_image_path"])
    body = ref_path.read_bytes()
    sha = hashlib.sha256(body).hexdigest()
    kind = run["evidence_kind"]
    pipe_color = run["pipe_color"]
    data_root, _, _ = get_data_paths()

    with connect_db() as db:
        initial_label = "absent" if scenario == "alarm" else "complete"
        exists = db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
        if not exists:
            db.execute(
                "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle_id,
                    created_utc,
                    updated_utc,
                    initial_label,
                    data["outcome"],
                    data.get("result_reason"),
                    pipe_color,
                    data.get("operator_anon_id"),
                    data.get("model_version"),
                    data.get("config_version"),
                    data.get("fsm_state", "WAIT_PART"),
                ),
            )
        else:
            db.execute(
                "UPDATE cycles SET updated_at_utc=?,final_outcome=?,result_reason=?,pipe_color=?,operator_anon_id=?,model_version=?,config_version=?,fsm_state=? WHERE cycle_id=?",
                (
                    updated_utc,
                    data["outcome"],
                    data.get("result_reason"),
                    pipe_color,
                    data.get("operator_anon_id"),
                    data.get("model_version"),
                    data.get("config_version"),
                    data.get("fsm_state", "WAIT_PART"),
                    cycle_id,
                ),
            )

        state_transitions = [tr for tr in transitions if tr.get("record_type") == "STATE_TRANSITION"]
        last_event_id: str | None = None
        for idx, tr in enumerate(state_transitions):
            ev_time = (now_dt - timedelta(seconds=max(0.01, elapsed_s * (1.0 - (idx + 1) / max(1, len(state_transitions)))))).isoformat(timespec="milliseconds")
            ev_id = uuid.uuid4().hex
            ev_type = str(tr.get("reason") or tr.get("event", {}).get("kind", "STATE_TRANSITION"))
            db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ev_id,
                    cycle_id,
                    ev_time,
                    tr.get("capture_monotonic_s"),
                    data.get("frame_id"),
                    ev_type,
                    tr.get("state_before"),
                    tr.get("state_after"),
                    tr.get("outcome"),
                    tr.get("reason"),
                    json.dumps(tr, ensure_ascii=False),
                ),
            )
            last_event_id = ev_id

        existing_ev = db.execute(
            "SELECT evidence_id FROM evidence WHERE cycle_id=? AND sha256=? LIMIT 1",
            (cycle_id, sha),
        ).fetchone()
        if not existing_ev:
            evidence_id = uuid.uuid4().hex
            relpath = Path("evidence") / cycle_id / f"{evidence_id}.jpg"
            absolute = data_root / relpath
            absolute.parent.mkdir(parents=True, exist_ok=True)
            with absolute.open("xb") as stream:
                stream.write(body)
            metadata = {
                "pipe_color": pipe_color,
                "operation_event": f"SCENARIO_{scenario.upper()}",
                "operator_anon_id": data.get("operator_anon_id"),
                "camera_metadata": {
                    "camera_device_id": data.get("camera_device_id"),
                    "resolution": f"{data.get('camera_width')}x{data.get('camera_height')}",
                    "fps": data.get("fps"),
                },
                "label_status": data.get("label_status"),
                "captured_at_utc": updated_utc,
            }
            if last_event_id is None:
                last_event_id = _event(
                    db,
                    cycle_id,
                    f"EVIDENCE_{kind}",
                    metadata,
                    state_before="WAIT_PART",
                    state_after=data.get("fsm_state"),
                    outcome=data.get("outcome"),
                    reason=data.get("result_reason"),
                    capture_monotonic_s=data.get("capture_monotonic_s"),
                    frame_id=data.get("frame_id"),
                )
            db.execute(
                "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    evidence_id,
                    cycle_id,
                    last_event_id,
                    updated_utc,
                    kind,
                    relpath.as_posix(),
                    "image/jpeg",
                    sha,
                    data.get("frame_id") or 1,
                    data.get("capture_monotonic_s"),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )

        db.execute(
            "UPDATE station_state SET updated_at_utc=?, state_json=? WHERE singleton_id=1",
            (updated_utc, json.dumps(data, ensure_ascii=False)),
        )
        return _attach_latest_evidence(db, data)


@app.post("/api/runtime/snapshot")
def runtime_snapshot(snapshot: RuntimeSnapshot, x_station_token: str | None = Header(default=None)):
    admin_or_station_token(x_station_token, "IGBT_STATION_TOKEN")
    data = snapshot.model_dump()
    data["mode"] = "shadow"
    data["interlock_output_enabled"] = False
    # Never display an affirmative result when the mandatory validity gates fail.
    if data["outcome"] == "PASS" and (
        data["view_valid"] is not True or data["quality_ok"] is not True or
        data["camera_status"] != "ONLINE" or data["label_status"] == "unknown" or data["fsm_state"] != "PASS"
    ):
        data["outcome"] = "VIEW_INVALID" if data["view_valid"] is not True else "ALARM"
        data["result_reason"] = "SHADOW_PASS_REJECTED_MISSING_VALID_EVIDENCE"
    if data["view_valid"] is not True and data["outcome"] == "WAITING":
        data["outcome"] = "VIEW_INVALID"
        data["result_reason"] = data["result_reason"] or "VIEW_INVALID"
    data["updated_at_utc"] = utc_now()
    before = None
    with connect_db() as db:
        old = db.execute("SELECT state_json FROM station_state WHERE singleton_id=1").fetchone()
        previous = json.loads(old[0]) if old else default_station_state()
        before = previous.get("fsm_state")
        db.execute("UPDATE station_state SET updated_at_utc=?, state_json=? WHERE singleton_id=1",
                   (data["updated_at_utc"], json.dumps(data, ensure_ascii=False)))
        cycle_id = data.get("cycle_id")
        if cycle_id:
            cycle_id = _safe_cycle_id(cycle_id)
            exists = db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
            now = utc_now()
            if not exists:
                db.execute("INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (cycle_id, now, now, data.get("label_status"), data["outcome"], data.get("result_reason"), data.get("pipe_color"), data.get("operator_anon_id"), data.get("model_version"), data.get("config_version"), data.get("fsm_state", "WAIT_PART")))
            else:
                db.execute("UPDATE cycles SET updated_at_utc=?,final_outcome=?,result_reason=?,pipe_color=?,operator_anon_id=?,model_version=?,config_version=?,fsm_state=? WHERE cycle_id=?",
                           (now, data["outcome"], data.get("result_reason"), data.get("pipe_color"), data.get("operator_anon_id"), data.get("model_version"), data.get("config_version"), data.get("fsm_state", "WAIT_PART"), cycle_id))
            changed = any(previous.get(k) != data.get(k) for k in ("fsm_state", "outcome", "label_status", "view_valid", "hand_contact_candidate", "result_reason"))
            if changed or data.get("event_type"):
                event_type = data.get("event_type") or "STATE_CHANGE"
                _event(db, cycle_id, event_type, data, before, data.get("fsm_state"), data.get("outcome"),
                       data.get("result_reason"), data.get("capture_monotonic_s"), data.get("frame_id"))
    return data


@app.get("/api/config")
def get_config():
    initialize_database()
    with connect_db() as db:
        return _latest_config(db)


@app.put("/api/config")
def update_config(payload: ConfigUpdate, x_admin_token: str | None = Header(default=None)):
    actor = admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    config = _validate_config(payload)
    calibrated = utc_now()
    with connect_db() as db:
        latest = _latest_config(db)
        version = int(latest.get("config_version", 0)) + 1
        config.update({"config_version": version, "roi_config_version": f"ROI_V{version:03d}", "calibrated_at_utc": calibrated})
        db.execute("INSERT INTO configuration_history VALUES(?,?,?,?,?)",
                   (version, config["roi_config_version"], calibrated, actor, json.dumps(config, ensure_ascii=False)))
    return config


@app.get("/api/config/history")
def config_history(limit: int = 50):
    with connect_db() as db:
        rows = db.execute("SELECT config_version,roi_config_version,created_at_utc,changed_by,config_json FROM configuration_history ORDER BY config_version DESC LIMIT ?",
                           (max(1, min(limit, 200)),)).fetchall()
    return [{"config_version": r[0], "roi_config_version": r[1], "created_at_utc": r[2],
             "changed_by": r[3], "config": json.loads(r[4])} for r in rows]


def _dir_usage(directory: Path) -> tuple[int, int]:
    if not directory.is_dir():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for item in directory.rglob("*"):
        try:
            if item.is_file():
                file_count += 1
                total_bytes += item.stat().st_size
        except OSError:
            continue
    return file_count, total_bytes


def _safe_unlink_evidence(data_root: Path, evidence_root: Path, file_relpath: str) -> bool:
    ev_root_resolved = evidence_root.resolve()
    try:
        candidate = (data_root / file_relpath).resolve()
    except OSError:
        return False
    if ev_root_resolved not in candidate.parents:
        logger.warning("Refused to delete evidence path outside evidence_root: %s", file_relpath)
        return False
    if candidate.is_file():
        candidate.unlink(missing_ok=True)
        parent = candidate.parent
        if parent != ev_root_resolved and ev_root_resolved in parent.parents:
            with suppress(OSError):
                parent.rmdir()
        return True
    return False


def _safe_clean_evidence_dir(evidence_root: Path) -> int:
    ev_root_resolved = evidence_root.resolve()
    if not ev_root_resolved.is_dir():
        return 0
    removed_files = 0
    for item in sorted(ev_root_resolved.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if ev_root_resolved not in resolved.parents:
            continue
        if resolved.is_file():
            with suppress(OSError):
                resolved.unlink(missing_ok=True)
                removed_files += 1
        elif resolved.is_dir():
            with suppress(OSError):
                resolved.rmdir()
    return removed_files


def _safe_model_run_dir(data_root: Path, version: str) -> Path | None:
    if (
        not version
        or ".." in version
        or version in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", version)
    ):
        return None
    runs_root = (data_root / "runs" / "label_classifier").resolve()
    candidate = (runs_root / version).resolve()
    if runs_root not in candidate.parents or candidate.parent != runs_root:
        return None
    return candidate


def _clean_optional_header(value: str | None, max_len: int = 64) -> str | None:
    if value is None:
        return None
    decoded = unquote(value).replace("\r", "").replace("\n", "").replace("\t", "").strip()
    if not decoded:
        return None
    return decoded[:max_len]


@app.get("/api/storage/overview")
def storage_overview():
    initialize_database()
    data_root, db_path, evidence_root = get_data_paths()
    db_size_bytes = db_path.stat().st_size if db_path.is_file() else 0
    ev_files, ev_bytes = _dir_usage(evidence_root)
    with connect_db() as db:
        cycles_cnt = int(db.execute("SELECT COUNT(*) FROM cycles").fetchone()[0])
        events_cnt = int(db.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        evidence_cnt = int(db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
        config_cnt = int(db.execute("SELECT COUNT(*) FROM configuration_history").fetchone()[0])
        model_cnt = int(db.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0])

    ds_stats = TRAINING_MANAGER.storage_stats()
    with LIVE_CAMERA._lock:
        active_model_version = LIVE_CAMERA._onnx_model_version

    return {
        "business_db": {
            "db_path": str(db_path),
            "db_size_kb": round(db_size_bytes / 1024.0, 2),
            "evidence_dir": str(evidence_root),
            "evidence_files": ev_files,
            "evidence_size_mb": round(ev_bytes / (1024 * 1024), 3),
            "tables": {
                "cycles": cycles_cnt,
                "events": events_cnt,
                "evidence": evidence_cnt,
                "configuration_history": config_cnt,
                "model_registry": model_cnt,
            },
        },
        "training_dataset": {
            "dataset_dir": ds_stats["dataset_dir"],
            "manifest_path": ds_stats["manifest_path"],
            "total_samples": ds_stats["total_samples"],
            "class_counts": ds_stats["class_counts"],
            "independent_tube_groups": ds_stats["independent_tube_groups"],
            "coco_annotations": ds_stats["coco_annotations"],
            "dataset_size_mb": ds_stats["dataset_size_mb"],
        },
        "model_artifacts": {
            "runs_dir": ds_stats["runs_dir"],
            "run_count": ds_stats["run_count"],
            "models_size_mb": ds_stats["models_size_mb"],
            "active_model_version": active_model_version,
        },
    }


@app.post("/api/models")
def register_model(payload: ModelRegistration, x_admin_token: str | None = Header(default=None)):
    actor = admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    model_id = uuid.uuid4().hex
    with connect_db() as db:
        db.execute("INSERT INTO model_registry VALUES(?,?,?,?,?,?,?,?)",
                   (model_id, payload.model_name, payload.model_version, payload.sha256.lower(), payload.license_status,
                    payload.notes, utc_now(), actor))
    return {"model_id": model_id, **payload.model_dump()}


@app.get("/api/models")
def list_models(limit: int = 50):
    with connect_db() as db:
        rows = db.execute("SELECT model_id,model_name,model_version,sha256,license_status,notes,created_at_utc,registered_by FROM model_registry ORDER BY created_at_utc DESC LIMIT ?",
                           (max(1, min(limit, 200)),)).fetchall()
    return [dict(r) for r in rows]


@app.delete("/api/models/{model_id}")
def delete_registered_model(model_id: str, x_admin_token: str | None = Header(default=None)):
    admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", model_id):
        raise HTTPException(404, "MODEL_NOT_FOUND")
    initialize_database()
    with connect_db() as db:
        row = db.execute(
            "SELECT model_id, model_name, model_version FROM model_registry WHERE model_id=?",
            (model_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "MODEL_NOT_FOUND")

    version = str(row["model_version"])
    with LIVE_CAMERA._lock:
        active_version = LIVE_CAMERA._onnx_model_version
    if version == active_version:
        raise HTTPException(409, "不能删除当前正在实时运行的模型，请先切换其他模型")

    with connect_db() as db:
        db.execute("DELETE FROM model_registry WHERE model_id=?", (model_id,))
        remaining_same_version = int(
            db.execute("SELECT COUNT(*) FROM model_registry WHERE model_version=?", (version,)).fetchone()[0]
        )

    deleted_artifact_dir = False
    if remaining_same_version == 0:
        data_root, _, _ = get_data_paths()
        run_dir = _safe_model_run_dir(data_root, version)
        if run_dir is not None and run_dir.is_dir():
            shutil.rmtree(run_dir)
            deleted_artifact_dir = True

    return {
        "deleted": True,
        "model_id": model_id,
        "model_version": version,
        "deleted_artifact_dir": deleted_artifact_dir,
    }


class PurgeCyclesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["simulated_only", "all"]
    confirm: Literal[True]


@app.get("/api/cycles")
def list_cycles(limit: int = 100):
    with connect_db() as db:
        rows = db.execute("SELECT * FROM cycles ORDER BY updated_at_utc DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/cycles/purge")
def purge_cycles(payload: PurgeCyclesRequest, x_admin_token: str | None = Header(default=None)):
    admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    initialize_database()
    data_root, _, evidence_root = get_data_paths()

    if payload.scope == "simulated_only":
        with connect_db() as db:
            ev_rows = db.execute("SELECT file_relpath FROM evidence WHERE cycle_id LIKE 'CYC\\_%' ESCAPE '\\'").fetchall()
            deleted_evidence = db.execute("DELETE FROM evidence WHERE cycle_id LIKE 'CYC\\_%' ESCAPE '\\'").rowcount
            deleted_events = db.execute("DELETE FROM events WHERE cycle_id LIKE 'CYC\\_%' ESCAPE '\\'").rowcount
            deleted_cycles = db.execute("DELETE FROM cycles WHERE cycle_id LIKE 'CYC\\_%' ESCAPE '\\'").rowcount
        deleted_files = sum(
            1 for r in ev_rows if _safe_unlink_evidence(data_root, evidence_root, r["file_relpath"])
        )
        vacuumed = False
    else:
        with connect_db() as db:
            ev_rows = db.execute("SELECT file_relpath FROM evidence").fetchall()
            deleted_evidence = db.execute("DELETE FROM evidence").rowcount
            deleted_events = db.execute("DELETE FROM events").rowcount
            deleted_cycles = db.execute("DELETE FROM cycles").rowcount
        deleted_files = sum(
            1 for r in ev_rows if _safe_unlink_evidence(data_root, evidence_root, r["file_relpath"])
        )
        deleted_files += _safe_clean_evidence_dir(evidence_root)
        with connect_db() as vac_db:
            vac_db.isolation_level = None
            vac_db.execute("VACUUM")
        vacuumed = True

    return {
        "purged": True,
        "scope": payload.scope,
        "deleted_cycles": deleted_cycles,
        "deleted_events": deleted_events,
        "deleted_evidence": deleted_evidence,
        "deleted_files": deleted_files,
        "freed_files": deleted_files,
        "vacuumed": vacuumed,
    }


@app.delete("/api/cycles/{cycle_id}")
def delete_cycle(cycle_id: str, x_admin_token: str | None = Header(default=None)):
    admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    cycle_id = _safe_cycle_id(cycle_id)
    initialize_database()
    data_root, _, evidence_root = get_data_paths()

    with connect_db() as db:
        exists = db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
        if not exists:
            raise HTTPException(404, "CYCLE_NOT_FOUND")
        ev_rows = db.execute("SELECT file_relpath FROM evidence WHERE cycle_id=?", (cycle_id,)).fetchall()
        deleted_evidence = db.execute("DELETE FROM evidence WHERE cycle_id=?", (cycle_id,)).rowcount
        deleted_events = db.execute("DELETE FROM events WHERE cycle_id=?", (cycle_id,)).rowcount
        deleted_cycles = db.execute("DELETE FROM cycles WHERE cycle_id=?", (cycle_id,)).rowcount

    deleted_files = sum(
        1 for r in ev_rows if _safe_unlink_evidence(data_root, evidence_root, r["file_relpath"])
    )
    cycle_ev_dir = (evidence_root / cycle_id).resolve()
    if evidence_root.resolve() in cycle_ev_dir.parents and cycle_ev_dir.is_dir():
        with suppress(OSError):
            cycle_ev_dir.rmdir()

    return {
        "deleted": True,
        "cycle_id": cycle_id,
        "deleted_cycles": deleted_cycles,
        "deleted_events": deleted_events,
        "deleted_evidence": deleted_evidence,
        "deleted_files": deleted_files,
        "freed_files": deleted_files,
    }


@app.get("/api/cycles/{cycle_id}/events")
def cycle_events(cycle_id: str, limit: int = 200):
    cycle_id = _safe_cycle_id(cycle_id)
    with connect_db() as db:
        rows = db.execute("SELECT * FROM events WHERE cycle_id=? ORDER BY created_at_utc DESC LIMIT ?",
                           (cycle_id, max(1, min(limit, 1000)))).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload_json"])} for r in rows]


@app.post("/api/cycles/{cycle_id}/evidence/{kind}")
async def upload_evidence(cycle_id: str, kind: Literal["initial_label", "A", "B", "C", "anomaly"],
                          request: Request, x_frame_id: int = Header(ge=0),
                          x_capture_monotonic_s: float | None = Header(default=None),
                          x_pipe_color: str | None = Header(default=None),
                          x_operation_event: str | None = Header(default=None),
                          x_operator_anon_id: str | None = Header(default=None),
                          x_camera_metadata: str = Header(min_length=2),
                          x_label_status: str | None = Header(default=None),
                          x_captured_at_utc: str = Header(min_length=20, max_length=40)):
    cycle_id = _safe_cycle_id(cycle_id)
    if x_pipe_color not in {None, "yellow", "blue", "other", "unknown"}:
        raise HTTPException(422, "INVALID_PIPE_COLOR")
    if x_label_status not in {None, *LABEL_STATES}:
        raise HTTPException(422, "INVALID_LABEL_STATUS")
    try:
        captured = datetime.fromisoformat(x_captured_at_utc.replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError("timezone required")
        camera_metadata = json.loads(x_camera_metadata)
        if not isinstance(camera_metadata, dict) or not camera_metadata:
            raise ValueError("camera metadata must be a non-empty object")
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA") from exc
    content_length = request.headers.get("content-length")
    if not content_length or not content_length.isdigit() or int(content_length) > MAX_EVIDENCE_BYTES:
        raise HTTPException(413, "EVIDENCE_SIZE_LIMIT_OR_LENGTH_REQUIRED")
    body = await request.body()
    if not body or len(body) != int(content_length) or len(body) > MAX_EVIDENCE_BYTES:
        raise HTTPException(400, "INVALID_EVIDENCE_BODY")
    mime = request.headers.get("content-type", "").split(";")[0].lower()
    if mime == "image/jpeg" and body.startswith(b"\xff\xd8") and body.endswith(b"\xff\xd9"):
        suffix = ".jpg"
    elif mime == "image/png" and body.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = ".png"
    else:
        raise HTTPException(415, "ONLY_VALID_JPEG_OR_PNG_EVIDENCE_IS_ACCEPTED")
    evidence_id = uuid.uuid4().hex
    event_id = uuid.uuid4().hex
    data_root, _, _ = get_data_paths()
    relpath = Path("evidence") / cycle_id / f"{evidence_id}{suffix}"
    absolute = data_root / relpath
    absolute.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation makes accidental overwrites impossible.
    try:
        with absolute.open("xb") as stream:
            stream.write(body)
    except FileExistsError as exc:
        raise HTTPException(409, "EVIDENCE_NAME_COLLISION_RETRY") from exc
    sha = hashlib.sha256(body).hexdigest()
    try:
        metadata = {
            "pipe_color": x_pipe_color, "operation_event": x_operation_event,
            "operator_anon_id": x_operator_anon_id,
            "camera_metadata": camera_metadata,
            "label_status": x_label_status, "captured_at_utc": captured.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
        }
        with connect_db() as db:
            now = metadata["captured_at_utc"]
            if not db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone():
                db.execute("INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state) VALUES(?,?,?,?,?)",
                           (cycle_id, now, now, "WAITING", "WAIT_PART"))
            event_id = _event(db, cycle_id, f"EVIDENCE_{kind}", metadata, outcome="WAITING",
                              reason=x_operation_event, capture_monotonic_s=x_capture_monotonic_s, frame_id=x_frame_id)
            db.execute("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                evidence_id, cycle_id, event_id, now, kind, relpath.as_posix(), mime, sha,
                x_frame_id, x_capture_monotonic_s, json.dumps(metadata, ensure_ascii=False),
            ))
            if kind == "initial_label" and x_label_status:
                db.execute("UPDATE cycles SET initial_label=COALESCE(initial_label,?) WHERE cycle_id=?",
                           (x_label_status, cycle_id))
            if x_pipe_color or x_operator_anon_id:
                db.execute("UPDATE cycles SET updated_at_utc=?,pipe_color=COALESCE(?,pipe_color),operator_anon_id=COALESCE(?,operator_anon_id) WHERE cycle_id=?",
                           (now, x_pipe_color, x_operator_anon_id, cycle_id))
    except Exception:
        # If metadata/database persistence fails, remove only this newly created file.
        absolute.unlink(missing_ok=True)
        raise
    return {"evidence_id": evidence_id, "cycle_id": cycle_id, "kind": kind, "sha256": sha,
            "frame_id": x_frame_id, "capture_monotonic_s": x_capture_monotonic_s}


@app.get("/api/evidence")
def list_evidence(limit: int = 100):
    with connect_db() as db:
        rows = db.execute("SELECT evidence_id,cycle_id,created_at_utc,kind,mime_type,sha256,frame_id,capture_monotonic_s,metadata_json FROM evidence ORDER BY created_at_utc DESC LIMIT ?",
                           (max(1, min(limit, 500)),)).fetchall()
    return [
        {
            **{k: v for k, v in dict(r).items() if k != "metadata_json"},
            "metadata": json.loads(r["metadata_json"]),
            "url": f"/api/evidence/{r['evidence_id']}/file",
        }
        for r in rows
    ]


@app.get("/api/evidence/{evidence_id}/file")
def evidence_file(evidence_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", evidence_id):
        raise HTTPException(404, "EVIDENCE_NOT_FOUND")
    with connect_db() as db:
        row = db.execute("SELECT file_relpath,mime_type FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
    if not row:
        raise HTTPException(404, "EVIDENCE_NOT_FOUND")
    data_root, _, evidence_root = get_data_paths()
    path = (data_root / row["file_relpath"]).resolve()
    if evidence_root.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "EVIDENCE_NOT_FOUND")
    return FileResponse(path, media_type=row["mime_type"], headers={"Cache-Control": "no-store"})


@app.delete("/api/evidence/{evidence_id}")
def delete_evidence(evidence_id: str, x_admin_token: str | None = Header(default=None)):
    admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    if not re.fullmatch(r"[0-9a-f]{32}", evidence_id):
        raise HTTPException(404, "EVIDENCE_NOT_FOUND")
    initialize_database()
    data_root, _, evidence_root = get_data_paths()
    with connect_db() as db:
        row = db.execute(
            "SELECT evidence_id, cycle_id, file_relpath FROM evidence WHERE evidence_id=?",
            (evidence_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "EVIDENCE_NOT_FOUND")
        db.execute("DELETE FROM evidence WHERE evidence_id=?", (evidence_id,))
    deleted_file = _safe_unlink_evidence(data_root, evidence_root, row["file_relpath"])
    freed = 1 if deleted_file else 0
    return {
        "deleted": True,
        "evidence_id": evidence_id,
        "cycle_id": row["cycle_id"],
        "deleted_files": freed,
        "freed_files": freed,
    }


class AuxiliaryRoiToggle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    show: bool


@app.post("/api/camera/auxiliary_rois")
def toggle_auxiliary_rois(payload: AuxiliaryRoiToggle):
    show = LIVE_CAMERA.set_show_auxiliary_rois(payload.show)
    return {"show_auxiliary_rois": show}


class MirrorToggle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mirror: bool


@app.post("/api/camera/mirror")
def toggle_camera_mirror(payload: MirrorToggle):
    mirror = LIVE_CAMERA.set_mirror_horizontal(payload.mirror)
    return {"mirror_horizontal": mirror}


class WorkModeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    work_mode: Literal["debug", "strict"]


@app.post("/api/camera/work_mode")
def update_work_mode(payload: WorkModeUpdate):
    mode = LIVE_CAMERA.set_work_mode(payload.work_mode)
    return {"work_mode": mode}


class DatasetCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["complete", "crossed_out", "covered", "absent", "unknown_challenge"]
    pipe_color: Literal["yellow", "blue"] = "yellow"
    physical_tube_id: str | None = None
    batch_id: str | None = None
    box: list[float] | None = None


@app.get("/api/dataset/summary")
def dataset_summary():
    initialize_database()
    return TRAINING_MANAGER.get_dataset_summary()


@app.get("/api/dataset/samples")
def dataset_list_samples(label: str | None = None, q: str | None = None, limit: int = 200):
    initialize_database()
    try:
        return TRAINING_MANAGER.list_samples(label=label, query=q, limit=limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/dataset/upload")
async def dataset_upload_sample(
    request: Request,
    x_sample_label: str = Header(...),
    x_pipe_color: str = Header(default="yellow"),
    x_tube_id: str | None = Header(default=None),
    x_batch_id: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
):
    if os.environ.get("IGBT_ADMIN_TOKEN"):
        admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    initialize_database()

    clean_label = x_sample_label.strip()
    if clean_label not in ALL_LABELS:
        raise HTTPException(422, f"INVALID_SAMPLE_LABEL:{clean_label}")
    clean_color = x_pipe_color.strip().lower() or "yellow"
    if clean_color not in {"yellow", "blue"}:
        raise HTTPException(422, f"INVALID_PIPE_COLOR:{clean_color}")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not content_length.isdigit() or int(content_length) > MAX_EVIDENCE_BYTES:
            raise HTTPException(413, "SAMPLE_IMAGE_TOO_LARGE_OR_INVALID_LENGTH")
    body = await request.body()
    if not body:
        raise HTTPException(400, "EMPTY_SAMPLE_IMAGE_BODY")
    if len(body) > MAX_EVIDENCE_BYTES:
        raise HTTPException(413, "SAMPLE_IMAGE_EXCEEDS_MAX_SIZE")

    mime = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if mime and mime not in {"image/jpeg", "image/jpg", "image/png", "application/octet-stream"}:
        raise HTTPException(415, "ONLY_JPEG_OR_PNG_SAMPLE_IS_ACCEPTED")
    is_jpeg = body.startswith(b"\xff\xd8")
    is_png = body.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise HTTPException(415, "ONLY_VALID_JPEG_OR_PNG_BYTES_ACCEPTED")

    try:
        return TRAINING_MANAGER.upload_sample_bytes(
            image_bytes=body,
            label=clean_label,
            pipe_color=clean_color,
            physical_tube_id=_clean_optional_header(x_tube_id),
            batch_id=_clean_optional_header(x_batch_id),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/dataset/capture")
def dataset_capture_live_sample(payload: DatasetCaptureRequest):
    initialize_database()
    crop_bgr = LIVE_CAMERA.get_custom_crop_bgr(payload.box)
    if crop_bgr is None or crop_bgr.size == 0:
        raise HTTPException(503, "当前未获取到实时摄像头画面")

    sample = TRAINING_MANAGER.capture_sample(
        crop_bgr=crop_bgr,
        label=payload.label,
        pipe_color=payload.pipe_color,
        physical_tube_id=payload.physical_tube_id,
        batch_id=payload.batch_id,
        box_xyxy=payload.box,
    )

    # Simultaneously archive full-frame evidence into SQLite so evidence history stays complete
    jpg = LIVE_CAMERA.get_raw_frame_jpeg()
    if jpg:
        live_state = LIVE_CAMERA.get_state()
        cycle_id = _safe_cycle_id(live_state.get("cycle_id") or f"LIVE_{datetime.now().strftime('%H%M%S')}")
        now = utc_now()
        evidence_id = uuid.uuid4().hex
        sha = hashlib.sha256(jpg).hexdigest()
        data_root, _, _ = get_data_paths()
        relpath = Path("evidence") / cycle_id / f"{evidence_id}.jpg"
        absolute = data_root / relpath
        absolute.parent.mkdir(parents=True, exist_ok=True)
        with absolute.open("xb") as stream:
            stream.write(jpg)

        kind_map = {
            "complete": "initial_label",
            "crossed_out": "A",
            "covered": "B",
            "absent": "C",
            "unknown_challenge": "anomaly",
        }
        ev_kind = kind_map.get(payload.label, "initial_label")
        lbl_status = "unknown" if payload.label == "unknown_challenge" else payload.label
        meta = {
            "pipe_color": payload.pipe_color,
            "operation_event": f"DATASET_SAMPLE_{payload.label.upper()}",
            "operator_anon_id": "OP_STATION_01",
            "sample_id": sample["sample_id"],
            "physical_tube_id": sample["physical_tube_id"],
            "batch_id": sample["batch_id"],
            "camera_metadata": {
                "camera_name": live_state.get("camera_name"),
                "camera_device_id": live_state.get("camera_device_id"),
                "resolution": f"{live_state.get('camera_width')}x{live_state.get('camera_height')}",
                "fps": live_state.get("fps"),
            },
            "label_status": lbl_status,
            "captured_at_utc": now,
        }
        with connect_db() as db:
            if not db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone():
                db.execute(
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cycle_id,
                        now,
                        now,
                        lbl_status,
                        live_state.get("outcome") or "WAITING",
                        live_state.get("result_reason"),
                        payload.pipe_color,
                        "OP_STATION_01",
                        live_state.get("model_version"),
                        live_state.get("config_version"),
                        live_state.get("fsm_state") or "INSPECTING",
                    ),
                )
            ev_id = _event(
                db,
                cycle_id,
                f"SAMPLE_{payload.label.upper()}",
                meta,
                state_before=live_state.get("fsm_state"),
                state_after=live_state.get("fsm_state"),
                outcome=live_state.get("outcome"),
                reason=f"采集训练样本 ({payload.label})",
                capture_monotonic_s=live_state.get("capture_monotonic_s"),
                frame_id=live_state.get("frame_id"),
            )
            db.execute(
                "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    evidence_id,
                    cycle_id,
                    ev_id,
                    now,
                    ev_kind,
                    relpath.as_posix(),
                    "image/jpeg",
                    sha,
                    live_state.get("frame_id") or 1,
                    live_state.get("capture_monotonic_s"),
                    json.dumps(meta, ensure_ascii=False),
                ),
            )
    return sample


@app.post("/api/dataset/bootstrap")
def dataset_bootstrap():
    initialize_database()
    crop_bgr = LIVE_CAMERA.get_tracked_crop_bgr()
    return TRAINING_MANAGER.bootstrap_standard_dataset(live_crop_bgr=crop_bgr)


class BatchDeleteSamplesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_ids: list[str] | None = None
    label: Literal["complete", "crossed_out", "covered", "absent", "unknown_challenge"] | None = None


@app.post("/api/dataset/batch_delete")
def dataset_batch_delete(
    payload: BatchDeleteSamplesRequest,
    x_admin_token: str | None = Header(default=None),
):
    admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    initialize_database()
    if not payload.sample_ids and payload.label is None:
        raise HTTPException(422, "MUST_SPECIFY_SAMPLE_IDS_OR_LABEL")
    if payload.sample_ids is not None:
        for sid in payload.sample_ids:
            if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", sid):
                raise HTTPException(422, f"INVALID_SAMPLE_ID:{sid}")
    try:
        return TRAINING_MANAGER.batch_delete_samples(
            sample_ids=payload.sample_ids,
            label=payload.label,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/dataset/clear")
def dataset_clear(x_admin_token: str | None = Header(default=None)):
    if os.environ.get("IGBT_ADMIN_TOKEN"):
        admin_or_station_token(x_admin_token, "IGBT_ADMIN_TOKEN")
    initialize_database()
    return TRAINING_MANAGER.clear_dataset()


@app.get("/api/dataset/coco")
def dataset_coco_export():
    data_root, _, _ = get_data_paths()
    coco_path = data_root / "dataset" / "annotations_coco.json"
    if not coco_path.is_file():
        return {"images": [], "annotations": [], "categories": []}
    return FileResponse(coco_path, media_type="application/json", filename="igbt_annotations_coco.json")


@app.get("/api/dataset/sample/{sample_id}/image")
def dataset_sample_image(sample_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", sample_id):
        raise HTTPException(404, "SAMPLE_NOT_FOUND")
    path = TRAINING_MANAGER.get_sample_image_path(sample_id)
    if not path or not path.is_file():
        raise HTTPException(404, "SAMPLE_NOT_FOUND")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class SampleAnnotationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["complete", "crossed_out", "covered", "absent", "unknown_challenge"] | None = None
    physical_tube_id: str | None = None
    batch_id: str | None = None
    pipe_color: Literal["yellow", "blue"] | None = None


@app.patch("/api/dataset/sample/{sample_id}")
def update_dataset_sample(sample_id: str, payload: SampleAnnotationUpdate):
    initialize_database()
    try:
        return TRAINING_MANAGER.update_sample_annotation(
            sample_id=sample_id,
            new_label=payload.label,
            physical_tube_id=payload.physical_tube_id,
            batch_id=payload.batch_id,
            pipe_color=payload.pipe_color,
        )
    except KeyError as exc:
        raise HTTPException(404, "SAMPLE_NOT_FOUND") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/api/dataset/sample/{sample_id}")
def delete_dataset_sample(sample_id: str):
    initialize_database()
    try:
        return TRAINING_MANAGER.delete_sample(sample_id)
    except KeyError as exc:
        raise HTTPException(404, "SAMPLE_NOT_FOUND") from exc


class TrainingStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=8, ge=2, le=50)
    input_size: int = Field(default=96, ge=64, le=320)
    batch_size: int = Field(default=8, ge=2, le=64)
    learning_rate: float = Field(default=0.0025, gt=0.0, le=0.1)
    resume_from_latest: bool = True
    use_augmentation: bool = True


def _on_training_completed(run_summary: dict) -> None:
    try:
        initialize_database()
        model_id = uuid.uuid4().hex
        notes = (
            f"测试集准确率: {run_summary['test_accuracy'] * 100:.1f}% · "
            f"完整标签误判率: {run_summary['complete_false_accept_total']} · "
            f"ONNX 误差: {run_summary['onnx_max_abs_diff']} · "
            f"样本(训练/验证/测试): {run_summary['train_samples']}/{run_summary['val_samples']}/{run_summary['test_samples']}"
        )
        with connect_db() as db:
            db.execute(
                "INSERT INTO model_registry VALUES(?,?,?,?,?,?,?,?)",
                (
                    model_id,
                    run_summary["model_name"],
                    run_summary["model_version"],
                    run_summary["onnx_sha256"].lower(),
                    "reviewed",
                    notes,
                    utc_now(),
                    "trainer-pipeline",
                ),
            )
        LIVE_CAMERA.load_onnx_model(
            onnx_path=run_summary["onnx_path"],
            model_version=run_summary["model_version"],
            sha256=run_summary["onnx_sha256"],
            input_size=run_summary["input_size"],
        )
    except Exception:
        logger.warning("Failed to persist or hot-load completed training model", exc_info=True)


@app.get("/api/training/status")
def training_status():
    return TRAINING_MANAGER.get_training_status()


@app.post("/api/training/start")
def start_training(payload: TrainingStartRequest):
    initialize_database()
    try:
        return TRAINING_MANAGER.start_training(
            epochs=payload.epochs,
            input_size=payload.input_size,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
            resume_from_latest=payload.resume_from_latest,
            use_augmentation=payload.use_augmentation,
            on_complete=_on_training_completed,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/models/{model_id}/activate")
def activate_registered_model(model_id: str):
    initialize_database()
    with connect_db() as db:
        row = db.execute(
            "SELECT model_id,model_name,model_version,sha256 FROM model_registry WHERE model_id=?",
            (model_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "MODEL_NOT_FOUND")
    version = row["model_version"]
    data_root, _, _ = get_data_paths()
    run_dir = _safe_model_run_dir(data_root, version)
    onnx_candidate = (run_dir / "label_classifier.onnx") if run_dir is not None else None
    if onnx_candidate is not None and onnx_candidate.is_file():
        info = LIVE_CAMERA.load_onnx_model(onnx_candidate, model_version=version, sha256=row["sha256"])
        return {"activated": True, "mode": "onnx_runtime", **info}
    with LIVE_CAMERA._lock:
        LIVE_CAMERA._onnx_model_version = version
        LIVE_CAMERA._state["model_version"] = version
    return {"activated": True, "mode": "metadata_switch", "model_version": version, "sha256": row["sha256"]}


if __name__ == "__main__":
    import uvicorn
    os.environ.setdefault("IGBT_ENABLE_LIVE_CAMERA", "1")
    initialize_database()
    # Deliberately fixed: this application has no option to bind to a LAN address.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("IGBT_UI_PORT", "8765")), reload=False)
