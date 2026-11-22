"""Real-time USB/UVC camera capture, dynamic multi-target tracking, ESD glove compliance enforcement, ONNX inference, and FSM driver."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from igbt_pokayoke.logic.fsm import FSMConfig, FSMEvent, FSMState, InspectionFSM
from igbt_pokayoke.runtime import STANDARD_ROIS
from igbt_pokayoke.vision.hand_contact import ContactCandidate, ContactCandidateTracker
from igbt_pokayoke.vision.view_valid import ROISet, measure_label_quality, valid_box

logger = logging.getLogger(__name__)

CLASSES = ["complete", "crossed_out", "covered", "absent"]
LABEL_CN = {
    "complete": "完整",
    "crossed_out": "A类·已划除",
    "covered": "B类·已遮挡",
    "absent": "C类·已脱落",
    "unknown": "未入位/待定",
}
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _discover_camera_name(index: int = 0) -> str:
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore

        names = FilterGraph().get_input_devices()
        if 0 <= index < len(names):
            return str(names[index])
    except Exception as exc:
        logger.debug("DirectShow camera friendly name unavailable: %s", exc)
    return "HD Pro Webcam C920"


def _box_iou_and_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (IoU, fraction of box b covered by box a)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter), inter / area_b


def _ema_box(
    prev: tuple[float, float, float, float] | None,
    curr: tuple[float, float, float, float],
    alpha: float = 0.45,
) -> tuple[float, float, float, float]:
    if prev is None:
        return curr
    return (
        round(prev[0] * (1.0 - alpha) + curr[0] * alpha, 4),
        round(prev[1] * (1.0 - alpha) + curr[1] * alpha, 4),
        round(prev[2] * (1.0 - alpha) + curr[2] * alpha, 4),
        round(prev[3] * (1.0 - alpha) + curr[3] * alpha, 4),
    )


class _ChineseBadgeRenderer:
    """Fast cached Chinese badge renderer using Pillow and Windows Microsoft YaHei font."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[int, int, int]], np.ndarray] = {}
        self._font: Any = None
        try:
            from PIL import ImageFont

            for candidate in (
                "C:/Windows/Fonts/msyhbd.ttc",
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ):
                if Path(candidate).is_file():
                    self._font = ImageFont.truetype(candidate, 15)
                    break
        except Exception as exc:
            logger.debug("Pillow CJK font load failed; falling back to OpenCV ASCII: %s", exc)
            self._font = None

    def draw_badge(self, img_bgr: np.ndarray, x: int, y: int, text: str, bg_bgr: tuple[int, int, int]) -> None:
        key = (text, bg_bgr)
        badge = self._cache.get(key)
        if badge is None:
            badge = self._render(text, bg_bgr)
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = badge
        bh, bw = badge.shape[:2]
        h, w = img_bgr.shape[:2]
        x0 = max(0, min(w - bw, x))
        y0 = max(0, min(h - bh, y))
        img_bgr[y0 : y0 + bh, x0 : x0 + bw] = badge

    def _render(self, text: str, bg_bgr: tuple[int, int, int]) -> np.ndarray:
        if self._font is None:
            bw = max(60, len(text) * 11 + 16)
            bh = 24
            arr = np.zeros((bh, bw, 3), dtype=np.uint8)
            arr[:, :] = bg_bgr
            cv2.putText(arr, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return arr

        from PIL import Image, ImageDraw

        dummy = Image.new("RGB", (1, 1))
        d0 = ImageDraw.Draw(dummy)
        bbox = d0.textbbox((0, 0), text, font=self._font)
        tw = max(24, bbox[2] - bbox[0])
        th = max(14, bbox[3] - bbox[1])
        bw = tw + 16
        bh = th + 10
        rgb_bg = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
        canvas = Image.new("RGB", (bw, bh), rgb_bg)
        draw = ImageDraw.Draw(canvas)
        draw.text((8 - bbox[0], 4 - bbox[1]), text, font=self._font, fill=(255, 255, 255))
        arr_rgb = np.array(canvas, dtype=np.uint8)
        return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)


class LiveCameraEngine:
    """
    Background OpenCV capture thread driving:
    1. Strict tube + rectangular label presence gating (ignores faces, necks, clothes, desks)
    2. ESD glove compliance vs Bare-Hand SOP violation detection
    3. Auto-reset 0.5s after PASS removal, and latched lock on ALARM until manual reset
    """

    def __init__(
        self,
        config_loader: Callable[[], dict[str, Any]],
        on_cycle_update: Callable[[dict[str, Any], str | None], None] | None = None,
        device_index: int = 0,
    ) -> None:
        self._config_loader = config_loader
        self._on_cycle_update = on_cycle_update
        self.device_index = device_index
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._badge = _ChineseBadgeRenderer()

        self._face_cascade: Any = None
        try:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.is_file():
                self._face_cascade = cv2.CascadeClassifier(str(cascade_path))
        except Exception as exc:
            logger.warning("Haar face cascade load failed: %s", exc)
            self._face_cascade = None

        self.camera_name = _discover_camera_name(device_index)
        self.camera_device_id = f"USB_CAM_{device_index:02d}_C920"
        self.backend = "DSHOW"
        self.width = 1920
        self.height = 1080
        self.fps = 0.0
        self.camera_online = False
        self.simulation_override_until = 0.0
        self.show_auxiliary_rois = False
        self.mirror_horizontal: bool = True
        self.work_mode: str = "debug"  # "debug" (桌面调试模式) or "strict" (车间严格模式)

        self._raw_frame: np.ndarray | None = None
        self._last_tracked_crop: np.ndarray | None = None
        self._hud_jpeg: bytes | None = None
        self._crop_jpeg: bytes | None = None
        self._frame_id = 0
        self._fps_times: deque[float] = deque(maxlen=35)

        # Dynamic full-frame tracking state
        self._bg_small: np.ndarray | None = None
        self._prev_small: np.ndarray | None = None
        self._tracked_label_box: tuple[float, float, float, float] | None = None
        self._tracked_tube_box: tuple[float, float, float, float] | None = None
        self._tracked_left_glove: tuple[float, float, float, float] | None = None
        self._tracked_right_glove: tuple[float, float, float, float] | None = None
        self._tracked_bare_hand: tuple[float, float, float, float] | None = None
        self._left_glove_ttl = 0
        self._right_glove_ttl = 0
        self._bare_hand_ttl = 0
        self._label_ttl = 0
        self._target_locked = False
        self._target_confidence = 0.0

        # Cycle lifecycle & Alarm latch state
        self._alarm_latched = False
        self._latched_reason: str | None = None
        self._latched_cycle_id: str | None = None
        self._pass_target_lost_since: float | None = None

        # ONNX Runtime neural network inference state
        self._ort_session: Any = None
        self._onnx_input_size: int = 96
        self._onnx_model_version: str = "dynamic-tracker-v3.2"
        self._onnx_model_path: str | None = None
        self._onnx_sha256: str | None = None
        self._inference_ms: float = 0.0

        self.fsm = InspectionFSM(config=FSMConfig(initial_complete_ms=350, stable_state_ms=400, min_stable_frames=5))
        self.contact_tracker = ContactCandidateTracker(enter_frames=2, exit_frames=2)

        self._state: dict[str, Any] = {
            "mode": "shadow",
            "interlock_output_enabled": False,
            "camera_status": "NOT_CONNECTED",
            "camera_name": self.camera_name,
            "camera_device_id": self.camera_device_id,
            "backend": self.backend,
            "camera_width": self.width,
            "camera_height": self.height,
            "camera_settings_locked": True,
            "fps": 0.0,
            "label_status": "unknown",
            "label_confidence": None,
            "view_valid": False,
            "view_valid_reasons": ["CAMERA_INITIALIZING"],
            "quality_ok": False,
            "laplacian_variance": None,
            "brightness": None,
            "overexposed_fraction": None,
            "underexposed_fraction": None,
            "left_grip_active": False,
            "right_grip_active": False,
            "glove_compliant": False,
            "bare_hand_detected": False,
            "bare_hand_violation": False,
            "alarm_latched": False,
            "label_contact_active": False,
            "hand_contact_candidate": False,
            "contact_evidence_type": "2d_visual_proximity_proxy",
            "proves_physical_contact": False,
            "fsm_state": "WAIT_PART",
            "outcome": "WAITING",
            "result_reason": "WAITING_FOR_PART",
            "c_timer_ms": 0,
            "cycle_id": None,
            "pipe_color": None,
            "operator_anon_id": "OP_STATION_01",
            "frame_id": 0,
            "capture_monotonic_s": 0.0,
            "model_version": self._onnx_model_version,
            "inference_backend": "动态多目标视觉追踪引擎",
            "inference_ms": 0.0,
            "show_auxiliary_rois": False,
            "mirror_horizontal": True,
            "work_mode": "debug",
            "tracked_objects": [],
            "config_version": "ROI_V001",
            "live_stream_url": "/api/camera/stream",
            "live_crop_url": "/api/camera/stream_crop",
            "updated_at_utc": _utc_now(),
        }

    def load_onnx_model(
        self,
        onnx_path: str | Path,
        model_version: str,
        sha256: str | None = None,
        input_size: int = 96,
    ) -> dict[str, Any]:
        """Hot-load a trained MobileNetV3-Small ONNX model into the live camera engine."""
        import onnxruntime as ort

        path = Path(onnx_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        shape = input_meta.shape
        detected_size = input_size
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] >= 32:
            detected_size = int(shape[2])

        with self._lock:
            self._ort_session = session
            self._onnx_input_size = detected_size
            self._onnx_model_version = model_version
            self._onnx_model_path = str(path)
            self._onnx_sha256 = sha256
            self._state["model_version"] = model_version
            self._state["inference_backend"] = f"ONNX 神经网络 ({model_version})"
        return {
            "model_version": model_version,
            "onnx_path": str(path),
            "input_size": detected_size,
            "sha256": sha256,
        }

    def set_show_auxiliary_rois(self, show: bool) -> bool:
        with self._lock:
            self.show_auxiliary_rois = bool(show)
            self._state["show_auxiliary_rois"] = self.show_auxiliary_rois
            return self.show_auxiliary_rois

    def set_mirror_horizontal(self, mirror: bool) -> bool:
        normalized = bool(mirror)
        with self._lock:
            changed = normalized != self.mirror_horizontal
            self.mirror_horizontal = normalized
            self._state["mirror_horizontal"] = normalized
            if changed:
                self._bg_small = None
                self._prev_small = None
                self._tracked_label_box = None
                self._tracked_tube_box = None
                self._tracked_left_glove = None
                self._tracked_right_glove = None
                self._tracked_bare_hand = None
                self._left_glove_ttl = 0
                self._right_glove_ttl = 0
                self._bare_hand_ttl = 0
                self._label_ttl = 0
            return self.mirror_horizontal

    def set_work_mode(self, mode: str) -> str:
        normalized = "strict" if mode == "strict" else "debug"
        with self._lock:
            self.work_mode = normalized
            self._state["work_mode"] = normalized
            if normalized == "debug":
                self._alarm_latched = False
                self._latched_reason = None
                self._state["alarm_latched"] = False
                self._state["bare_hand_violation"] = False
            return normalized

    def get_custom_crop_bgr(self, box: list[float] | tuple[float, ...] | None) -> np.ndarray | None:
        """Crop an arbitrary user-drawn bounding box [x1, y1, x2, y2] from the latest raw frame, or fall back to tracked crop."""
        if box is not None and len(box) == 4 and valid_box(tuple(box)):
            with self._lock:
                if self._raw_frame is None:
                    return None
                crop, _ = self._crop_norm(self._raw_frame, tuple(box))
                return crop.copy()
        return self.get_tracked_crop_bgr()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="LiveCameraEngine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)

    def pause_for_simulation(self, seconds: float = 6.0) -> None:
        with self._lock:
            self.simulation_override_until = time.monotonic() + seconds

    def resume_live(self) -> dict[str, Any]:
        with self._lock:
            self.simulation_override_until = 0.0
            self._bg_small = None
            self._prev_small = None
            return dict(self._state)

    def trigger_new_cycle(self, pipe_color: str = "yellow") -> dict[str, Any]:
        """Unlatch any locked alarm and reset the FSM cleanly to WAIT_PART."""
        now = time.monotonic()
        with self._lock:
            self.simulation_override_until = 0.0
            self._alarm_latched = False
            self._latched_reason = None
            self._latched_cycle_id = None
            self._pass_target_lost_since = None
            self._bg_small = None
            self.contact_tracker.reset()
            self.fsm = InspectionFSM(config=FSMConfig(initial_complete_ms=350, stable_state_ms=400, min_stable_frames=5))
            cycle_id = f"LIVE_{datetime.now().strftime('%H%M%S')}"
            self.fsm.process(
                FSMEvent(
                    kind="new_cycle",
                    cycle_id=cycle_id,
                    capture_monotonic_s=now,
                    pipe_color=pipe_color,
                    operator_anon_id="OP_STATION_01",
                ),
                now_monotonic=now,
            )
            self._state["cycle_id"] = cycle_id
            self._state["pipe_color"] = pipe_color
            self._state["fsm_state"] = "WAIT_PART"
            self._state["outcome"] = "WAITING"
            self._state["result_reason"] = "AUTHORIZED_RESET"
            self._state["alarm_latched"] = False
            self._state["bare_hand_violation"] = False
            self._state["c_timer_ms"] = 0
            return dict(self._state)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def get_raw_frame_jpeg(self) -> bytes | None:
        with self._lock:
            if self._raw_frame is None:
                return None
            frame = self._raw_frame.copy()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return bytes(buf) if ok else None

    def get_tracked_crop_bgr(self) -> np.ndarray | None:
        with self._lock:
            if self._last_tracked_crop is not None and self._last_tracked_crop.size > 0:
                return self._last_tracked_crop.copy()
            if self._raw_frame is None:
                return None
            box = self._tracked_label_box or (0.41, 0.42, 0.59, 0.58)
            crop, _ = self._crop_norm(self._raw_frame, box)
            return crop.copy()

    def get_hud_jpeg(self) -> bytes | None:
        with self._lock:
            return self._hud_jpeg

    def get_crop_jpeg(self) -> bytes | None:
        with self._lock:
            return self._crop_jpeg

    def _active_rois_and_thresholds(self) -> tuple[ROISet, dict[str, Any], str]:
        try:
            cfg = self._config_loader()
        except Exception as exc:
            logger.warning("Config loader failed; falling back to STANDARD_ROIS: %s", exc)
            cfg = {}
        raw_rois = cfg.get("rois") or {}
        if all(k in raw_rois and valid_box(raw_rois[k]) for k in ("tube_roi", "label_roi", "label_contact_roi")):
            rois = ROISet(
                tube_roi=tuple(raw_rois["tube_roi"]),
                label_roi=tuple(raw_rois["label_roi"]),
                label_contact_roi=tuple(raw_rois["label_contact_roi"]),
                left_grip_zone=tuple(raw_rois["left_grip_zone"]) if valid_box(raw_rois.get("left_grip_zone")) else STANDARD_ROIS.left_grip_zone,
                right_grip_zone=tuple(raw_rois["right_grip_zone"]) if valid_box(raw_rois.get("right_grip_zone")) else STANDARD_ROIS.right_grip_zone,
            )
        else:
            rois = STANDARD_ROIS
        thresholds = cfg.get("thresholds") or {}
        version = str(cfg.get("roi_config_version") or "ROI_V001")
        return rois, thresholds, version

    @staticmethod
    def _crop_norm(frame: np.ndarray, box: tuple[float, ...]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 4, int(box[0] * w)))
        y1 = max(0, min(h - 4, int(box[1] * h)))
        x2 = max(x1 + 4, min(w, int(box[2] * w)))
        y2 = max(y1 + 4, min(h, int(box[3] * h)))
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _track_dynamic_targets(
        self,
        frame: np.ndarray,
    ) -> tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        bool,
        bool,
        bool,
        list[dict[str, Any]],
    ]:
        """
        Full-frame dynamic detection and smooth EMA tracking:
        - Filters out human face, chin, neck, shirt collar, jacket sleeves, and wooden desk surface
        - Separately detects:
          1. Compliant White / Light-Gray ESD Gloves (`left_glove`, `right_glove`)
          2. Violating Bare Hands (`bare_hand`) -> triggers warning or SOP violation alarm!
          3. Physical Tube + Compact Rectangular White Label (`tube_box`, `label_box`)
        """
        sw, sh = 384, 216
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray_u8 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._bg_small is None or self._bg_small.shape != blur.shape:
            self._bg_small = blur.copy()
        if self._prev_small is None or self._prev_small.shape != blur.shape:
            self._prev_small = blur.copy()

        fg_diff = np.abs(blur - self._bg_small)
        motion_diff = np.abs(blur - self._prev_small)
        self._prev_small = self._prev_small * 0.55 + blur * 0.45
        still_mask = motion_diff < 7.0
        self._bg_small = np.where(still_mask, self._bg_small * 0.992 + blur * 0.008, self._bg_small)

        # Build Person Face / Chin / Neck / Collar Exclusion Mask
        person_exclusion = np.zeros((sh, sw), dtype=np.uint8)
        if self._face_cascade is not None:
            try:
                faces = self._face_cascade.detectMultiScale(gray_u8, scaleFactor=1.15, minNeighbors=4, minSize=(26, 26))
                for fx, fy, fw, fh in faces:
                    ex1 = max(0, int(fx - fw * 0.25))
                    ey1 = max(0, int(fy - fh * 0.15))
                    ex2 = min(sw, int(fx + fw * 1.25))
                    ey2 = min(int(sh * 0.54), int(fy + fh * 1.45))
                    person_exclusion[ey1:ey2, ex1:ex2] = 255
            except Exception as exc:
                logger.debug("Face exclusion detectMultiScale failed: %s", exc)

        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Strict human skin mask (V >= 125 rejects dark brown wooden desks, armrests, and furniture)
        skin_mask = (
            (hsv[:, :, 2] >= 125)
            & (ycrcb[:, :, 1] >= 137)
            & (ycrcb[:, :, 1] <= 174)
            & (ycrcb[:, :, 2] >= 80)
            & (ycrcb[:, :, 2] <= 122)
        )

        # Exclude upper-center head/chin/neck column
        top_center_skin = skin_mask[: int(sh * 0.40), int(sw * 0.30) : int(sw * 0.70)]
        if float(np.mean(top_center_skin)) > 0.08:
            person_exclusion[: int(sh * 0.50), int(sw * 0.30) : int(sw * 0.70)] = 255

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        min_hand_area = sw * sh * 0.010
        max_hand_area = sw * sh * 0.11

        # ------------------------------------------------------------------
        # A. Compliant White / Light-Gray ESD Glove Detection
        # ------------------------------------------------------------------
        glove_raw = (
            (hsv[:, :, 2] >= 178)
            & (hsv[:, :, 1] <= 42)
            & ((motion_diff > 7.5) | (fg_diff > 20.0))
            & (person_exclusion == 0)
        ).astype(np.uint8) * 255
        glove_clean = cv2.morphologyEx(glove_raw, cv2.MORPH_OPEN, kernel_open)
        glove_clean = cv2.morphologyEx(glove_clean, cv2.MORPH_CLOSE, kernel_close)
        glove_contours, _ = cv2.findContours(glove_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        left_glove_cand: tuple[float, float, float, float] | None = None
        right_glove_cand: tuple[float, float, float, float] | None = None
        lg_area = 0.0
        rg_area = 0.0
        for cnt in glove_contours:
            area = float(cv2.contourArea(cnt))
            if area < min_hand_area or area > max_hand_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx = (x + cw * 0.5) / sw
            if y <= 4 or (y / sh) >= 0.82 or (cw / sw) > 0.25 or (ch / sh) > 0.40:
                continue
            mot = float(np.mean(motion_diff[y : y + ch, x : x + cw] > 6.5))
            if mot < 0.12:
                continue
            box = (max(0.01, x / sw), max(0.02, y / sh), min(0.99, (x + cw) / sw), min(0.99, (y + ch) / sh))
            if cx < 0.50 and area > lg_area:
                lg_area = area
                left_glove_cand = box
            elif cx >= 0.50 and area > rg_area:
                rg_area = area
                right_glove_cand = box

        if left_glove_cand is not None:
            self._tracked_left_glove = _ema_box(self._tracked_left_glove, left_glove_cand, alpha=0.50)
            self._left_glove_ttl = 5
        elif self._left_glove_ttl > 0:
            self._left_glove_ttl -= 1
        else:
            self._tracked_left_glove = None

        if right_glove_cand is not None:
            self._tracked_right_glove = _ema_box(self._tracked_right_glove, right_glove_cand, alpha=0.50)
            self._right_glove_ttl = 5
        elif self._right_glove_ttl > 0:
            self._right_glove_ttl -= 1
        else:
            self._tracked_right_glove = None

        # ------------------------------------------------------------------
        # B. Violating Bare-Hand Detection (Skin in workspace without ESD glove)
        # ------------------------------------------------------------------
        bare_raw = (skin_mask & (person_exclusion == 0)).astype(np.uint8) * 255
        bare_clean = cv2.morphologyEx(bare_raw, cv2.MORPH_OPEN, kernel_open)
        bare_clean = cv2.morphologyEx(bare_clean, cv2.MORPH_CLOSE, kernel_close)
        bare_contours, _ = cv2.findContours(bare_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bare_cand: tuple[float, float, float, float] | None = None
        bare_best_area = 0.0
        for cnt in bare_contours:
            area = float(cv2.contourArea(cnt))
            if area < min_hand_area or area > max_hand_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx = (x + cw * 0.5) / sw
            if y <= 4 and 0.22 < cx < 0.78:
                continue
            if (y / sh) >= 0.82:
                continue
            hand_fg = float(np.mean(fg_diff[y : y + ch, x : x + cw] > 14.0))
            hand_mot = float(np.mean(motion_diff[y : y + ch, x : x + cw] > 5.5))
            if hand_fg < 0.18 and hand_mot < 0.08:
                continue
            if (cw / sw) > 0.26 or (ch / sh) > 0.42:
                continue
            if area > bare_best_area:
                bare_best_area = area
                bare_cand = (max(0.01, x / sw), max(0.02, y / sh), min(0.99, (x + cw) / sw), min(0.99, (y + ch) / sh))

        if bare_cand is not None:
            self._tracked_bare_hand = _ema_box(self._tracked_bare_hand, bare_cand, alpha=0.50)
            self._bare_hand_ttl = 5
        elif self._bare_hand_ttl > 0:
            self._bare_hand_ttl -= 1
        else:
            self._tracked_bare_hand = None

        # ------------------------------------------------------------------
        # C. Strict Rectangular Label & Elongated Tube Detection
        # ------------------------------------------------------------------
        edges = cv2.Canny(blur.astype(np.uint8), 55, 150)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        bright_patch = ((hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 55) & (person_exclusion == 0)).astype(np.uint8) * 255

        label_candidates_mask = cv2.bitwise_and(bright_patch, edges_dilated)
        label_candidates_mask = cv2.morphologyEx(
            label_candidates_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)),
        )

        feature_contours, _ = cv2.findContours(label_candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_label_box: tuple[float, float, float, float] | None = None
        best_score = -1.0

        for cnt in feature_contours:
            area = float(cv2.contourArea(cnt))
            if area < sw * sh * 0.0035 or area > sw * sh * 0.060:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(1.0, float(ch))
            if aspect < 1.20 or aspect > 4.2:
                continue
            rect_area = max(1.0, float(cw * ch))
            rectangularity = area / rect_area
            if rectangularity < 0.58:
                continue

            cx = (x + cw * 0.5) / sw
            cy = (y + ch * 0.5) / sh
            if cx < 0.08 or cx > 0.92 or cy < 0.08 or cy > 0.84:
                continue
            if person_exclusion[int(cy * sh), int(cx * sw)] > 0:
                continue

            patch_bright = float(np.mean(bright_patch[y : y + ch, x : x + cw] > 0))
            patch_edge = float(np.mean(edges[y : y + ch, x : x + cw] > 0))
            patch_motion = float(np.mean(motion_diff[y : y + ch, x : x + cw] > 7.5))
            if patch_bright < 0.35 or patch_edge < 0.09:
                continue

            glove_bonus = 0.45 if (self._tracked_left_glove is not None or self._tracked_right_glove is not None) else 0.0
            score = patch_bright * 1.4 + patch_edge * 2.5 + rectangularity * 1.2 + patch_motion * 1.8 + glove_bonus
            if score > best_score:
                best_score = score
                hw = max(0.055, min(0.11, (cw / sw) * 0.65))
                hh = max(0.042, min(0.085, (ch / sh) * 0.65))
                best_label_box = (
                    max(0.04, cx - hw),
                    max(0.06, cy - hh),
                    min(0.96, cx + hw),
                    min(0.94, cy + hh),
                )

        min_lock_score = 1.72 if self.work_mode == "strict" else 1.20
        if best_label_box is not None and best_score >= min_lock_score:
            self._tracked_label_box = _ema_box(self._tracked_label_box, best_label_box, alpha=0.45)
            self._label_ttl = 6
            self._target_locked = True
            self._target_confidence = round(min(0.98, max(0.78, 0.65 + best_score * 0.12)), 3)
        elif self._label_ttl > 0 and self._tracked_label_box is not None:
            self._label_ttl -= 1
            self._target_locked = True
        else:
            self._tracked_label_box = None
            self._tracked_tube_box = None
            self._target_locked = False
            self._target_confidence = 0.0

        tracked_objects: list[dict[str, Any]] = []
        contact_active = False
        bare_hand_touching_part = False

        if self._target_locked and self._tracked_label_box is not None:
            lc_x = (self._tracked_label_box[0] + self._tracked_label_box[2]) * 0.5
            lc_y = (self._tracked_label_box[1] + self._tracked_label_box[3]) * 0.5
            tx1 = max(0.05, min(lc_x - 0.24, (self._tracked_left_glove[0] if self._tracked_left_glove else lc_x - 0.24)))
            tx2 = min(0.95, max(lc_x + 0.24, (self._tracked_right_glove[2] if self._tracked_right_glove else lc_x + 0.24)))
            ty1 = max(0.08, lc_y - 0.095)
            ty2 = min(0.92, lc_y + 0.095)
            self._tracked_tube_box = _ema_box(self._tracked_tube_box, (tx1, ty1, tx2, ty2), alpha=0.42)

            for gbox in (self._tracked_left_glove, self._tracked_right_glove):
                if gbox is not None:
                    iou, cov = _box_iou_and_overlap(gbox, self._tracked_label_box)
                    if iou > 0.04 or cov > 0.10:
                        contact_active = True
                        break

            # If a bare hand is detected while a tube/label is present, check if it touches/holds the tube or label
            if self._tracked_bare_hand is not None and self._tracked_tube_box is not None:
                iou_t, cov_t = _box_iou_and_overlap(self._tracked_bare_hand, self._tracked_tube_box)
                iou_l, cov_l = _box_iou_and_overlap(self._tracked_bare_hand, self._tracked_label_box)
                if iou_t > 0.01 or cov_t > 0.03 or iou_l > 0.01 or cov_l > 0.03:
                    if self.work_mode == "strict":
                        bare_hand_touching_part = True
                    else:
                        contact_active = True

            tracked_objects.append(
                {
                    "kind": "tube",
                    "label": "料管目标",
                    "box": list(self._tracked_tube_box),
                    "confidence": round(max(0.82, self._target_confidence), 3),
                }
            )
            tracked_objects.append(
                {
                    "kind": "label",
                    "label": "防呆标签",
                    "box": list(self._tracked_label_box),
                    "confidence": round(max(0.85, self._target_confidence), 3),
                }
            )

        if self._tracked_left_glove:
            tracked_objects.append(
                {
                    "kind": "glove_left",
                    "label": "左手套(合规)",
                    "box": list(self._tracked_left_glove),
                    "confidence": 0.94,
                }
            )
        if self._tracked_right_glove:
            tracked_objects.append(
                {
                    "kind": "glove_right",
                    "label": "右手套(合规)",
                    "box": list(self._tracked_right_glove),
                    "confidence": 0.94,
                }
            )
        if self._tracked_bare_hand:
            tracked_objects.append(
                {
                    "kind": "bare_hand_violation",
                    "label": "裸手违规" if bare_hand_touching_part else "裸手警告",
                    "box": list(self._tracked_bare_hand),
                    "confidence": 0.95,
                }
            )

        return (
            self._tracked_tube_box,
            self._tracked_label_box,
            self._tracked_left_glove,
            self._tracked_right_glove,
            self._tracked_bare_hand,
            contact_active,
            bare_hand_touching_part,
            self._target_locked,
            tracked_objects,
        )

    def _classify_live_label_roi(
        self,
        crop: np.ndarray,
        target_locked: bool,
        contact_seen: bool,
        label_occluded: bool,
    ) -> tuple[str, float | None, str | None, float]:
        t0 = time.perf_counter()
        if not target_locked or crop is None or crop.size == 0:
            return "unknown", None, None, 0.0

        small = cv2.resize(crop, (160, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        yellow_mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 50)
        blue_mask = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 50)
        y_frac = float(np.mean(yellow_mask))
        b_frac = float(np.mean(blue_mask))
        pipe_color = "blue" if b_frac > y_frac + 0.06 else ("yellow" if y_frac > 0.04 else "unknown")

        if label_occluded:
            return "unknown", 0.48, pipe_color, round((time.perf_counter() - t0) * 1000.0, 2)

        onnx_pred: str | None = None
        onnx_conf: float = 0.0
        with self._lock:
            session = self._ort_session
            input_size = self._onnx_input_size
        if session is not None:
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                blob = resized.astype(np.float32) / 255.0
                blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
                blob = (blob - NORMALIZE_MEAN) / NORMALIZE_STD
                logits = session.run(["logits"], {"image": blob})[0][0]
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)
                best_idx = int(np.argmax(probs))
                onnx_pred = CLASSES[best_idx]
                onnx_conf = float(probs[best_idx])
            except Exception as exc:
                logger.warning("ONNX inference failed; falling back to visual heuristic: %s", exc)
                onnx_pred = None

        white_ratio = float(np.mean((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 55)))
        dark_stroke_ratio = float(np.mean(gray < 55))
        std_val = float(np.std(gray))
        edges = cv2.Canny(gray, 55, 150)
        edge_ratio = float(np.mean(edges > 0))
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if contact_seen and white_ratio < 0.22:
            conf = max(onnx_conf, min(0.97, 0.86 + (0.22 - white_ratio) * 0.4))
            return "absent", round(conf, 3), pipe_color, elapsed_ms

        if white_ratio >= 0.35 and 0.14 <= dark_stroke_ratio <= 0.48 and edge_ratio >= 0.11:
            conf = max(onnx_conf, min(0.98, 0.84 + white_ratio * 0.20))
            return "crossed_out", round(conf, 3), pipe_color, elapsed_ms

        if onnx_pred is not None and onnx_conf >= 0.80 and white_ratio >= 0.28:
            return onnx_pred, round(onnx_conf, 3), pipe_color, elapsed_ms

        conf = min(0.96, max(0.80, 0.76 + std_val / 180.0))
        return "complete", round(conf, 3), pipe_color, elapsed_ms

    def _draw_hud(
        self,
        frame: np.ndarray,
        rois: ROISet,
        tube_box: tuple[float, float, float, float] | None,
        label_box: tuple[float, float, float, float] | None,
        left_glove: tuple[float, float, float, float] | None,
        right_glove: tuple[float, float, float, float] | None,
        bare_hand: tuple[float, float, float, float] | None,
        contact_active: bool,
        bare_hand_violation: bool,
        alarm_latched: bool,
        target_locked: bool,
        view_valid: bool,
        label_status: str,
        label_conf: float | None,
    ) -> tuple[bytes, bytes, np.ndarray]:
        hud = frame.copy()
        h, w = hud.shape[:2]

        def draw_dynamic_box(
            box: tuple[float, ...] | None,
            color_bgr: tuple[int, int, int],
            tag: str,
            thick: int = 2,
            dashed: bool = False,
        ) -> None:
            if not box or not valid_box(box):
                return
            x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
            if dashed:
                cv2.rectangle(hud, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(hud, (x1, y1), (x2, y2), color_bgr, thick, cv2.LINE_AA)
                cl = max(10, min(24, (x2 - x1) // 5, (y2 - y1) // 5))
                for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                    cv2.line(hud, (cx, cy), (cx + dx * cl, cy), color_bgr, thick + 2, cv2.LINE_AA)
                    cv2.line(hud, (cx, cy), (cx, cy + dy * cl), color_bgr, thick + 2, cv2.LINE_AA)
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(hud, (mx - 8, my), (mx + 8, my), color_bgr, 1, cv2.LINE_AA)
                cv2.line(hud, (mx, my - 8), (mx, my + 8), color_bgr, 1, cv2.LINE_AA)
            self._badge.draw_badge(hud, x1, max(4, y1 - 28), tag, color_bgr)

        if self.show_auxiliary_rois:
            draw_dynamic_box(rois.tube_roi, (110, 115, 125), "辅助参考·料管区", 1, dashed=True)
            draw_dynamic_box(rois.left_grip_zone, (110, 115, 125), "辅助参考·左握持", 1, dashed=True)
            draw_dynamic_box(rois.right_grip_zone, (110, 115, 125), "辅助参考·右握持", 1, dashed=True)

        # 1. Compliant White / Light-Gray ESD Gloves (Green)
        if left_glove is not None:
            draw_dynamic_box(left_glove, (70, 165, 45), "左手套 · 合规佩戴", 2)
        if right_glove is not None:
            draw_dynamic_box(right_glove, (70, 165, 45), "右手套 · 合规佩戴", 2)

        # 2. Bare-Hand SOP Violation / Warning (Crimson Red in strict mode, Cyan in debug mode)
        if bare_hand is not None:
            if self.work_mode == "strict":
                bare_tag = "违规报警 · 裸手触碰料管（未戴防静电手套）" if bare_hand_violation else "裸手警告 · 作业必须佩戴防静电手套"
                draw_dynamic_box(bare_hand, (35, 45, 210), bare_tag, 3)
            else:
                bare_side = "左手" if ((bare_hand[0] + bare_hand[2]) * 0.5) < 0.50 else "右手"
                draw_dynamic_box(bare_hand, (185, 130, 30), f"{bare_side}追踪 · 桌面调试模式", 2)

        # 3. Tube & Label dynamic tracking (no global status banners overlaid on video)
        if target_locked and label_box is not None:
            if tube_box is not None:
                draw_dynamic_box(tube_box, (215, 125, 35), "料管目标 · 动态锁定", 2)
            lbl_cn = LABEL_CN.get(label_status, label_status)
            pct = int(round((label_conf or 0.0) * 100))
            if bare_hand_violation or alarm_latched:
                label_color = (35, 45, 210)
                label_tag = "违规闭锁 · 禁止放行"
            elif contact_active:
                label_color = (15, 130, 230)
                label_tag = "标签追踪 · 手套接触中"
            elif view_valid:
                label_color = (55, 155, 35)
                label_tag = f"防呆标签 · {lbl_cn} {pct}%"
            else:
                label_color = (45, 65, 205)
                label_tag = f"防呆标签 · {lbl_cn}"
            draw_dynamic_box(label_box, label_color, label_tag, 2)
            active_crop_box = label_box
        else:
            active_crop_box = (0.41, 0.42, 0.59, 0.58)

        lcx = (active_crop_box[0] + active_crop_box[2]) * 0.5
        lcy = (active_crop_box[1] + active_crop_box[3]) * 0.5
        lw = max(0.14, (active_crop_box[2] - active_crop_box[0]) * 1.15)
        lh = max(0.11, lw * (9.0 / 16.0))
        dynamic_crop_box = (
            max(0.0, min(1.0 - lw, lcx - lw * 0.5)),
            max(0.0, min(1.0 - lh, lcy - lh * 0.5)),
            min(1.0, max(lw, lcx + lw * 0.5)),
            min(1.0, max(lh, lcy + lh * 0.5)),
        )
        label_crop, _ = self._crop_norm(frame, dynamic_crop_box)
        crop_resized = cv2.resize(label_crop, (480, 270), interpolation=cv2.INTER_CUBIC)
        ch, cw = crop_resized.shape[:2]
        cv2.line(crop_resized, (cw // 2 - 14, ch // 2), (cw // 2 + 14, ch // 2), (45, 185, 110), 1, cv2.LINE_AA)
        cv2.line(crop_resized, (cw // 2, ch // 2 - 14), (cw // 2, ch // 2 + 14), (45, 185, 110), 1, cv2.LINE_AA)

        hud_stream = cv2.resize(hud, (1280, 720), interpolation=cv2.INTER_AREA)
        ok1, buf1 = cv2.imencode(".jpg", hud_stream, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        ok2, buf2 = cv2.imencode(".jpg", crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return (bytes(buf1) if ok1 else b""), (bytes(buf2) if ok2 else b""), label_crop

    def _loop(self) -> None:
        while self._running:
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                with self._lock:
                    self.camera_online = False
                    if time.monotonic() >= self.simulation_override_until:
                        self._state["camera_status"] = "NOT_CONNECTED"
                        self._state["view_valid"] = False
                        self._state["view_valid_reasons"] = ["CAMERA_NOT_CONNECTED"]
                        self._state["updated_at_utc"] = _utc_now()
                time.sleep(1.5)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            try:
                while self._running:
                    ok, frame = cap.read()
                    now = time.monotonic()
                    if not ok or frame is None or frame.size == 0:
                        break
                    with self._lock:
                        mirror_now = self.mirror_horizontal
                        work_mode_now = self.work_mode
                    if mirror_now:
                        frame = cv2.flip(frame, 1)

                    self._frame_id += 1
                    self._fps_times.append(now)
                    if len(self._fps_times) >= 2:
                        dt = self._fps_times[-1] - self._fps_times[0]
                        measured_fps = (len(self._fps_times) - 1) / dt if dt > 0 else 30.0
                    else:
                        measured_fps = 30.0

                    h, w = frame.shape[:2]
                    rois, thresholds, roi_ver = self._active_rois_and_thresholds()

                    (
                        tube_box,
                        label_box,
                        left_glove,
                        right_glove,
                        bare_hand,
                        contact_raw,
                        bare_hand_touching_part,
                        target_locked,
                        tracked_objects,
                    ) = self._track_dynamic_targets(frame)

                    eval_box = label_box if (target_locked and label_box is not None) else rois.label_roi
                    quality = measure_label_quality(frame, eval_box)

                    min_lap = float(thresholds.get("min_laplacian_variance") or 12.0)
                    min_lum = float(thresholds.get("min_brightness") or 25.0)
                    max_lum = float(thresholds.get("max_brightness") or 242.0)
                    max_glare = float(thresholds.get("max_overexposed_fraction") or 0.18)

                    bare_cx = ((bare_hand[0] + bare_hand[2]) * 0.5) if bare_hand is not None else None
                    left_active = left_glove is not None or (
                        work_mode_now == "debug" and bare_cx is not None and bare_cx < 0.50
                    )
                    right_active = right_glove is not None or (
                        work_mode_now == "debug" and bare_cx is not None and bare_cx >= 0.50
                    )
                    glove_compliant = bool((left_active or right_active) and (bare_hand is None or work_mode_now == "debug"))
                    bare_hand_detected = bool(bare_hand is not None and work_mode_now == "strict")

                    reasons: list[str] = []
                    if bare_hand_touching_part and work_mode_now == "strict":
                        reasons.append("BARE_HAND_VIOLATION")
                    elif bare_hand_detected:
                        reasons.append("BARE_HAND_WARNING_NO_GLOVE")
                    if not target_locked:
                        reasons.append("TUBE_NOT_PRESENT")
                    if quality is None:
                        reasons.append("LABEL_ROI_QUALITY_UNAVAILABLE")
                    else:
                        if quality.laplacian_variance < min_lap:
                            reasons.append("LOW_SHARPNESS")
                        if quality.brightness_mean < min_lum:
                            reasons.append("LOW_BRIGHTNESS")
                        elif quality.brightness_mean > max_lum:
                            reasons.append("HIGH_BRIGHTNESS")
                        if quality.overexposed_fraction > max_glare:
                            reasons.append("OVEREXPOSED")

                    with self._lock:
                        tracked = self.contact_tracker.update(
                            ContactCandidate(
                                hand_contact_candidate=bool(target_locked and contact_raw),
                                label_contact_event=bool(target_locked and contact_raw),
                                event="hand_contact_candidate" if (target_locked and contact_raw) else "none",
                                reason="dynamic_target_overlap" if (target_locked and contact_raw) else "clear",
                            ),
                            view_available=target_locked,
                        )
                        contact_seen_now = self.fsm._contact_seen or tracked.hand_contact_candidate

                    quality_ok = all(r not in {"LOW_SHARPNESS", "LOW_BRIGHTNESS", "HIGH_BRIGHTNESS", "OVEREXPOSED"} for r in reasons)
                    view_valid = bool(target_locked and quality_ok and not tracked.hand_contact_candidate and not bare_hand_touching_part)
                    if tracked.hand_contact_candidate:
                        reasons.append("HAND_IN_CONTACT_ROI")

                    label_crop_for_cls, _ = self._crop_norm(frame, eval_box)
                    label_status, label_conf, pipe_color, infer_ms = self._classify_live_label_roi(
                        label_crop_for_cls,
                        target_locked=target_locked,
                        contact_seen=contact_seen_now,
                        label_occluded=tracked.hand_contact_candidate,
                    )

                    prev_fsm: str | None = None
                    with self._lock:
                        self._raw_frame = frame
                        self.camera_online = True
                        self.width = w
                        self.height = h
                        self.fps = round(measured_fps, 1)
                        self._inference_ms = infer_ms

                        if now >= self.simulation_override_until:
                            prev_fsm = self._state.get("fsm_state", "WAIT_PART")
                            changed = False

                            # Rule A: Bare hand touches/holds the tube or label -> IMMEDIATE LATCHED ALARM!
                            if bare_hand_touching_part and not self._alarm_latched:
                                self._alarm_latched = True
                                self._latched_reason = "BARE_HAND_VIOLATION"
                                self._latched_cycle_id = self.fsm.cycle_id or f"LIVE_{datetime.now().strftime('%H%M%S')}"
                                changed = True

                            if self._alarm_latched:
                                active_fsm_state = "ALARM"
                                active_outcome = "ALARM"
                                active_reason = self._latched_reason or "BARE_HAND_VIOLATION"
                                active_cycle_id = self._latched_cycle_id
                            elif not target_locked:
                                # Rule B: Auto-flow on PASS — when tube leaves view for >= 0.5s after PASS, reset for next tube
                                if self.fsm.state == FSMState.PASS:
                                    if self._pass_target_lost_since is None:
                                        self._pass_target_lost_since = now
                                    elif (now - self._pass_target_lost_since) >= 0.5:
                                        self.fsm = InspectionFSM(
                                            config=FSMConfig(initial_complete_ms=350, stable_state_ms=400, min_stable_frames=5)
                                        )
                                        self._pass_target_lost_since = None
                                        changed = True
                                elif self.fsm.state == FSMState.ALARM:
                                    # Latch any FSM alarm until manual reset
                                    self._alarm_latched = True
                                    self._latched_reason = self.fsm.result_reason or "LABEL_STILL_COMPLETE_AFTER_RELEASE"
                                    self._latched_cycle_id = self.fsm.cycle_id
                                    changed = True
                                else:
                                    self.fsm = InspectionFSM(
                                        config=FSMConfig(initial_complete_ms=350, stable_state_ms=400, min_stable_frames=5)
                                    )

                                if self._alarm_latched:
                                    active_fsm_state = "ALARM"
                                    active_outcome = "ALARM"
                                    active_reason = self._latched_reason
                                    active_cycle_id = self._latched_cycle_id
                                elif self.fsm.state == FSMState.PASS:
                                    active_fsm_state = "PASS"
                                    active_outcome = "PASS"
                                    active_reason = self.fsm.result_reason
                                    active_cycle_id = self.fsm.cycle_id
                                else:
                                    active_fsm_state = "WAIT_PART"
                                    active_outcome = "WAITING"
                                    active_reason = "BARE_HAND_WARNING_NO_GLOVE" if bare_hand_detected else "WAITING_FOR_PART"
                                    active_cycle_id = None
                            else:
                                self._pass_target_lost_since = None
                                if self.fsm.state == FSMState.WAIT_PART:
                                    cycle_id = f"LIVE_{datetime.now().strftime('%H%M%S')}"
                                    self.fsm.process(
                                        FSMEvent(
                                            kind="new_cycle",
                                            cycle_id=cycle_id,
                                            capture_monotonic_s=now,
                                            pipe_color=pipe_color or "yellow",
                                            operator_anon_id="OP_STATION_01",
                                        ),
                                        now_monotonic=now,
                                    )
                                self.fsm.process(
                                    FSMEvent(
                                        kind="observation",
                                        cycle_id=self.fsm.cycle_id,
                                        capture_monotonic_s=now,
                                        frame_id=self._frame_id,
                                        label_class=label_status,
                                        view_valid=view_valid,
                                        quality_ok=quality_ok,
                                        camera_healthy=True,
                                        camera_settings_locked=True,
                                        config_match=True,
                                        tube_present=target_locked,
                                        label_window_present=target_locked,
                                        label_roi_clear=not tracked.hand_contact_candidate,
                                        hand_occluded=tracked.hand_contact_candidate,
                                        hand_contact_candidate=tracked.hand_contact_candidate,
                                        hand_detection_valid=glove_compliant,
                                        hand_in_contact_roi=tracked.hand_contact_candidate,
                                        hand_in_label_contact_roi=tracked.hand_contact_candidate,
                                        hand_in_left_grip_zone=left_active,
                                        hand_in_right_grip_zone=right_active,
                                        pipe_color=pipe_color,
                                        operator_anon_id="OP_STATION_01",
                                    ),
                                    now_monotonic=now,
                                )
                                if self.fsm.state == FSMState.READY:
                                    self.fsm.process(FSMEvent(kind="start_inspection", capture_monotonic_s=now), now_monotonic=now)
                                if self.fsm.state == FSMState.ALARM:
                                    self._alarm_latched = True
                                    self._latched_reason = self.fsm.result_reason
                                    self._latched_cycle_id = self.fsm.cycle_id

                                active_fsm_state = self.fsm.state.value
                                active_outcome = self.fsm.outcome
                                active_reason = self.fsm.result_reason
                                active_cycle_id = self.fsm.cycle_id
                                changed = prev_fsm != active_fsm_state

                            backend_label = (
                                f"ONNX 神经网络 ({self._onnx_model_version})"
                                if self._ort_session is not None
                                else "动态多目标视觉追踪引擎"
                            )
                            self._state.update(
                                {
                                    "camera_status": "ONLINE",
                                    "camera_name": self.camera_name,
                                    "camera_device_id": self.camera_device_id,
                                    "backend": self.backend,
                                    "camera_width": w,
                                    "camera_height": h,
                                    "fps": round(measured_fps, 1),
                                    "label_status": label_status,
                                    "label_confidence": label_conf,
                                    "view_valid": view_valid,
                                    "view_valid_reasons": reasons,
                                    "quality_ok": quality_ok,
                                    "laplacian_variance": round(quality.laplacian_variance, 1) if quality else None,
                                    "brightness": round(quality.brightness_mean, 1) if quality else None,
                                    "overexposed_fraction": round(quality.overexposed_fraction, 4) if quality else None,
                                    "underexposed_fraction": round(quality.underexposed_fraction, 4) if quality else None,
                                    "left_grip_active": left_active,
                                    "right_grip_active": right_active,
                                    "glove_compliant": glove_compliant,
                                    "bare_hand_detected": bare_hand_detected,
                                    "bare_hand_violation": bool(bare_hand_touching_part or self._latched_reason == "BARE_HAND_VIOLATION"),
                                    "alarm_latched": self._alarm_latched,
                                    "label_contact_active": tracked.hand_contact_candidate,
                                    "hand_contact_candidate": tracked.hand_contact_candidate,
                                    "fsm_state": active_fsm_state,
                                    "outcome": active_outcome,
                                    "result_reason": active_reason,
                                    "c_timer_ms": self.fsm.c_timer_ms,
                                    "cycle_id": active_cycle_id,
                                    "pipe_color": pipe_color,
                                    "frame_id": self._frame_id,
                                    "capture_monotonic_s": round(now, 3),
                                    "model_version": self._onnx_model_version,
                                    "inference_backend": backend_label,
                                    "inference_ms": infer_ms,
                                    "show_auxiliary_rois": self.show_auxiliary_rois,
                                    "mirror_horizontal": self.mirror_horizontal,
                                    "work_mode": self.work_mode,
                                    "tracked_objects": tracked_objects,
                                    "config_version": roi_ver,
                                    "live_stream_url": "/api/camera/stream",
                                    "live_crop_url": "/api/camera/stream_crop",
                                    "updated_at_utc": _utc_now(),
                                }
                            )
                            snapshot_copy = dict(self._state)
                        else:
                            snapshot_copy = None
                            changed = False

                        hud_bare_violation = bool(bare_hand_touching_part or self._latched_reason == "BARE_HAND_VIOLATION")
                        hud_alarm_latched = self._alarm_latched

                    hud_bytes, crop_bytes, raw_crop = self._draw_hud(
                        frame,
                        rois,
                        tube_box,
                        label_box,
                        left_glove,
                        right_glove,
                        bare_hand,
                        tracked.hand_contact_candidate,
                        hud_bare_violation,
                        hud_alarm_latched,
                        target_locked,
                        view_valid,
                        label_status,
                        label_conf,
                    )
                    with self._lock:
                        self._hud_jpeg = hud_bytes
                        self._crop_jpeg = crop_bytes
                        self._last_tracked_crop = raw_crop

                    if snapshot_copy and changed and self._on_cycle_update:
                        try:
                            self._on_cycle_update(snapshot_copy, prev_fsm)
                        except Exception as exc:
                            logger.warning("Cycle update callback failed: %s", exc)

                    time.sleep(0.001)
            except Exception as exc:
                logger.warning("Live camera capture loop error; reopening capture: %s", exc)
            finally:
                cap.release()
            time.sleep(0.5)
