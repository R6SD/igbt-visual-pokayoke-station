"""Integrated station runtime bridging ViewValidEvaluator, ContactCandidateTracker, and InspectionFSM."""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from igbt_pokayoke.logic.fsm import FSMConfig, FSMEvent, FSMState, InspectionFSM
from igbt_pokayoke.vision.hand_contact import (
    ContactCandidate,
    ContactCandidateTracker,
    ContactConfig,
    detect_contact_candidate,
    overlap_of_box_fraction,
)
from igbt_pokayoke.vision.view_valid import (
    FrameObservation,
    QualityMetrics,
    ROISet,
    ViewConfig,
    ViewResult,
    ViewValidEvaluator,
    measure_label_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = PROJECT_ROOT / "assets"
YELLOW_REF_IMAGE = ASSETS_ROOT / "01_黄管_站立持管检测框参考图.jpg"
BLUE_REF_IMAGE = ASSETS_ROOT / "02_蓝管_站立持管检测框参考图.jpg"

ScenarioName = Literal["a_pass", "b_pass", "c_pass", "alarm", "glare_invalid", "reset"]
SCENARIO_NAMES: tuple[ScenarioName, ...] = (
    "a_pass",
    "b_pass",
    "c_pass",
    "alarm",
    "glare_invalid",
    "reset",
)

STANDARD_ROIS = ROISet(
    tube_roi=(0.16, 0.42, 0.84, 0.58),
    label_roi=(0.43, 0.45, 0.57, 0.55),
    label_contact_roi=(0.40, 0.38, 0.60, 0.62),
    left_grip_zone=(0.16, 0.34, 0.31, 0.82),
    right_grip_zone=(0.69, 0.34, 0.84, 0.82),
)

QUALITY_FAILURE_REASONS = {
    "QUALITY_THRESHOLDS_NOT_CALIBRATED",
    "LABEL_ROI_QUALITY_UNAVAILABLE",
    "LABEL_ROI_BRIGHTNESS_OUT_OF_RANGE",
    "LABEL_ROI_OVEREXPOSED",
    "LABEL_ROI_UNDEREXPOSED",
    "LABEL_ROI_BLURRED",
}


def load_image_unicode(path: Path) -> np.ndarray:
    """Decode an image from disk safely across Windows non-ASCII paths."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Unable to decode reference image: {path}")
    return image


def default_view_config(
    width: int = 1408,
    height: int = 768,
    camera_device_id: str = "CAM_STATION_01",
    roi_config_version: str = "ROI_V001",
    rois: ROISet = STANDARD_ROIS,
) -> ViewConfig:
    return ViewConfig(
        camera_device_id=camera_device_id,
        calibrated_width=width,
        calibrated_height=height,
        roi_config_version=roi_config_version,
        rois=rois,
        quality_thresholds_approved=True,
        orientation_detector_ready=True,
        min_brightness=55.0,
        max_brightness=225.0,
        max_overexposed_fraction=0.08,
        max_underexposed_fraction=0.15,
        min_laplacian_variance=120.0,
        max_frame_gap_ms=200.0,
        stable_window_ms=400.0,
        min_stable_frames=5,
    )


class StationRuntime:
    """Cohesive runtime pipeline combining optical quality, view gate, contact tracker, and FSM."""

    def __init__(
        self,
        view_config: ViewConfig | None = None,
        fsm_config: FSMConfig | None = None,
        contact_config: ContactConfig | None = None,
        enter_frames: int = 2,
        exit_frames: int = 2,
        clock: Callable[[], float] = time.monotonic,
        camera_name: str = "Basler acA1920-40uc",
        backend: str = "DirectShow",
        fps: float = 30.0,
        model_version: str = "mobilenetv3-igbt-v1.2.0",
        config_version: str = "ROI_V001",
        operator_anon_id: str = "OP_STATION_01",
    ) -> None:
        self.view_config = view_config or default_view_config()
        self.fsm_config = fsm_config or FSMConfig()
        self.contact_config = contact_config or ContactConfig()
        self._clock = clock
        self.view_evaluator = ViewValidEvaluator(self.view_config)
        self.contact_tracker = ContactCandidateTracker(enter_frames=enter_frames, exit_frames=exit_frames)
        self.fsm = InspectionFSM(config=self.fsm_config, clock=clock)
        self.camera_name = camera_name
        self.backend = backend
        self.fps = fps
        self.model_version = model_version
        self.config_version = config_version
        self.operator_anon_id = operator_anon_id
        self.pipe_color: Literal["yellow", "blue", "other", "unknown"] | None = "yellow"
        self._last_snapshot: dict = self._build_snapshot(
            frame_id=0,
            capture_monotonic_s=0.0,
            label_status="unknown",
            label_confidence=None,
            view_result=None,
            quality=None,
            quality_ok=False,
            left_grip_active=False,
            right_grip_active=False,
            contact=ContactCandidate(False, False, "none", "runtime_initialized"),
            camera_healthy=True,
            camera_settings_locked=True,
            event_type="RUNTIME_INIT",
        )

    def start_cycle(
        self,
        cycle_id: str,
        pipe_color: Literal["yellow", "blue", "other", "unknown"] = "yellow",
        operator_anon_id: str | None = None,
        now_monotonic: float | None = None,
    ) -> dict:
        now = self._clock() if now_monotonic is None else float(now_monotonic)
        self.pipe_color = pipe_color
        if operator_anon_id:
            self.operator_anon_id = operator_anon_id
        self.view_evaluator.reset()
        self.contact_tracker.reset()
        self.fsm.process(
            FSMEvent(
                kind="new_cycle",
                cycle_id=cycle_id,
                capture_monotonic_s=now,
                pipe_color=pipe_color,
                operator_anon_id=self.operator_anon_id,
            ),
            now_monotonic=now,
        )
        self._last_snapshot = {
            **self._last_snapshot,
            "fsm_state": self.fsm.state.value,
            "outcome": self.fsm.outcome,
            "result_reason": self.fsm.result_reason,
            "c_timer_ms": self.fsm.c_timer_ms,
            "cycle_id": self.fsm.cycle_id,
            "pipe_color": self.pipe_color,
            "operator_anon_id": self.operator_anon_id,
            "event_type": "NEW_CYCLE",
            "capture_monotonic_s": round(now, 4),
        }
        return dict(self._last_snapshot)

    def start_inspection(self, now_monotonic: float | None = None) -> dict:
        now = self._clock() if now_monotonic is None else float(now_monotonic)
        self.fsm.process(FSMEvent(kind="start_inspection", capture_monotonic_s=now), now_monotonic=now)
        self._last_snapshot = {
            **self._last_snapshot,
            "fsm_state": self.fsm.state.value,
            "outcome": self.fsm.outcome,
            "result_reason": self.fsm.result_reason,
            "c_timer_ms": self.fsm.c_timer_ms,
            "event_type": "INSPECTION_STARTED",
            "capture_monotonic_s": round(now, 4),
        }
        return dict(self._last_snapshot)

    def reset(
        self,
        reason: str = "AUTHORIZED_RESET",
        authorized: bool = True,
        now_monotonic: float | None = None,
    ) -> dict:
        now = self._clock() if now_monotonic is None else float(now_monotonic)
        self.view_evaluator.reset()
        self.contact_tracker.reset()
        if self.fsm.state not in {FSMState.WAIT_PART, FSMState.PASS, FSMState.ALARM, FSMState.FAULT, FSMState.REVIEW_REQUIRED}:
            self.fsm.process(FSMEvent(kind="fault", reason="MANUAL_ABORT_BEFORE_RESET", capture_monotonic_s=now), now_monotonic=now)
        self.fsm.process(
            FSMEvent(
                kind="reset",
                reason=reason,
                reset_authorized=authorized,
                capture_monotonic_s=now,
            ),
            now_monotonic=now,
        )
        self._last_snapshot = {
            **self._last_snapshot,
            "fsm_state": self.fsm.state.value,
            "outcome": self.fsm.outcome,
            "result_reason": self.fsm.result_reason,
            "c_timer_ms": 0,
            "cycle_id": self.fsm.cycle_id,
            "hand_contact_candidate": False,
            "label_contact_active": False,
            "event_type": "AUTHORIZED_RESET",
            "capture_monotonic_s": round(now, 4),
        }
        return dict(self._last_snapshot)

    def step(
        self,
        frame: np.ndarray,
        observation: FrameObservation,
        detections: Iterable[dict] = (),
        label_status: Literal["complete", "crossed_out", "covered", "absent", "unknown"] = "unknown",
        label_confidence: float | None = None,
        pipe_color: Literal["yellow", "blue", "other", "unknown"] | None = None,
        camera_settings_locked: bool = True,
        hand_detection_valid: bool = True,
        now_monotonic: float | None = None,
    ) -> dict:
        now = float(observation.captured_monotonic_s if now_monotonic is None else now_monotonic)
        if pipe_color is not None:
            self.pipe_color = pipe_color

        det_list = list(detections)
        view_result = self.view_evaluator.evaluate(frame, observation)
        quality = view_result.quality or measure_label_quality(frame, self.view_config.rois.label_roi)
        quality_ok = bool(
            quality is not None
            and not any(r in QUALITY_FAILURE_REASONS for r in view_result.reasons)
        )

        rois = self.view_config.rois
        left_grip_active = False
        right_grip_active = False
        hand_in_contact_roi = False
        marker_detected = False

        for det in det_list:
            cls_name = str(det.get("class_name", ""))
            score = float(det.get("score", 0.0))
            box = det.get("box")
            if cls_name == "marker" and score >= 0.5:
                marker_detected = True
            if cls_name != "hand_glove" or score < self.contact_config.hand_score_threshold or not box:
                continue
            if rois.left_grip_zone and overlap_of_box_fraction(box, rois.left_grip_zone) >= self.contact_config.grip_overlap_ratio_exclusion:
                left_grip_active = True
            if rois.right_grip_zone and overlap_of_box_fraction(box, rois.right_grip_zone) >= self.contact_config.grip_overlap_ratio_exclusion:
                right_grip_active = True
            if (
                overlap_of_box_fraction(box, rois.label_contact_roi) >= self.contact_config.contact_overlap_ratio
                and not (rois.left_grip_zone and overlap_of_box_fraction(box, rois.left_grip_zone) >= self.contact_config.grip_overlap_ratio_exclusion)
                and not (rois.right_grip_zone and overlap_of_box_fraction(box, rois.right_grip_zone) >= self.contact_config.grip_overlap_ratio_exclusion)
            ):
                hand_in_contact_roi = True

        raw_candidate = detect_contact_candidate(
            det_list,
            rois.label_contact_roi,
            rois.left_grip_zone,
            rois.right_grip_zone,
            self.contact_config,
        )
        view_available = bool(
            frame is not None
            and isinstance(frame, np.ndarray)
            and frame.size > 0
            and observation.camera_healthy is True
            and hand_detection_valid
        )
        tracked_candidate = self.contact_tracker.update(raw_candidate, view_available=view_available)

        config_match = not any(
            r in {
                "CALIBRATION_RESOLUTION_MISMATCH",
                "CAMERA_DEVICE_MISMATCH",
                "ROI_CONFIG_VERSION_MISMATCH",
                "ROI_CALIBRATION_INVALID",
                "BOTH_GRIP_ZONES_NOT_CALIBRATED",
                "CONTACT_AND_GRIP_ZONES_OVERLAP",
            }
            for r in view_result.reasons
        )
        hand_occluded = bool(
            observation.label_occluded is not False
            or "DETECTED_OCCLUDER_OVERLAPS_LABEL_ROI" in view_result.reasons
        )
        label_roi_clear = not hand_occluded
        fsm_view_valid = bool(
            view_result.view_valid
            or view_result.reasons == ("STABILITY_WINDOW_NOT_MET",)
        )

        # Note: in InspectionFSM._contact_eligible, central contact requires
        # hand_in_left_grip_zone=False and hand_in_right_grip_zone=False for the contacting hand.
        # Because detect_contact_candidate already excludes grip-zone detections per box,
        # we only set hand_in_left/right_grip_zone on the FSM event when no central contact candidate is active.
        fsm_event = FSMEvent(
            kind="observation",
            cycle_id=self.fsm.cycle_id,
            capture_monotonic_s=observation.captured_monotonic_s,
            frame_id=observation.frame_id,
            label_class=label_status,
            view_valid=fsm_view_valid,
            quality_ok=quality_ok,
            camera_healthy=observation.camera_healthy,
            camera_settings_locked=camera_settings_locked,
            config_match=config_match,
            tube_present=observation.tube_present,
            label_window_present=observation.label_window_present,
            label_roi_clear=label_roi_clear,
            hand_occluded=hand_occluded,
            hand_contact_candidate=tracked_candidate.hand_contact_candidate,
            hand_detection_valid=hand_detection_valid,
            hand_in_contact_roi=hand_in_contact_roi or tracked_candidate.hand_contact_candidate,
            hand_in_label_contact_roi=hand_in_contact_roi or tracked_candidate.hand_contact_candidate,
            hand_in_left_grip_zone=False if tracked_candidate.hand_contact_candidate else left_grip_active,
            hand_in_right_grip_zone=False if tracked_candidate.hand_contact_candidate else right_grip_active,
            marker_detected=marker_detected,
            pipe_color=self.pipe_color,
            operator_anon_id=self.operator_anon_id,
        )
        before_state = self.fsm.state
        self.fsm.process(fsm_event, now_monotonic=now)
        after_state = self.fsm.state

        if tracked_candidate.label_contact_event:
            event_type = "HAND_CONTACT_CANDIDATE"
        elif before_state != after_state:
            event_type = f"FSM_{after_state.value}"
        else:
            event_type = None

        self._last_snapshot = self._build_snapshot(
            frame_id=observation.frame_id,
            capture_monotonic_s=observation.captured_monotonic_s,
            label_status=label_status,
            label_confidence=label_confidence,
            view_result=view_result,
            quality=quality,
            quality_ok=quality_ok,
            left_grip_active=left_grip_active,
            right_grip_active=right_grip_active,
            contact=tracked_candidate,
            camera_healthy=observation.camera_healthy,
            camera_settings_locked=camera_settings_locked,
            event_type=event_type,
        )
        return dict(self._last_snapshot)

    def _build_snapshot(
        self,
        *,
        frame_id: int | None,
        capture_monotonic_s: float | None,
        label_status: Literal["complete", "crossed_out", "covered", "absent", "unknown"],
        label_confidence: float | None,
        view_result: ViewResult | None,
        quality: QualityMetrics | None,
        quality_ok: bool,
        left_grip_active: bool,
        right_grip_active: bool,
        contact: ContactCandidate,
        camera_healthy: bool | None,
        camera_settings_locked: bool,
        event_type: str | None,
    ) -> dict:
        camera_status: Literal["ONLINE", "OFFLINE", "FAULT", "UNVERIFIED"] = (
            "ONLINE" if camera_healthy is True else ("FAULT" if camera_healthy is False else "UNVERIFIED")
        )
        return {
            "mode": "shadow",
            "camera_status": camera_status,
            "camera_name": self.camera_name,
            "camera_device_id": self.view_config.camera_device_id,
            "backend": self.backend,
            "camera_width": self.view_config.calibrated_width,
            "camera_height": self.view_config.calibrated_height,
            "camera_settings_locked": camera_settings_locked,
            "fps": self.fps,
            "label_status": label_status,
            "label_confidence": round(float(label_confidence), 4) if label_confidence is not None else None,
            "view_valid": view_result.view_valid if view_result is not None else False,
            "view_valid_reasons": list(view_result.reasons) if view_result is not None else [],
            "quality_ok": quality_ok,
            "laplacian_variance": round(quality.laplacian_variance, 2) if quality is not None else None,
            "brightness": round(quality.brightness_mean, 2) if quality is not None else None,
            "overexposed_fraction": round(quality.overexposed_fraction, 4) if quality is not None else None,
            "underexposed_fraction": round(quality.underexposed_fraction, 4) if quality is not None else None,
            "left_grip_active": left_grip_active,
            "right_grip_active": right_grip_active,
            "label_contact_active": contact.hand_contact_candidate,
            "hand_contact_candidate": contact.hand_contact_candidate,
            "contact_evidence_type": "2d_visual_proximity_proxy",
            "proves_physical_contact": False,
            "fsm_state": self.fsm.state.value,
            "outcome": self.fsm.outcome,
            "result_reason": self.fsm.result_reason,
            "c_timer_ms": self.fsm.c_timer_ms,
            "cycle_id": self.fsm.cycle_id,
            "pipe_color": self.pipe_color,
            "operator_anon_id": self.operator_anon_id,
            "event_type": event_type,
            "frame_id": frame_id,
            "capture_monotonic_s": round(float(capture_monotonic_s), 4) if capture_monotonic_s is not None else None,
            "model_version": self.model_version,
            "config_version": self.config_version,
        }


def _synthesize_glare_frame(frame: np.ndarray, label_roi: tuple[float, float, float, float]) -> np.ndarray:
    """Create a copy of the frame with realistic specular glare inside the label ROI."""
    glare_frame = frame.copy()
    height, width = glare_frame.shape[:2]
    x1, y1 = int(label_roi[0] * width), int(label_roi[1] * height)
    x2, y2 = int(np.ceil(label_roi[2] * width)), int(np.ceil(label_roi[3] * height))
    # Saturate top-half of the label ROI to simulate harsh tube reflection (>0.08 overexposed fraction)
    glare_y2 = y1 + max(4, int((y2 - y1) * 0.45))
    glare_frame[y1:glare_y2, x1:x2] = 255
    return glare_frame


def build_scenario_run(scenario: str) -> dict:
    """Execute a deterministic end-to-end FSM + OpenCV optical pipeline scenario on real reference frames."""
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"Unsupported simulation scenario: {scenario}")

    use_blue = scenario in {"b_pass", "glare_invalid"}
    ref_path = BLUE_REF_IMAGE if use_blue else YELLOW_REF_IMAGE
    pipe_color: Literal["yellow", "blue"] = "blue" if use_blue else "yellow"
    base_frame = load_image_unicode(ref_path)
    height, width = base_frame.shape[:2]

    now = 1000.0
    frame_id = 0
    view_cfg = default_view_config(width=width, height=height)
    fsm_cfg = FSMConfig(
        initial_complete_ms=400,
        stable_state_ms=400,
        min_stable_frames=5,
        c_absent_duration_ms=1200,
        max_frame_gap_ms=200,
        hand_leave_stable_ms=200,
    )
    runtime = StationRuntime(
        view_config=view_cfg,
        fsm_config=fsm_cfg,
        clock=lambda: now,
    )

    grip_detections = [
        {"class_name": "hand_glove", "score": 0.96, "box": (0.18, 0.42, 0.29, 0.74)},
        {"class_name": "hand_glove", "score": 0.95, "box": (0.71, 0.42, 0.82, 0.74)},
    ]
    central_contact_detections = [
        *grip_detections,
        {"class_name": "hand_glove", "score": 0.94, "box": (0.43, 0.42, 0.57, 0.58)},
    ]

    def next_obs(
        *,
        dt: float = 0.1,
        label_occluded: bool = False,
        occluders: tuple[tuple[float, float, float, float], ...] = (),
    ) -> FrameObservation:
        nonlocal now, frame_id
        now += dt
        frame_id += 1
        return FrameObservation(
            frame_id=frame_id,
            captured_monotonic_s=now,
            camera_device_id=view_cfg.camera_device_id,
            roi_config_version=view_cfg.roi_config_version,
            camera_healthy=True,
            tube_present=True,
            label_window_present=True,
            label_occluded=label_occluded,
            front_face_confirmed=True,
            orientation_reference_visible=True,
            orientation_evidence_source="machine_reference",
            tube_center_norm=(0.50, 0.50),
            label_center_norm=(0.50, 0.50),
            occluder_boxes_norm=occluders,
        )

    suffix = uuid.uuid4().hex[:6].upper()
    prefix_map = {
        "a_pass": "CYC_A",
        "b_pass": "CYC_B",
        "c_pass": "CYC_C",
        "alarm": "CYC_ALM",
        "glare_invalid": "CYC_GLR",
        "reset": "CYC_RST",
    }
    cycle_id = f"{prefix_map[scenario]}_{suffix}"

    if scenario == "reset":
        runtime.start_cycle(cycle_id=cycle_id, pipe_color=pipe_color, now_monotonic=now)
        obs = next_obs()
        runtime.step(
            base_frame,
            obs,
            detections=grip_detections,
            label_status="absent",
            label_confidence=0.93,
            pipe_color=pipe_color,
            now_monotonic=now,
        )
        now += 0.1
        snapshot = runtime.reset(reason="AUTHORIZED_RESET", authorized=True, now_monotonic=now)
        # Compute live optical metrics on the reference frame so gauges remain informative after reset
        quality = measure_label_quality(base_frame, view_cfg.rois.label_roi)
        if quality is not None:
            snapshot["laplacian_variance"] = round(quality.laplacian_variance, 2)
            snapshot["brightness"] = round(quality.brightness_mean, 2)
            snapshot["overexposed_fraction"] = round(quality.overexposed_fraction, 4)
            snapshot["underexposed_fraction"] = round(quality.underexposed_fraction, 4)
        snapshot["view_valid"] = True
        snapshot["view_valid_reasons"] = []
        snapshot["quality_ok"] = True
        snapshot["left_grip_active"] = True
        snapshot["right_grip_active"] = True
        return {
            "scenario": scenario,
            "snapshot": snapshot,
            "transitions": runtime.fsm.transition_log(),
            "reference_image_path": str(ref_path),
            "evidence_kind": "initial_label",
            "pipe_color": pipe_color,
        }

    runtime.start_cycle(cycle_id=cycle_id, pipe_color=pipe_color, now_monotonic=now)

    if scenario == "alarm":
        # Prime ViewValidEvaluator stability window (5 frames across >=400 ms), then observe initial label absent -> ALARM
        for _ in range(5):
            obs = next_obs(dt=0.1)
            snapshot = runtime.step(
                base_frame,
                obs,
                detections=grip_detections,
                label_status="absent",
                label_confidence=0.96,
                pipe_color=pipe_color,
                now_monotonic=now,
            )
        return {
            "scenario": scenario,
            "snapshot": snapshot,
            "transitions": runtime.fsm.transition_log(),
            "reference_image_path": str(ref_path),
            "evidence_kind": "anomaly",
            "pipe_color": pipe_color,
        }

    # For a_pass, b_pass, c_pass, glare_invalid: first confirm initial complete label -> READY -> start_inspection -> INSPECTING
    for _ in range(10):
        obs = next_obs(dt=0.1)
        snapshot = runtime.step(
            base_frame,
            obs,
            detections=grip_detections,
            label_status="complete",
            label_confidence=0.98,
            pipe_color=pipe_color,
            now_monotonic=now,
        )
        if runtime.fsm.state == FSMState.READY:
            break

    runtime.start_inspection(now_monotonic=now)

    if scenario == "a_pass":
        for _ in range(6):
            obs = next_obs(dt=0.1)
            snapshot = runtime.step(
                base_frame,
                obs,
                detections=[
                    *grip_detections,
                    {"class_name": "marker", "score": 0.91, "box": (0.45, 0.46, 0.55, 0.54)},
                ],
                label_status="crossed_out",
                label_confidence=0.97,
                pipe_color=pipe_color,
                now_monotonic=now,
            )
            if runtime.fsm.state == FSMState.PASS:
                break
        evidence_kind = "A"

    elif scenario == "b_pass":
        for _ in range(6):
            obs = next_obs(dt=0.1)
            snapshot = runtime.step(
                base_frame,
                obs,
                detections=grip_detections,
                label_status="covered",
                label_confidence=0.96,
                pipe_color=pipe_color,
                now_monotonic=now,
            )
            if runtime.fsm.state == FSMState.PASS:
                break
        evidence_kind = "B"

    elif scenario == "c_pass":
        # 1. Hand enters central label contact ROI for >=2 frames -> ContactCandidateTracker emits rising edge -> FSM PROCESSING
        for _ in range(2):
            obs = next_obs(dt=0.1, label_occluded=True)
            runtime.step(
                base_frame,
                obs,
                detections=central_contact_detections,
                label_status="complete",
                label_confidence=0.72,
                pipe_color=pipe_color,
                now_monotonic=now,
            )
        # 2. Hand leaves central contact ROI (exit debounce + hand_leave_stable_ms + ViewValidEvaluator re-stabilizes)
        #    and label is absent for >= 1200 ms -> VERIFY_C_ABSENT -> PASS (c_timer_ms == 1200)
        for _ in range(20):
            obs = next_obs(dt=0.1, label_occluded=False)
            snapshot = runtime.step(
                base_frame,
                obs,
                detections=grip_detections,
                label_status="absent",
                label_confidence=0.98,
                pipe_color=pipe_color,
                now_monotonic=now,
            )
            if runtime.fsm.state == FSMState.PASS:
                break
        evidence_kind = "C"

    elif scenario == "glare_invalid":
        glare_frame = _synthesize_glare_frame(base_frame, view_cfg.rois.label_roi)
        for _ in range(2):
            obs = next_obs(dt=0.1)
            snapshot = runtime.step(
                glare_frame,
                obs,
                detections=grip_detections,
                label_status="complete",
                label_confidence=0.61,
                pipe_color=pipe_color,
                now_monotonic=now,
            )
        evidence_kind = "anomaly"

    return {
        "scenario": scenario,
        "snapshot": snapshot,
        "transitions": runtime.fsm.transition_log(),
        "reference_image_path": str(ref_path),
        "evidence_kind": evidence_kind,
        "pipe_color": pipe_color,
    }
