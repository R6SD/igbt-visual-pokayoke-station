"""Fail-closed image/view quality gate; this module never produces PASS."""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

Box = Sequence[float]  # normalized xyxy


@dataclass(frozen=True)
class ROISet:
    tube_roi: Box
    label_roi: Box
    label_contact_roi: Box
    left_grip_zone: Box | None = None
    right_grip_zone: Box | None = None


@dataclass(frozen=True)
class ViewConfig:
    camera_device_id: str
    calibrated_width: int
    calibrated_height: int
    roi_config_version: str
    rois: ROISet
    quality_thresholds_approved: bool = False
    orientation_detector_ready: bool = False
    min_brightness: float | None = None
    max_brightness: float | None = None
    max_overexposed_fraction: float | None = None
    max_underexposed_fraction: float | None = None
    min_laplacian_variance: float | None = None
    max_frame_gap_ms: float = 200.0
    stable_window_ms: float = 400.0
    min_stable_frames: int = 5
    max_center_drift_norm: float = 0.008
    max_label_roi_offset_norm: float = 0.04

    def __post_init__(self):
        if self.calibrated_width <= 0 or self.calibrated_height <= 0:
            raise ValueError("Calibrated camera dimensions must be positive.")
        if self.min_stable_frames < 2 or self.stable_window_ms <= 0 or self.max_frame_gap_ms <= 0:
            raise ValueError("Stability frame count and timing limits must be positive.")
        if not self.camera_device_id or not self.roi_config_version:
            raise ValueError("Camera device ID and ROI config version are required.")
        thresholds = (self.min_brightness, self.max_brightness, self.max_overexposed_fraction,
                      self.max_underexposed_fraction, self.min_laplacian_variance)
        if self.quality_thresholds_approved and any(value is None for value in thresholds):
            raise ValueError("Approved quality gates require every site-calibrated threshold value.")


@dataclass(frozen=True)
class FrameObservation:
    frame_id: int
    captured_monotonic_s: float
    camera_device_id: str
    roi_config_version: str | None
    camera_healthy: bool | None
    tube_present: bool | None
    label_window_present: bool | None
    label_occluded: bool | None
    front_face_confirmed: bool | None
    orientation_reference_visible: bool | None
    orientation_evidence_source: str | None
    tube_center_norm: tuple[float, float] | None
    label_center_norm: tuple[float, float] | None
    # Include non-glove occluders (pen, foreign objects); glove detections are also checked.
    occluder_boxes_norm: tuple[Box, ...] = ()


@dataclass(frozen=True)
class QualityMetrics:
    brightness_mean: float
    overexposed_fraction: float
    underexposed_fraction: float
    laplacian_variance: float


@dataclass(frozen=True)
class ViewResult:
    view_valid: bool
    reasons: tuple[str, ...]
    quality: QualityMetrics | None
    stable_frames: int
    roi_config_version: str
    event: str = "VIEW_VALIDITY_OBSERVATION"


def valid_box(box: Box | None) -> bool:
    return bool(box is not None and len(box) == 4 and all(0.0 <= float(x) <= 1.0 for x in box)
                and box[2] > box[0] and box[3] > box[1])


def box_intersection_fraction(box: Box, region: Box) -> float:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return 0.0
    intersection = max(0.0, min(box[2], region[2]) - max(box[0], region[0])) * \
                   max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    return intersection / area


def measure_label_quality(frame: np.ndarray, label_roi: Box) -> QualityMetrics | None:
    if frame is None or frame.size == 0 or not valid_box(label_roi):
        return None
    height, width = frame.shape[:2]
    x1, y1 = int(label_roi[0] * width), int(label_roi[1] * height)
    x2, y2 = int(np.ceil(label_roi[2] * width)), int(np.ceil(label_roi[3] * height))
    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return QualityMetrics(
        brightness_mean=float(np.mean(gray)),
        overexposed_fraction=float(np.mean(gray >= 250)),
        underexposed_fraction=float(np.mean(gray <= 5)),
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


class ViewValidEvaluator:
    """Evaluate positive evidence only; missing, stale, shifted or ambiguous data is invalid."""

    def __init__(self, config: ViewConfig):
        self.config = config
        self._history: deque[tuple[float, int, tuple[float, float], tuple[float, float]]] = deque()
        self._last_frame_id: int | None = None
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._history.clear()
        self._last_frame_id = None
        self._last_timestamp = None

    def _base_checks(self, frame: np.ndarray, obs: FrameObservation) -> tuple[list[str], QualityMetrics | None]:
        cfg = self.config
        reasons: list[str] = []
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
            return ["FRAME_MISSING_OR_INVALID"], None
        h, w = frame.shape[:2]
        if (w, h) != (cfg.calibrated_width, cfg.calibrated_height):
            reasons.append("CALIBRATION_RESOLUTION_MISMATCH")
        if obs.camera_device_id != cfg.camera_device_id:
            reasons.append("CAMERA_DEVICE_MISMATCH")
        if obs.roi_config_version != cfg.roi_config_version:
            reasons.append("ROI_CONFIG_VERSION_MISMATCH")
        if obs.camera_healthy is not True:
            reasons.append("CAMERA_NOT_CONFIRMED_HEALTHY")
        if obs.tube_present is not True:
            reasons.append("TUBE_NOT_CONFIRMED_IN_VIEW")
        if obs.label_window_present is not True:
            reasons.append("LABEL_WINDOW_OUT_OF_VIEW_OR_SHIFTED")
        if obs.label_occluded is not False:
            reasons.append("LABEL_OCCLUDED_OR_OCCLUSION_UNKNOWN")
        required_rois = {
            "tube_roi": cfg.rois.tube_roi,
            "label_roi": cfg.rois.label_roi,
            "label_contact_roi": cfg.rois.label_contact_roi,
        }
        if cfg.rois.left_grip_zone is not None:
            required_rois["left_grip_zone"] = cfg.rois.left_grip_zone
        if cfg.rois.right_grip_zone is not None:
            required_rois["right_grip_zone"] = cfg.rois.right_grip_zone
        if any(not valid_box(box) for box in required_rois.values()):
            reasons.append("ROI_CALIBRATION_INVALID")
        elif cfg.rois.left_grip_zone is None or cfg.rois.right_grip_zone is None:
            # No assumption that hands cannot enter the central region if grip zones are absent.
            reasons.append("BOTH_GRIP_ZONES_NOT_CALIBRATED")
        else:
            if (box_intersection_fraction(cfg.rois.label_contact_roi, cfg.rois.left_grip_zone) > 0.01 or
                    box_intersection_fraction(cfg.rois.label_contact_roi, cfg.rois.right_grip_zone) > 0.01):
                reasons.append("CONTACT_AND_GRIP_ZONES_OVERLAP")

        if obs.front_face_confirmed is not True:
            reasons.append("FRONT_FACE_NOT_CONFIRMED")
        if obs.orientation_reference_visible is not True or obs.orientation_evidence_source != "machine_reference":
            reasons.append("MACHINE_ORIENTATION_REFERENCE_NOT_CONFIRMED")
        if not cfg.orientation_detector_ready:
            reasons.append("ORIENTATION_DETECTOR_NOT_VALIDATED")
        thresholds_ready = cfg.quality_thresholds_approved and all(value is not None for value in (
            cfg.min_brightness, cfg.max_brightness, cfg.max_overexposed_fraction,
            cfg.max_underexposed_fraction, cfg.min_laplacian_variance,
        ))
        if not thresholds_ready:
            reasons.append("QUALITY_THRESHOLDS_NOT_CALIBRATED")

        if valid_box(cfg.rois.label_roi):
            for box in obs.occluder_boxes_norm:
                if valid_box(box) and box_intersection_fraction(box, cfg.rois.label_roi) >= 0.03:
                    reasons.append("DETECTED_OCCLUDER_OVERLAPS_LABEL_ROI")
                    break
        quality = measure_label_quality(frame, cfg.rois.label_roi)
        if quality is None:
            reasons.append("LABEL_ROI_QUALITY_UNAVAILABLE")
        elif thresholds_ready:
            if not cfg.min_brightness <= quality.brightness_mean <= cfg.max_brightness:
                reasons.append("LABEL_ROI_BRIGHTNESS_OUT_OF_RANGE")
            if quality.overexposed_fraction > cfg.max_overexposed_fraction:
                reasons.append("LABEL_ROI_OVEREXPOSED")
            if quality.underexposed_fraction > cfg.max_underexposed_fraction:
                reasons.append("LABEL_ROI_UNDEREXPOSED")
            if quality.laplacian_variance < cfg.min_laplacian_variance:
                reasons.append("LABEL_ROI_BLURRED")
        if obs.tube_center_norm is None or obs.label_center_norm is None:
            reasons.append("ROI_POSITION_OBSERVATION_MISSING")
        else:
            tx, ty = obs.tube_center_norm
            tube_roi = cfg.rois.tube_roi
            if valid_box(tube_roi) and not (tube_roi[0] <= tx <= tube_roi[2] and tube_roi[1] <= ty <= tube_roi[3]):
                reasons.append("TUBE_CENTER_OUTSIDE_TUBE_ROI")
            lx, ly = obs.label_center_norm
            rx = (cfg.rois.label_roi[0] + cfg.rois.label_roi[2]) / 2 if valid_box(cfg.rois.label_roi) else lx
            ry = (cfg.rois.label_roi[1] + cfg.rois.label_roi[3]) / 2 if valid_box(cfg.rois.label_roi) else ly
            label_roi = cfg.rois.label_roi
            if valid_box(label_roi) and not (label_roi[0] <= lx <= label_roi[2] and label_roi[1] <= ly <= label_roi[3]):
                reasons.append("LABEL_CENTER_OUTSIDE_LABEL_ROI")
            elif float(np.hypot(lx - rx, ly - ry)) > cfg.max_label_roi_offset_norm:
                reasons.append("LABEL_ROI_POSITION_SHIFTED")
        return reasons, quality

    def evaluate(self, frame: np.ndarray, observation: FrameObservation) -> ViewResult:
        reasons, quality = self._base_checks(frame, observation)
        now = observation.captured_monotonic_s
        if self._last_timestamp is not None:
            gap_ms = (now - self._last_timestamp) * 1000.0
            frame_gap = observation.frame_id - self._last_frame_id if self._last_frame_id is not None else 1
            if gap_ms <= 0 or gap_ms > self.config.max_frame_gap_ms or frame_gap != 1:
                reasons.append("FRAME_GAP_OR_TIMESTAMP_DISCONTINUITY")
                self._history.clear()
        self._last_timestamp = now
        self._last_frame_id = observation.frame_id

        centers_present = observation.tube_center_norm is not None and observation.label_center_norm is not None
        if reasons or not centers_present:
            self._history.clear()
            return ViewResult(False, tuple(dict.fromkeys(reasons or ["VIEW_EVIDENCE_INCOMPLETE"])), quality,
                              0, self.config.roi_config_version)

        self._history.append((now, observation.frame_id, observation.tube_center_norm, observation.label_center_norm))
        stability_cutoff = now - self.config.stable_window_ms / 1000.0
        # Keep enough frames to span the full time window; capping at only
        # min_stable_frames would make a 400 ms window impossible at 30 FPS.
        while len(self._history) > self.config.min_stable_frames and self._history[1][0] <= stability_cutoff:
            self._history.popleft()
        stable_count = len(self._history)
        stable_duration_ms = (self._history[-1][0] - self._history[0][0]) * 1000.0 if stable_count > 1 else 0.0
        if stable_count < self.config.min_stable_frames or stable_duration_ms < self.config.stable_window_ms:
            return ViewResult(False, ("STABILITY_WINDOW_NOT_MET",), quality, stable_count,
                              self.config.roi_config_version)
        tube_centers = [row[2] for row in self._history]
        label_centers = [row[3] for row in self._history]
        drift = max(
            max(float(np.hypot(p[0] - q[0], p[1] - q[1])) for p in points for q in points)
            for points in (tube_centers, label_centers)
        )
        if drift > self.config.max_center_drift_norm:
            self._history.clear()
            return ViewResult(False, ("TUBE_OR_LABEL_MOTION_UNSTABLE",), quality, stable_count,
                              self.config.roi_config_version)
        return ViewResult(True, (), quality, stable_count, self.config.roi_config_version)
