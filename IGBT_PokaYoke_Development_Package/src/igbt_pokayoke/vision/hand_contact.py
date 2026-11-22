"""White-glove detection post-processing and conservative contact candidates.

An overlap candidate is 2D proximity evidence only. It is not proof of touch.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


Box = Sequence[float]  # normalized xyxy in [0, 1]


@dataclass(frozen=True)
class ContactConfig:
    hand_score_threshold: float = 0.70
    contact_overlap_ratio: float = 0.20
    grip_overlap_ratio_exclusion: float = 0.10


@dataclass(frozen=True)
class ContactCandidate:
    hand_contact_candidate: bool
    label_contact_event: bool
    event: str
    reason: str
    evidence_type: str = "2d_visual_proximity_proxy"
    proves_physical_contact: bool = False


def validate_class_map(class_names: Iterable[str]) -> None:
    names = set(class_names)
    if "hand_or_pen" in names:
        raise ValueError("Merged class 'hand_or_pen' is prohibited; use separate 'hand_glove' and 'marker'.")
    if "hand_glove" not in names:
        raise ValueError("Detector class map must include 'hand_glove'.")
    if "marker" in names:
        return


def _valid_box(box: Box) -> bool:
    return len(box) == 4 and all(0.0 <= float(v) <= 1.0 for v in box) and box[2] > box[0] and box[3] > box[1]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap_of_box_fraction(box: Box, region: Box) -> float:
    if not _valid_box(box) or not _valid_box(region):
        return 0.0
    width = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    height = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = _area(box)
    return width * height / area if area else 0.0


def detect_contact_candidate(
    detections: Iterable[dict], label_contact_roi: Box,
    left_grip_zone: Box | None, right_grip_zone: Box | None,
    config: ContactConfig | None = None,
) -> ContactCandidate:
    """Return candidate only for a confident hand_glove in central contact ROI.

    Any detection materially overlapping a normal grip zone is excluded. Marker,
    unknown labels and the deprecated merged label never trigger a candidate.
    """
    cfg = config or ContactConfig()
    if not _valid_box(label_contact_roi):
        return ContactCandidate(False, False, "none", "contact_roi_invalid")
    if not _valid_box(left_grip_zone) or not _valid_box(right_grip_zone):
        return ContactCandidate(False, False, "none", "both_grip_zones_must_be_calibrated")
    grip_zones = [left_grip_zone, right_grip_zone]
    saw_hand = False
    for detection in detections:
        name = str(detection.get("class_name", ""))
        if name == "hand_or_pen":
            continue
        if name != "hand_glove" or float(detection.get("score", 0.0)) < cfg.hand_score_threshold:
            continue
        box = detection.get("box")
        if not box or not _valid_box(box):
            continue
        saw_hand = True
        in_grip = any(overlap_of_box_fraction(box, zone) >= cfg.grip_overlap_ratio_exclusion for zone in grip_zones)
        if in_grip:
            continue
        if overlap_of_box_fraction(box, label_contact_roi) >= cfg.contact_overlap_ratio:
            return ContactCandidate(
                True, False, "hand_contact_candidate", "hand_glove_overlaps_label_contact_roi",
            )
    if saw_hand:
        return ContactCandidate(False, False, "none", "hand_outside_contact_roi_or_in_grip_zone")
    return ContactCandidate(False, False, "none", "no_confident_hand_glove_detection")


class ContactCandidateTracker:
    """Emit a rising-edge candidate; never calls it physical contact.

    ``view_available`` means the frame and detector output are usable for this
    observation. Do not pass overall ``view_valid`` here: a hand can occlude
    the label and make the view invalid while still being a contact candidate.
    """

    def __init__(self, enter_frames: int = 2, exit_frames: int = 2) -> None:
        if enter_frames < 1 or exit_frames < 1:
            raise ValueError("Contact tracker enter/exit frame counts must be positive.")
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self._inside = False
        self._enter_count = 0
        self._exit_count = 0

    def update(self, candidate: ContactCandidate, view_available: bool = True) -> ContactCandidate:
        now_inside = bool(view_available and candidate.hand_contact_candidate)
        if not view_available:
            self.reset()
            return ContactCandidate(False, False, "none", "frame_or_detector_unavailable")
        if now_inside:
            self._enter_count += 1
            self._exit_count = 0
            if not self._inside and self._enter_count >= self.enter_frames:
                self._inside = True
                return ContactCandidate(True, True, "hand_contact_candidate", candidate.reason)
            return ContactCandidate(self._inside, False, "none", "candidate_continuing" if self._inside else "entry_debounce")
        self._enter_count = 0
        if self._inside:
            self._exit_count += 1
            if self._exit_count < self.exit_frames:
                return ContactCandidate(True, False, "none", "exit_debounce")
        self._inside = False
        self._exit_count = 0
        return ContactCandidate(False, False, "none", candidate.reason)

    def reset(self) -> None:
        self._inside = False
        self._enter_count = 0
        self._exit_count = 0
