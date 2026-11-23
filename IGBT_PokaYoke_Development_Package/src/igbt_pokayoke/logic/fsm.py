"""Fail-safe A/B/C observation FSM. No camera, model, or production I/O dependency."""
from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FSMState(str, Enum):
    BOOT = "BOOT"
    WAIT_PART = "WAIT_PART"
    VERIFY_INITIAL = "VERIFY_INITIAL"
    READY = "READY"
    INSPECTING = "INSPECTING"
    PROCESSING = "PROCESSING"
    VERIFY_C_ABSENT = "VERIFY_C_ABSENT"
    VIEW_INVALID = "VIEW_INVALID"
    PASS = "PASS"
    ALARM = "ALARM"
    FAULT = "FAULT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class FSMConfig:
    initial_complete_ms: int = 800
    stable_state_ms: int = 500
    min_stable_frames: int = 5
    c_absent_duration_ms: int = 1200
    max_frame_gap_ms: int = 200
    max_capture_age_ms: int = 500
    hand_leave_stable_ms: int = 250
    cycle_timeout_s: float = 60.0
    action_timeout_s: float = 40.0
    max_log_records: int = 1000

    def __post_init__(self):
        if self.c_absent_duration_ms != 1200:
            raise ValueError("C removal confirmation is fixed at 1200 ms by the process requirement.")
        if self.initial_complete_ms < 0 or self.stable_state_ms < 0 or self.min_stable_frames < 1:
            raise ValueError("Initial/stable frame requirements are invalid.")
        if self.max_frame_gap_ms <= 0 or self.max_capture_age_ms <= 0:
            raise ValueError("Capture timing limits must be positive.")
        if min(self.cycle_timeout_s, self.action_timeout_s) <= 0:
            raise ValueError("Cycle and action timeouts must be positive.")
        if self.max_log_records < 1:
            raise ValueError("Log retention limit must be positive.")


@dataclass(frozen=True)
class FSMEvent:
    kind: str
    cycle_id: str | None = None
    capture_monotonic_s: float | None = None
    frame_id: int | None = None
    label_class: str = "unknown"
    view_valid: bool | None = None
    quality_ok: bool | None = None
    camera_healthy: bool | None = None
    camera_settings_locked: bool | None = None
    config_match: bool | None = None
    tube_present: bool | None = None
    label_window_present: bool | None = None
    label_roi_clear: bool | None = None
    hand_occluded: bool | None = None
    hand_contact_candidate: bool = False
    hand_detection_valid: bool | None = None
    hand_in_contact_roi: bool | None = None
    hand_in_label_contact_roi: bool | None = None
    hand_in_left_grip_zone: bool = False
    hand_in_right_grip_zone: bool = False
    marker_detected: bool = False
    pipe_color: str | None = None
    reason: str | None = None
    reset_authorized: bool = False
    operator_anon_id: str | None = None


class InspectionFSM:
    """State machine that accepts already-evaluated visual observations.

    Capture deltas drive stable-label and C absent timers. An injected monotonic
    clock drives cycle/action timeout and stale-frame checks. The class has no
    hardware output API by design.
    """

    TERMINAL = {FSMState.PASS, FSMState.ALARM, FSMState.FAULT, FSMState.REVIEW_REQUIRED}

    def __init__(self, config: FSMConfig | None = None, clock: Callable[[], float] = time.monotonic,
                 restored_cycle: dict | None = None,
                 audit_sink: Callable[[dict], None] | None = None):
        self.config = config or FSMConfig()
        self._clock = clock
        self._audit_sink = audit_sink
        self.state = FSMState.WAIT_PART
        self.outcome = "WAITING"
        self.result_reason: str | None = None
        self.cycle_id: str | None = None
        self.initial_complete_confirmed = False
        self.transition_records: deque[dict] = deque(maxlen=self.config.max_log_records)
        self._pending_audit_records: deque[dict] = deque()
        self._log_sequence = 0
        self._cycle_started: float | None = None
        self._inspection_started: float | None = None
        self._last_capture: float | None = None
        self._last_frame_id: int | None = None
        self._stable_class: str | None = None
        self._stable_start: float | None = None
        self._stable_frames = 0
        self._contact_seen = False
        self._hand_left_start: float | None = None
        self._hand_left_confirmed = False
        self._c_absent_start: float | None = None
        self._c_absent_last: float | None = None
        self._resume_state = FSMState.VERIFY_INITIAL
        if restored_cycle and restored_cycle.get("state") not in {"WAIT_PART", "PASS", "ALARM", "FAULT"}:
            self.cycle_id = restored_cycle.get("cycle_id")
            self.result_reason = "RESTART_DURING_ACTIVE_CYCLE"
            self.state = FSMState.REVIEW_REQUIRED
            self.outcome = "ALARM"
            self._append_log({"kind": "startup_restore"}, self._clock(), None,
                             FSMState.WAIT_PART, FSMState.REVIEW_REQUIRED, self.result_reason)

    def _c_absent_elapsed_ms(self) -> float:
        if self._c_absent_start is None or self._c_absent_last is None:
            return 0.0
        return max(0.0, (self._c_absent_last - self._c_absent_start) * 1000.0)

    @property
    def c_timer_ms(self) -> int:
        elapsed = self._c_absent_elapsed_ms()
        if elapsed <= 0.0:
            return 0
        limit = self.config.c_absent_duration_ms
        if self.state == FSMState.PASS and self.result_reason == "C_REMOVED":
            return limit
        if elapsed + 1e-6 >= limit:
            return limit
        return max(0, min(limit - 1, int(round(elapsed))))

    @property
    def pending_audit_count(self) -> int:
        return len(self._pending_audit_records)

    def flush_audit_sink(self) -> int:
        """Deliver queued audit records in FIFO order, retaining failed records for retry (R2-12)."""
        if self._audit_sink is None:
            self._pending_audit_records.clear()
            return 0
        delivered = 0
        while self._pending_audit_records:
            record = self._pending_audit_records[0]
            try:
                self._audit_sink(dict(record))
            except Exception as exc:
                logger.warning("FSM audit sink delivery failed (will retry seq %s): %s", record.get("sequence"), exc)
                break
            self._pending_audit_records.popleft()
            delivered += 1
        return delivered

    def invalidate_geometry_evidence(
        self,
        reason: str = "CAMERA_GEOMETRY_CHANGED",
        now_monotonic: float | None = None,
    ) -> None:
        """Invalidate geometry-dependent contact/stability/C evidence on active cycles (R2-08)."""
        if self.state in self.TERMINAL or self.state == FSMState.WAIT_PART:
            return
        now = self._clock() if now_monotonic is None else float(now_monotonic)
        self._clear_stability()
        self._clear_c_timer()
        self._contact_seen = False
        self._hand_left_start = None
        self._hand_left_confirmed = False
        self._resume_state = self._phase_for_resume()
        self._transition(FSMState.VIEW_INVALID, reason, "geometry_reset", now, None, None)

    def transition_log(self) -> list[dict]:
        return [dict(row) for row in self.transition_records]

    def process(self, event: FSMEvent, now_monotonic: float | None = None) -> FSMState:
        if not isinstance(event, FSMEvent):
            raise TypeError("InspectionFSM.process expects an FSMEvent.")
        now = self._clock() if now_monotonic is None else float(now_monotonic)
        before = self.state
        reason = "NO_STATE_CHANGE"
        if event.kind == "new_cycle":
            reason = self._new_cycle(event, now)
        elif event.kind == "start_inspection":
            reason = self._start_inspection(event, now)
        elif event.kind == "observation":
            reason = self._observation(event, now)
        elif event.kind == "fault":
            reason = event.reason or "FAULT_REPORTED"
            self._transition(FSMState.FAULT, reason, event.kind, now, event.capture_monotonic_s, event.frame_id)
        elif event.kind in {"reset", "part_removed"}:
            reason = self._reset(event, now)
        else:
            reason = "UNKNOWN_EVENT_TYPE"
        after = self.state
        # Retain bounded in-memory audit records (and stream full audit to audit_sink if provided).
        self._append_log(
            asdict(event),
            now,
            event.capture_monotonic_s,
            before,
            after,
            reason,
            record_type="EVENT",
            frame_id=event.frame_id,
        )
        return self.state

    def _new_cycle(self, event: FSMEvent, now: float) -> str:
        if self.state != FSMState.WAIT_PART:
            return "CYCLE_ALREADY_ACTIVE"
        self._clear_cycle_data()
        self.cycle_id = event.cycle_id or f"cycle-{int(now * 1000)}"
        self._cycle_started = now
        self.result_reason = None
        self._transition(FSMState.VERIFY_INITIAL, "NEW_CYCLE", event.kind, now, event.capture_monotonic_s, event.frame_id)
        return "NEW_CYCLE"

    def _start_inspection(self, event: FSMEvent, now: float) -> str:
        if self.state != FSMState.READY or not self.initial_complete_confirmed:
            return "INITIAL_COMPLETE_NOT_CONFIRMED"
        self._inspection_started = now
        self._transition(
            FSMState.INSPECTING,
            "INSPECTION_STARTED",
            event.kind,
            now,
            event.capture_monotonic_s,
            event.frame_id,
        )
        return "INSPECTION_STARTED"

    def _reset(self, event: FSMEvent, now: float) -> str:
        if event.kind == "part_removed" and self.state == FSMState.PASS:
            self._clear_cycle_data()
            self.cycle_id = None
            self._transition(FSMState.WAIT_PART, "PART_REMOVED", event.kind, now, event.capture_monotonic_s, event.frame_id)
            return "PART_REMOVED"
        if not event.reset_authorized:
            return "RESET_NOT_AUTHORIZED"
        if self.state not in {FSMState.ALARM, FSMState.FAULT, FSMState.REVIEW_REQUIRED, FSMState.PASS}:
            return "RESET_NOT_ALLOWED_IN_ACTIVE_CYCLE"
        self._clear_cycle_data()
        self.cycle_id = None
        self._transition(
            FSMState.WAIT_PART,
            event.reason or "AUTHORIZED_RESET",
            event.kind,
            now,
            event.capture_monotonic_s,
            event.frame_id,
        )
        return event.reason or "AUTHORIZED_RESET"

    def _phase_for_resume(self) -> FSMState:
        if not self.initial_complete_confirmed:
            return FSMState.VERIFY_INITIAL
        if self._inspection_started is None:
            return FSMState.READY
        if self._contact_seen:
            return FSMState.PROCESSING
        return FSMState.INSPECTING

    def _observation(self, event: FSMEvent, now: float) -> str:
        if self.state in self.TERMINAL or self.state == FSMState.WAIT_PART:
            return "OBSERVATION_IGNORED_OUTSIDE_ACTIVE_CYCLE"
        if self._cycle_started is not None and now - self._cycle_started >= self.config.cycle_timeout_s:
            self._transition(FSMState.ALARM, "CYCLE_TIMEOUT", event.kind, now, event.capture_monotonic_s, event.frame_id)
            return "CYCLE_TIMEOUT"
        if self._inspection_started is not None and now - self._inspection_started >= self.config.action_timeout_s:
            self._transition(FSMState.ALARM, "ACTION_TIMEOUT", event.kind, now, event.capture_monotonic_s, event.frame_id)
            return "ACTION_TIMEOUT"

        if event.camera_healthy is False:
            self._transition(FSMState.FAULT, "FAULT_CAMERA", event.kind, now, event.capture_monotonic_s, event.frame_id)
            return "FAULT_CAMERA"
        if event.camera_settings_locked is False:
            self._transition(
                FSMState.FAULT,
                "CAMERA_SETTINGS_NOT_LOCKED",
                event.kind,
                now,
                event.capture_monotonic_s,
                event.frame_id,
            )
            return "CAMERA_SETTINGS_NOT_LOCKED"

        frame_reason = self._capture_timing_reason(event, now)
        if frame_reason:
            self._clear_stability()
            self._clear_c_timer()
            self._hand_left_start = None
            self._hand_left_confirmed = False
            self._resume_state = self._phase_for_resume()
            self._transition(FSMState.VIEW_INVALID, frame_reason, event.kind, now, event.capture_monotonic_s, event.frame_id)
            return frame_reason

        if event.config_match is False:
            return self._invalidate(event, now, "CONFIGURATION_MISMATCH", preserve_contact=False)

        # A central hand candidate may be visible while it occludes the label;
        # it remains a visual candidate, never proof of physical touch.
        # Contact is accepted into the action sequence only after inspection has started.
        inspection_active = self.initial_complete_confirmed and self._inspection_started is not None
        contact = self._contact_eligible(event)
        if contact and inspection_active:
            self._contact_seen = True
            self._hand_left_start = None
            self._hand_left_confirmed = False
            if self.state != FSMState.PROCESSING:
                self._transition(
                    FSMState.PROCESSING,
                    "HAND_CONTACT_CANDIDATE",
                    event.kind,
                    now,
                    event.capture_monotonic_s,
                    event.frame_id,
                )
        elif event.hand_in_contact_roi is True and inspection_active:
            self._hand_left_start = None
            self._hand_left_confirmed = False

        valid, invalid_reason = self._observation_valid(event)
        if not valid:
            preserve = bool(self._contact_seen and event.hand_occluded is True)
            return self._invalidate(event, now, invalid_reason, preserve_contact=preserve)

        if self.state == FSMState.VIEW_INVALID:
            target = self._phase_for_resume()
            self._transition(target, "VIEW_RESTORED", event.kind, now, event.capture_monotonic_s, event.frame_id)

        if event.label_class == "unknown":
            self._clear_stability()
            self._clear_c_timer()
            self._transition(FSMState.ALARM, "ALARM_UNKNOWN_LABEL", event.kind, now, event.capture_monotonic_s, event.frame_id)
            return "ALARM_UNKNOWN_LABEL"

        if self.state == FSMState.VERIFY_INITIAL:
            if event.label_class != "complete":
                code = "INITIAL_LABEL_ABSENT" if event.label_class == "absent" else "INITIAL_LABEL_NOT_COMPLETE"
                self._transition(FSMState.ALARM, code, event.kind, now, event.capture_monotonic_s, event.frame_id)
                return code
            stable = self._observe_stable("complete", event.capture_monotonic_s)
            elapsed_ms = self._stable_elapsed_ms(event.capture_monotonic_s)
            if stable and elapsed_ms + 1e-6 >= self.config.initial_complete_ms:
                self.initial_complete_confirmed = True
                self._transition(
                    FSMState.READY,
                    "INITIAL_COMPLETE_CONFIRMED",
                    event.kind,
                    now,
                    event.capture_monotonic_s,
                    event.frame_id,
                )
                return "INITIAL_COMPLETE_CONFIRMED"
            return "VERIFYING_INITIAL_COMPLETE"

        if self.state not in {FSMState.READY, FSMState.INSPECTING, FSMState.PROCESSING, FSMState.VERIFY_C_ABSENT}:
            return "OBSERVATION_NOT_APPLICABLE"

        if self.state == FSMState.READY:
            # Do not evaluate action outcomes until inspection starts.
            return "WAITING_FOR_INSPECTION_START"

        # A/B are independent valid label-state evidence; the model itself never decides PASS.
        if event.label_class in {"crossed_out", "covered"}:
            self._clear_c_timer()
            if (
                self._observe_stable(event.label_class, event.capture_monotonic_s)
                and self._stable_elapsed_ms(event.capture_monotonic_s) + 1e-6 >= self.config.stable_state_ms
            ):
                code = "A_CROSSED_OUT" if event.label_class == "crossed_out" else "B_COVERED"
                self._transition(FSMState.PASS, code, event.kind, now, event.capture_monotonic_s, event.frame_id)
                return code
            return "VERIFYING_A_OR_B"

        if event.label_class in {"complete", "absent"}:
            self._clear_stability()

        if event.label_class == "complete":
            # The label reappeared after a C candidate: force a new contact sequence.
            self._clear_c_timer()
            if self.state in {FSMState.VERIFY_C_ABSENT, FSMState.PROCESSING} and event.hand_in_contact_roi is not True:
                self._contact_seen = False
                self._hand_left_confirmed = False
                self._hand_left_start = None
                self._transition(
                    FSMState.INSPECTING,
                    "LABEL_REAPPEARED_C_CANCELED",
                    event.kind,
                    now,
                    event.capture_monotonic_s,
                    event.frame_id,
                )
            return "LABEL_COMPLETE_NO_C_EVIDENCE"

        if self.state in {FSMState.PROCESSING, FSMState.VERIFY_C_ABSENT}:
            if event.hand_in_contact_roi is True or event.hand_contact_candidate:
                self._hand_left_start = None
                self._hand_left_confirmed = False
                self._clear_c_timer()
                if self.state == FSMState.VERIFY_C_ABSENT:
                    self._transition(
                        FSMState.PROCESSING,
                        "HAND_REENTERED_CONTACT_ROI",
                        event.kind,
                        now,
                        event.capture_monotonic_s,
                        event.frame_id,
                    )
                return "WAITING_FOR_HAND_TO_LEAVE"
            if not self._contact_seen:
                self._clear_c_timer()
                return "C_REQUIRES_CONTACT_CANDIDATE"
            if self._hand_left_start is None:
                self._hand_left_start = event.capture_monotonic_s
            left_ms = (event.capture_monotonic_s - self._hand_left_start) * 1000.0
            if left_ms + 1e-6 >= self.config.hand_leave_stable_ms:
                self._hand_left_confirmed = True
            if not self._hand_left_confirmed or event.label_roi_clear is not True:
                self._clear_c_timer()
                return "WAITING_FOR_CLEAR_ROI_AFTER_HAND_LEAVE"
            if event.label_class == "absent":
                if self.state != FSMState.VERIFY_C_ABSENT:
                    self._clear_c_timer()
                    self._c_absent_start = event.capture_monotonic_s
                    self._c_absent_last = event.capture_monotonic_s
                    self._transition(
                        FSMState.VERIFY_C_ABSENT,
                        "C_ABSENT_TIMER_STARTED",
                        event.kind,
                        now,
                        event.capture_monotonic_s,
                        event.frame_id,
                    )
                else:
                    if self._c_absent_last is None or (event.capture_monotonic_s - self._c_absent_last) * 1000.0 > self.config.max_frame_gap_ms:
                        self._clear_c_timer()
                        self._c_absent_start = event.capture_monotonic_s
                        self._c_absent_last = event.capture_monotonic_s
                        return "C_ABSENT_TIMER_RESTARTED_AFTER_GAP"
                    self._c_absent_last = event.capture_monotonic_s
                    if self._c_absent_elapsed_ms() + 1e-6 >= self.config.c_absent_duration_ms:
                        self._transition(FSMState.PASS, "C_REMOVED", event.kind, now, event.capture_monotonic_s, event.frame_id)
                        return "C_REMOVED"
                return "VERIFYING_C_ABSENT"
            self._clear_c_timer()
        return "OBSERVATION_RECORDED"

    def _contact_eligible(self, event: FSMEvent) -> bool:
        hand_in = event.hand_in_contact_roi is True or event.hand_in_label_contact_roi is True
        grip = event.hand_in_left_grip_zone or event.hand_in_right_grip_zone
        return bool(event.hand_contact_candidate and hand_in and not grip and event.hand_detection_valid is not False)

    def _observation_valid(self, event: FSMEvent) -> tuple[bool, str]:
        if event.view_valid is not True:
            return False, "VIEW_INVALID"
        if event.quality_ok is not True:
            return False, "IMAGE_QUALITY_INVALID"
        if event.tube_present is not True:
            return False, "TUBE_NOT_PRESENT_OR_UNKNOWN"
        if event.label_window_present is not True:
            return False, "LABEL_WINDOW_INVALID"
        if event.label_roi_clear is not True or event.hand_occluded is not False:
            return False, "LABEL_ROI_OCCLUDED_OR_UNKNOWN"
        if event.config_match is not True:
            return False, "CONFIGURATION_MISMATCH"
        if event.camera_healthy is not True:
            return False, "CAMERA_HEALTH_UNKNOWN"
        if event.camera_settings_locked is not True:
            return False, "CAMERA_SETTINGS_NOT_VERIFIED"
        return True, ""

    def _capture_timing_reason(self, event: FSMEvent, now: float) -> str | None:
        capture = event.capture_monotonic_s
        if capture is None or event.frame_id is None:
            return "CAPTURE_TIMESTAMP_OR_FRAME_ID_MISSING"
        age_ms = (now - capture) * 1000.0
        if age_ms < -1.0 or age_ms > self.config.max_capture_age_ms:
            return "CAPTURE_TIMESTAMP_STALE_OR_WRONG_CLOCK_DOMAIN"
        if self._last_capture is not None:
            delta_ms = (capture - self._last_capture) * 1000.0
            if delta_ms <= 0:
                return "CAPTURE_TIMESTAMP_NOT_MONOTONIC"
            if self._last_frame_id is not None and event.frame_id <= self._last_frame_id:
                return "FRAME_ID_NOT_MONOTONIC"
            if delta_ms > self.config.max_frame_gap_ms:
                self._last_capture = capture
                self._last_frame_id = event.frame_id
                return "FRAME_GAP_EXCEEDED"
        self._last_capture = capture
        self._last_frame_id = event.frame_id
        return None

    def _invalidate(self, event: FSMEvent, now: float, reason: str, preserve_contact: bool) -> str:
        self._clear_stability()
        self._clear_c_timer()
        self._hand_left_start = None
        self._hand_left_confirmed = False
        if not preserve_contact:
            self._contact_seen = False
        self._resume_state = self._phase_for_resume()
        self._transition(FSMState.VIEW_INVALID, reason, event.kind, now, event.capture_monotonic_s, event.frame_id)
        return reason

    def _observe_stable(self, label: str, capture: float) -> bool:
        if self._stable_class != label:
            self._stable_class = label
            self._stable_start = capture
            self._stable_frames = 1
            return self._stable_frames >= self.config.min_stable_frames
        self._stable_frames += 1
        return self._stable_frames >= self.config.min_stable_frames

    def _stable_elapsed_ms(self, capture: float) -> float:
        return 0.0 if self._stable_start is None else max(0.0, (capture - self._stable_start) * 1000.0)

    def _clear_stability(self) -> None:
        self._stable_class = None
        self._stable_start = None
        self._stable_frames = 0

    def _clear_c_timer(self) -> None:
        self._c_absent_start = None
        self._c_absent_last = None

    def _clear_cycle_data(self) -> None:
        self.initial_complete_confirmed = False
        self._cycle_started = None
        self._inspection_started = None
        self._last_capture = None
        self._last_frame_id = None
        self._contact_seen = False
        self._hand_left_start = None
        self._hand_left_confirmed = False
        self._resume_state = FSMState.VERIFY_INITIAL
        self._clear_stability()
        self._clear_c_timer()

    def _transition(self, state: FSMState, reason: str, event_name: str, now: float,
                    capture: float | None, frame_id: int | None = None) -> None:
        previous = self.state
        self.state = state
        self.result_reason = reason
        if state == FSMState.PASS:
            self.outcome = "PASS"
        elif state in {FSMState.ALARM, FSMState.REVIEW_REQUIRED}:
            self.outcome = "ALARM"
        elif state == FSMState.FAULT:
            self.outcome = "FAULT"
        elif state == FSMState.VIEW_INVALID:
            self.outcome = "VIEW_INVALID"
        else:
            self.outcome = "WAITING"
        if previous != state or reason not in {"NO_STATE_CHANGE", "OBSERVATION_RECORDED"}:
            self._append_log({"kind": event_name}, now, capture, previous, state, reason,
                             record_type="STATE_TRANSITION", frame_id=frame_id)

    def _append_log(self, event: dict, now: float, capture: float | None,
                    before: FSMState, after: FSMState, reason: str,
                    record_type: str = "STATE_TRANSITION",
                    frame_id: int | None = None) -> None:
        self._log_sequence += 1
        record = {
            "sequence": self._log_sequence,
            "record_type": record_type,
            "event": event,
            "now_monotonic_s": now,
            "capture_monotonic_s": capture,
            "frame_id": frame_id,
            "state_before": before.value,
            "state_after": after.value,
            "outcome": self.outcome,
            "reason": reason,
            "cycle_id": self.cycle_id,
            "c_timer_ms": self.c_timer_ms,
            "initial_complete_confirmed": self.initial_complete_confirmed,
        }
        self.transition_records.append(record)
        if self._audit_sink is not None:
            self._pending_audit_records.append(dict(record))
            self.flush_audit_sink()
