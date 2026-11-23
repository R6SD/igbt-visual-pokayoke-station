"""Comprehensive regression tests for Wave 2 CodeRabbit findings (BE-01..BE-19, BE-46..BE-52) and cross-module helpers."""
from __future__ import annotations

import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from igbt_pokayoke.live_camera import (
    HEURISTIC_BACKEND_LABEL,
    HEURISTIC_MODEL_VERSION,
    LiveCameraEngine,
    generate_live_cycle_id,
)
from igbt_pokayoke.logic.fsm import FSMConfig, FSMEvent, FSMState, InspectionFSM
from igbt_pokayoke.runtime import (
    STANDARD_ROIS,
    YELLOW_REF_IMAGE,
    StationRuntime,
    build_scenario_run,
    default_view_config,
    load_image_unicode,
)
from igbt_pokayoke.vision.hand_contact import _valid_box, detect_contact_candidate
from igbt_pokayoke.vision.view_valid import (
    FrameObservation,
    ROISet,
    ViewConfig,
    ViewValidEvaluator,
    measure_label_quality,
)


def _make_yellow_tube_frame(
    *,
    width: int = 1920,
    height: int = 1080,
    with_white_label: bool = True,
    with_overexposed_patch: bool = False,
) -> np.ndarray:
    """Create a realistic synthetic dark-desk frame with a yellow IGBT tube and optional textured white label."""
    frame = np.full((height, width, 3), (45, 48, 52), dtype=np.uint8)
    # Yellow elongated horizontal tube in center (BGR yellow = (20, 200, 235))
    tx1, ty1 = int(width * 0.18), int(height * 0.44)
    tx2, ty2 = int(width * 0.82), int(height * 0.56)
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (20, 200, 235), -1)
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (10, 110, 140), 3)
    # Add subtle texture lines along the tube so Laplacian sharpness stays healthy
    for x in range(tx1 + 12, tx2 - 12, 24):
        cv2.line(frame, (x, ty1 + 6), (x, ty2 - 6), (15, 165, 200), 2)

    if with_white_label:
        lx1, ly1 = int(width * 0.42), int(height * 0.45)
        lx2, ly2 = int(width * 0.58), int(height * 0.55)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (225, 225, 225), -1)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (35, 35, 35), 3)
        # Simulated barcode / printed markings on label (high edge density, low dark_stroke_ratio)
        for idx in range(6):
            bx = lx1 + 18 + idx * 42
            cv2.line(frame, (bx, ly1 + 18), (bx, ly2 - 18), (45, 45, 45), 2)
        cv2.putText(
            frame,
            "IGBT-2026",
            (lx1 + 24, (ly1 + ly2) // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

    if with_overexposed_patch:
        lx1, ly1 = int(width * 0.42), int(height * 0.45)
        lx2, ly2 = int(width * 0.58), int(height * 0.55)
        gx2 = lx1 + max(8, int((lx2 - lx1) * 0.12))
        gy2 = ly1 + max(8, int((ly2 - ly1) * 0.55))
        frame[ly1:gy2, lx1:gx2] = 255
    return frame


class VisionGateTests(unittest.TestCase):
    def test_BE46_hand_contact_missing_grip_zones_no_typeerror(self) -> None:
        self.assertFalse(_valid_box(None))
        self.assertTrue(_valid_box((0.1, 0.1, 0.3, 0.3)))
        detections = [{"class_name": "hand_glove", "score": 0.95, "box": (0.42, 0.42, 0.58, 0.58)}]

        res_left_none = detect_contact_candidate(
            detections,
            label_contact_roi=STANDARD_ROIS.label_contact_roi,
            left_grip_zone=None,
            right_grip_zone=STANDARD_ROIS.right_grip_zone,
        )
        self.assertFalse(res_left_none.hand_contact_candidate)
        self.assertEqual(res_left_none.reason, "both_grip_zones_must_be_calibrated")

        res_right_none = detect_contact_candidate(
            detections,
            label_contact_roi=STANDARD_ROIS.label_contact_roi,
            left_grip_zone=STANDARD_ROIS.left_grip_zone,
            right_grip_zone=None,
        )
        self.assertFalse(res_right_none.hand_contact_candidate)
        self.assertEqual(res_right_none.reason, "both_grip_zones_must_be_calibrated")

    def test_BE47_view_valid_rejects_full_frame_occluder_covering_label(self) -> None:
        frame = load_image_unicode(YELLOW_REF_IMAGE)
        h, w = frame.shape[:2]
        rois = ROISet(
            tube_roi=(0.16, 0.42, 0.84, 0.58),
            label_roi=(0.45, 0.45, 0.55, 0.55),
            label_contact_roi=(0.40, 0.38, 0.60, 0.62),
            left_grip_zone=(0.16, 0.34, 0.31, 0.82),
            right_grip_zone=(0.69, 0.34, 0.84, 0.82),
        )
        cfg = ViewConfig(
            camera_device_id="CAM_STATION_01",
            calibrated_width=w,
            calibrated_height=h,
            roi_config_version="ROI_V001",
            rois=rois,
            quality_thresholds_approved=True,
            orientation_detector_ready=True,
            min_brightness=20.0,
            max_brightness=240.0,
            max_overexposed_fraction=0.20,
            max_underexposed_fraction=0.20,
            min_laplacian_variance=20.0,
            min_stable_frames=2,
            stable_window_ms=100.0,
        )
        evaluator = ViewValidEvaluator(cfg)

        def make_obs(fid: int, ts: float, occluders: tuple[tuple[float, float, float, float], ...]) -> FrameObservation:
            return FrameObservation(
                frame_id=fid,
                captured_monotonic_s=ts,
                camera_device_id="CAM_STATION_01",
                roi_config_version="ROI_V001",
                camera_healthy=True,
                tube_present=True,
                label_window_present=True,
                label_occluded=False,
                front_face_confirmed=True,
                orientation_reference_visible=True,
                orientation_evidence_source="machine_reference",
                tube_center_norm=(0.50, 0.50),
                label_center_norm=(0.50, 0.50),
                occluder_boxes_norm=occluders,
            )

        # Full-frame occluder [0, 0, 1, 1] covers 100% of label_roi despite 1% of occluder area
        res_full = evaluator.evaluate(frame, make_obs(1, 10.0, ((0.0, 0.0, 1.0, 1.0),)))
        self.assertFalse(res_full.view_valid)
        self.assertIn("DETECTED_OCCLUDER_OVERLAPS_LABEL_ROI", res_full.reasons)

        # Small occluder partially overlapping label_roi still rejected
        evaluator.reset()
        res_small = evaluator.evaluate(frame, make_obs(1, 10.0, ((0.46, 0.46, 0.49, 0.49),)))
        self.assertFalse(res_small.view_valid)
        self.assertIn("DETECTED_OCCLUDER_OVERLAPS_LABEL_ROI", res_small.reasons)

        # Non-overlapping occluder outside label_roi is not flagged
        evaluator.reset()
        res_clear = evaluator.evaluate(frame, make_obs(1, 10.0, ((0.01, 0.01, 0.10, 0.10),)))
        self.assertNotIn("DETECTED_OCCLUDER_OVERLAPS_LABEL_ROI", res_clear.reasons)


class FSMHardeningTests(unittest.TestCase):
    def _make_fsm(self, **cfg_overrides) -> tuple[InspectionFSM, list[float]]:
        clock = [100.0]
        params = dict(
            initial_complete_ms=100,
            stable_state_ms=100,
            min_stable_frames=2,
            c_absent_duration_ms=1200,
            max_frame_gap_ms=200,
            hand_leave_stable_ms=250,
            cycle_timeout_s=30,
            action_timeout_s=20,
        )
        params.update(cfg_overrides)
        return InspectionFSM(FSMConfig(**params), clock=lambda: clock[0]), clock

    @staticmethod
    def _obs(ts: float, frame_id: int, **overrides) -> FSMEvent:
        fields = dict(
            kind="observation",
            capture_monotonic_s=ts,
            frame_id=frame_id,
            label_class="complete",
            view_valid=True,
            quality_ok=True,
            camera_healthy=True,
            camera_settings_locked=True,
            config_match=True,
            tube_present=True,
            label_window_present=True,
            label_roi_clear=True,
            hand_occluded=False,
            hand_contact_candidate=False,
            hand_in_contact_roi=False,
            hand_in_left_grip_zone=False,
            hand_in_right_grip_zone=False,
            pipe_color="yellow",
        )
        fields.update(overrides)
        return FSMEvent(**fields)

    def test_BE12_bounded_observation_retention_and_audit_sink(self) -> None:
        streamed: list[dict] = []
        clock = [100.0]
        fsm = InspectionFSM(
            FSMConfig(
                initial_complete_ms=100,
                stable_state_ms=100,
                min_stable_frames=2,
                max_log_records=40,
            ),
            clock=lambda: clock[0],
            audit_sink=streamed.append,
        )
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_BOUND"), now_monotonic=clock[0])
        fid = 0
        for cls in ("complete", "complete"):
            clock[0] += 0.1
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class=cls), now_monotonic=clock[0])
        fsm.process(FSMEvent(kind="start_inspection"), now_monotonic=clock[0])
        for cls in ("crossed_out", "crossed_out"):
            clock[0] += 0.1
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class=cls), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.PASS)

        # Simulate 250 unattended post-PASS observations
        for _ in range(250):
            clock[0] += 0.033
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class="crossed_out"), now_monotonic=clock[0])

        self.assertEqual(len(fsm.transition_records), 40)
        self.assertEqual(len(fsm.transition_log()), 40)
        self.assertGreater(len(streamed), 250)
        seqs = [r["sequence"] for r in fsm.transition_log()]
        self.assertEqual(seqs, list(range(seqs[0], seqs[0] + len(seqs))))

    def test_BE13_restarts_hand_departure_timing_after_frame_gap(self) -> None:
        fsm, clock = self._make_fsm(hand_leave_stable_ms=250)
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_GAP"), now_monotonic=clock[0])
        fid = 0
        for _ in range(2):
            clock[0] += 0.1
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class="complete"), now_monotonic=clock[0])
        fsm.process(FSMEvent(kind="start_inspection"), now_monotonic=clock[0])

        # Contact -> PROCESSING
        clock[0] += 0.1
        fid += 1
        fsm.process(
            self._obs(clock[0], fid, hand_contact_candidate=True, hand_in_contact_roi=True, label_roi_clear=False, hand_occluded=True),
            now_monotonic=clock[0],
        )
        # Hand leaves -> starts _hand_left_start at t0
        clock[0] += 0.1
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.PROCESSING)

        # Frame gap > 200ms (0.35s) must clear _hand_left_start and _hand_left_confirmed
        clock[0] += 0.35
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VIEW_INVALID)
        self.assertIsNone(fsm._hand_left_start)
        self.assertFalse(fsm._hand_left_confirmed)

        # First valid frame after gap must restart hand-departure timer, NOT enter VERIFY_C_ABSENT immediately
        clock[0] += 0.1
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.PROCESSING)
        self.assertFalse(fsm._hand_left_confirmed)

        # After 100ms more (total 100ms since new departure), still waiting (< 250ms)
        clock[0] += 0.1
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.PROCESSING)

        # After 250ms+ new continuous clear departure -> enters VERIFY_C_ABSENT
        clock[0] += 0.15
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VERIFY_C_ABSENT)

    def test_BE14_resumes_verify_initial_after_frame_gap_and_subsequent_invalid(self) -> None:
        fsm, clock = self._make_fsm()
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_VI_GAP"), now_monotonic=clock[0])
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 1, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VERIFY_INITIAL)

        # Frame gap in VERIFY_INITIAL -> VIEW_INVALID, resume must stay VERIFY_INITIAL
        clock[0] += 0.35
        fsm.process(self._obs(clock[0], 2, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VIEW_INVALID)

        # Another invalid frame while already in VIEW_INVALID must still preserve VERIFY_INITIAL
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 3, view_valid=False, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VIEW_INVALID)

        # Next valid frame with crossed_out MUST restore VERIFY_INITIAL -> ALARM (NOT PASS!)
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 4, label_class="crossed_out"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.ALARM)
        self.assertEqual(fsm.result_reason, "INITIAL_LABEL_NOT_COMPLETE")

    def test_BE15_does_not_enter_processing_from_ready_before_start_inspection(self) -> None:
        fsm, clock = self._make_fsm(hand_leave_stable_ms=0)
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_READY"), now_monotonic=clock[0])
        fid = 0
        for _ in range(2):
            clock[0] += 0.1
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.READY)

        # Eligible contact in READY before start_inspection must NOT enter PROCESSING
        clock[0] += 0.1
        fid += 1
        fsm.process(
            self._obs(clock[0], fid, hand_contact_candidate=True, hand_in_contact_roi=True),
            now_monotonic=clock[0],
        )
        self.assertEqual(fsm.state, FSMState.READY)
        self.assertFalse(fsm._contact_seen)

        # Clear absent frames for 1.5s without start_inspection must NEVER pass
        for _ in range(15):
            clock[0] += 0.1
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.READY)
        self.assertNotEqual(fsm.outcome, "PASS")

        # After start_inspection, occluded eligible contact IS accepted into PROCESSING
        fsm.process(FSMEvent(kind="start_inspection"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.INSPECTING)
        clock[0] += 0.1
        fid += 1
        fsm.process(
            self._obs(
                clock[0],
                fid,
                hand_contact_candidate=True,
                hand_in_contact_roi=True,
                hand_occluded=True,
                label_roi_clear=False,
                view_valid=False,
            ),
            now_monotonic=clock[0],
        )
        self.assertTrue(fsm._contact_seen)
        clock[0] += 0.1
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VERIFY_C_ABSENT)

    def test_BE16_does_not_restore_ready_before_initial_confirmation(self) -> None:
        fsm, clock = self._make_fsm()
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_VI_INV"), now_monotonic=clock[0])
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 1, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VERIFY_INITIAL)
        self.assertFalse(fsm.initial_complete_confirmed)

        # Invalid view during VERIFY_INITIAL
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 2, view_valid=False), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VIEW_INVALID)

        # First valid frame restores VERIFY_INITIAL (NOT READY!)
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 3, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VERIFY_INITIAL)
        self.assertFalse(fsm.initial_complete_confirmed)

        # Second valid frame completes initial confirmation -> READY -> start_inspection works!
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 4, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.READY)
        self.assertTrue(fsm.initial_complete_confirmed)

        # Invalid view AFTER confirmation in READY restores READY
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 5, view_valid=False), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VIEW_INVALID)
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 6, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.READY)
        fsm.process(FSMEvent(kind="start_inspection"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.INSPECTING)

    def test_BE50_unrounded_duration_for_c_confirmation(self) -> None:
        fsm, clock = self._make_fsm(hand_leave_stable_ms=0)
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_C_EXACT"), now_monotonic=clock[0])
        fid = 0
        for _ in range(2):
            clock[0] += 0.1
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class="complete"), now_monotonic=clock[0])
        fsm.process(FSMEvent(kind="start_inspection"), now_monotonic=clock[0])
        clock[0] += 0.1
        fid += 1
        fsm.process(
            self._obs(clock[0], fid, hand_contact_candidate=True, hand_in_contact_roi=True, hand_occluded=True, label_roi_clear=False),
            now_monotonic=clock[0],
        )
        # Start C timer at t0
        t0 = clock[0] + 0.1
        clock[0] = t0
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VERIFY_C_ABSENT)

        # Advance in <= 0.15s steps to t0 + 1.1996s (1199.6 ms -> rounds to 1200, MUST NOT PASS!)
        for offset in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.1996):
            clock[0] = t0 + offset
            fid += 1
            fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.VERIFY_C_ABSENT)
        self.assertNotEqual(fsm.outcome, "PASS")

        # Advance past 1200.0 ms -> PASS
        clock[0] = t0 + 1.2000
        fid += 1
        fsm.process(self._obs(clock[0], fid, label_class="absent"), now_monotonic=clock[0])
        self.assertEqual(fsm.state, FSMState.PASS)
        self.assertEqual(fsm.result_reason, "C_REMOVED")

    def test_BE51_frame_id_baseline_not_advanced_on_rejected_gap(self) -> None:
        fsm, clock = self._make_fsm()
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_FID"), now_monotonic=clock[0])
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 100, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm._last_frame_id, 100)

        # Gap frame with lower frame_id=5 must be rejected and NOT overwrite baseline 100
        clock[0] += 0.35
        fsm.process(self._obs(clock[0], 5, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.result_reason, "FRAME_ID_NOT_MONOTONIC")
        self.assertEqual(fsm._last_frame_id, 100)

        # Follow-up frame_id=6 (>5, <=100) must STILL be rejected!
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 6, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.result_reason, "FRAME_ID_NOT_MONOTONIC")
        self.assertEqual(fsm._last_frame_id, 100)

        # Gap frame with higher frame_id=105 triggers FRAME_GAP_EXCEEDED without advancing _last_frame_id
        clock[0] += 0.35
        fsm.process(self._obs(clock[0], 105, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.result_reason, "FRAME_GAP_EXCEEDED")
        self.assertEqual(fsm._last_frame_id, 100)

        # Replaying old frame_id=100 is still rejected
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 100, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm.result_reason, "FRAME_ID_NOT_MONOTONIC")

        # Advancing past last accepted baseline (101 > 100) recovers cleanly
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 101, label_class="complete"), now_monotonic=clock[0])
        self.assertEqual(fsm._last_frame_id, 101)
        self.assertEqual(fsm.state, FSMState.VERIFY_INITIAL)

    def test_BE55_state_transition_records_include_triggering_frame_id(self) -> None:
        fsm, clock = self._make_fsm()
        fsm.process(FSMEvent(kind="new_cycle", cycle_id="CYC_FID_LOG"), now_monotonic=clock[0])
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 11, label_class="complete"), now_monotonic=clock[0])
        clock[0] += 0.1
        fsm.process(self._obs(clock[0], 12, label_class="complete"), now_monotonic=clock[0])
        transitions = [r for r in fsm.transition_log() if r["record_type"] == "STATE_TRANSITION"]
        ready_rec = next(r for r in transitions if r["state_after"] == "READY")
        self.assertEqual(ready_rec["frame_id"], 12)


class StationRuntimeHardeningTests(unittest.TestCase):
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

    def _obs(self, dt: float = 0.1) -> FrameObservation:
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
            label_occluded=False,
            front_face_confirmed=True,
            orientation_reference_visible=True,
            orientation_evidence_source="machine_reference",
            tube_center_norm=(0.50, 0.50),
            label_center_norm=(0.50, 0.50),
        )

    def test_BE17_rejected_new_cycle_leaves_active_cycle_and_trackers_unchanged(self) -> None:
        self.runtime.start_cycle("CYC_A", pipe_color="yellow", operator_anon_id="OP_A", now_monotonic=self.now)
        for _ in range(3):
            self.runtime.step(self.frame, self._obs(), detections=self.grips, label_status="complete", now_monotonic=self.now)
        history_len_before = len(self.runtime.view_evaluator._history)
        self.assertGreater(history_len_before, 0)

        rejected = self.runtime.start_cycle("CYC_B", pipe_color="blue", operator_anon_id="OP_B", now_monotonic=self.now)
        self.assertEqual(self.runtime.fsm.cycle_id, "CYC_A")
        self.assertEqual(self.runtime.pipe_color, "yellow")
        self.assertEqual(self.runtime.operator_anon_id, "OP_A")
        self.assertEqual(len(self.runtime.view_evaluator._history), history_len_before)
        self.assertEqual(rejected["cycle_id"], "CYC_A")
        self.assertEqual(rejected["pipe_color"], "yellow")
        self.assertEqual(rejected["operator_anon_id"], "OP_A")
        self.assertEqual(rejected["event_type"], "NEW_CYCLE_REJECTED")

    def test_BE18_unauthorized_reset_does_not_mutate_cycle_or_trackers(self) -> None:
        self.runtime.start_cycle("CYC_ACT", pipe_color="yellow", now_monotonic=self.now)
        for _ in range(5):
            self.runtime.step(self.frame, self._obs(), detections=self.grips, label_status="complete", now_monotonic=self.now)
        self.runtime.start_inspection(now_monotonic=self.now)
        before_snap = self.runtime.step(
            self.frame, self._obs(), detections=self.grips, label_status="complete", now_monotonic=self.now
        )
        history_len_before = len(self.runtime.view_evaluator._history)

        denied = self.runtime.reset(authorized=False, now_monotonic=self.now)
        self.assertEqual(self.runtime.fsm.state, FSMState.INSPECTING)
        self.assertEqual(self.runtime.fsm.cycle_id, "CYC_ACT")
        self.assertEqual(len(self.runtime.view_evaluator._history), history_len_before)
        self.assertEqual(denied, before_snap)
        self.assertNotEqual(denied.get("event_type"), "AUTHORIZED_RESET")

        # Authorized reset still works cleanly
        done = self.runtime.reset(authorized=True, now_monotonic=self.now)
        self.assertEqual(self.runtime.fsm.state, FSMState.WAIT_PART)
        self.assertEqual(done["event_type"], "AUTHORIZED_RESET")

    def test_BE19_step_defaults_now_monotonic_to_runtime_clock(self) -> None:
        self.runtime.start_cycle("CYC_STALE", pipe_color="yellow", now_monotonic=self.now)
        stale_obs = self._obs(dt=0.1)
        # Advance runtime clock 1.0s past capture timestamp (> 500ms max_capture_age_ms)
        self.now += 1.0
        snap = self.runtime.step(
            self.frame,
            stale_obs,
            detections=self.grips,
            label_status="complete",
        )
        self.assertEqual(snap["fsm_state"], "VIEW_INVALID")
        self.assertEqual(snap["result_reason"], "CAPTURE_TIMESTAMP_STALE_OR_WRONG_CLOCK_DOMAIN")

    def test_BE52_start_inspection_emits_event_only_when_accepted(self) -> None:
        self.runtime.start_cycle("CYC_START", pipe_color="yellow", now_monotonic=self.now)
        # Rejected during VERIFY_INITIAL
        rej = self.runtime.start_inspection(now_monotonic=self.now)
        self.assertEqual(rej["fsm_state"], "VERIFY_INITIAL")
        self.assertEqual(rej["event_type"], "INSPECTION_START_REJECTED")

        # Confirm initial complete -> READY -> accepted start_inspection emits INSPECTION_STARTED
        for _ in range(5):
            self.runtime.step(self.frame, self._obs(), detections=self.grips, label_status="complete", now_monotonic=self.now)
        acc = self.runtime.start_inspection(now_monotonic=self.now)
        self.assertEqual(acc["fsm_state"], "INSPECTING")
        self.assertEqual(acc["event_type"], "INSPECTION_STARTED")

    def test_BE54_glare_scenario_returns_evaluated_glare_evidence_bytes(self) -> None:
        run = build_scenario_run("glare_invalid")
        raw_bytes = run.get("evidence_image_bytes")
        self.assertIsInstance(raw_bytes, bytes)
        assert raw_bytes is not None
        decoded = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        q = measure_label_quality(decoded, STANDARD_ROIS.label_roi)
        self.assertIsNotNone(q)
        assert q is not None
        self.assertGreater(q.overexposed_fraction, 0.08)


class LiveCameraHardeningTests(unittest.TestCase):
    def test_BE01_mirror_change_applied_between_frames_without_midframe_crash(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)
        # Prime tracking state
        engine._process_frame(frame, time.monotonic())
        self.assertIsNotNone(engine._bg_small)
        self.assertTrue(engine.get_state()["mirror_horizontal"])

        # Intercept _track_dynamic_targets to toggle mirror mid-frame
        orig_track = engine._track_dynamic_targets
        toggled = {"done": False}

        def track_and_toggle(f, work_mode=None):
            if not toggled["done"]:
                toggled["done"] = True
                engine.set_mirror_horizontal(False)
                # Setter must NOT null _bg_small while frame is executing!
                self.assertIsNotNone(engine._bg_small)
            return orig_track(f, work_mode=work_mode)

        engine._track_dynamic_targets = track_and_toggle  # type: ignore[method-assign]
        ok = engine._process_frame(frame, time.monotonic())
        self.assertTrue(ok)
        # In-flight frame was computed with mirror=True -> published under mirror=True!
        self.assertTrue(engine.get_state()["mirror_horizontal"])
        self.assertTrue(engine._pending_tracker_reset)

        # Next frame applies the mirror change cleanly at the frame boundary
        ok2 = engine._process_frame(frame, time.monotonic())
        self.assertTrue(ok2)
        self.assertFalse(engine.get_state()["mirror_horizontal"])
        self.assertFalse(engine._pending_tracker_reset)

    def test_BE02_refuse_restart_while_previous_capture_worker_alive(self) -> None:
        read_entered = threading.Event()
        release_read = threading.Event()
        opened_count = {"n": 0}
        state_updates_after_stop = {"n": 0}

        class HangingCapture:
            def __init__(self, *_args, **_kwargs) -> None:
                opened_count["n"] += 1

            def isOpened(self) -> bool:
                return True

            def set(self, *_args, **_kwargs) -> bool:
                return True

            def read(self):
                read_entered.set()
                release_read.wait(timeout=3.0)
                return True, np.zeros((360, 640, 3), dtype=np.uint8)

            def release(self) -> None:
                return None

        engine = LiveCameraEngine(config_loader=lambda: {})
        orig_process = engine._process_frame

        def spy_process(f, ts, stop_event=None):
            state_updates_after_stop["n"] += 1
            return orig_process(f, ts, stop_event=stop_event)

        engine._process_frame = spy_process  # type: ignore[method-assign]
        with mock.patch("igbt_pokayoke.live_camera.cv2.VideoCapture", HangingCapture):
            self.assertTrue(engine.start())
            self.assertTrue(read_entered.wait(timeout=2.0))
            # Timeout expires while HangingCapture.read() is still blocked
            exited = engine.stop(timeout=0.05)
            self.assertFalse(exited)
            # Refuses to spawn a second worker while the first is still alive
            self.assertFalse(engine.start())
            self.assertEqual(opened_count["n"], 1)

            # Release hanging read -> old worker sees its own stop_event and exits without mutating state
            release_read.set()
            assert engine._thread is not None
            engine._thread.join(timeout=2.0)
            self.assertFalse(engine._thread.is_alive())
            self.assertEqual(state_updates_after_stop["n"], 0)

            # Now restart is allowed cleanly
            read_entered.clear()
            self.assertTrue(engine.start())
            self.assertTrue(read_entered.wait(timeout=2.0))
            self.assertTrue(engine.stop(timeout=2.0))
            self.assertEqual(opened_count["n"], 2)

    def test_BE03_unique_live_cycle_ids(self) -> None:
        ids = {generate_live_cycle_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        engine = LiveCameraEngine(config_loader=lambda: {})
        reset_ids = {engine.trigger_new_cycle()["cycle_id"] for _ in range(10)}
        self.assertEqual(len(reset_ids), 10)
        for cid in reset_ids:
            self.assertRegex(cid, r"^LIVE_\d{8}_\d{6}_\d{4}_[0-9A-F]{8}$")

    def test_BE04_track_tube_independently_through_label_removal(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        engine.set_work_mode("debug")
        frame_with_label = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)
        frame_no_label = _make_yellow_tube_frame(width=640, height=360, with_white_label=False)
        empty_desk = np.full((360, 640, 3), (45, 48, 52), dtype=np.uint8)

        # Lock on tube with visible white label
        for _ in range(3):
            out = engine._track_dynamic_targets(frame_with_label, work_mode="debug")
        self.assertTrue(out[7])  # target_locked
        self.assertIsNotNone(out[0])  # tube_box
        self.assertIsNotNone(out[1])  # label_box

        # Remove white label completely for 20 frames (far beyond old 6-frame TTL!)
        for _ in range(20):
            out = engine._track_dynamic_targets(frame_no_label, work_mode="debug")
            self.assertTrue(out[7], "Target lock dropped when label was removed")
            self.assertIsNotNone(out[0], "Tube box dropped when label was removed")
            self.assertIsNotNone(out[1], "Projected label window dropped when label was removed")

        # Removing the entire tube from view eventually unlocks cleanly
        for _ in range(18):
            out = engine._track_dynamic_targets(empty_desk, work_mode="debug")
        self.assertFalse(out[7])

    def test_BE05_and_BE10_dark_covered_classification_and_model_provenance(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        # Dark crop with ~0% white pixels (simulating black opaque cover tape over label)
        dark_crop = np.full((120, 200, 3), (35, 35, 35), dtype=np.uint8)

        class FakeCoveredSession:
            def run(self, _outputs, _inputs):
                # CLASSES = ["complete", "crossed_out", "covered", "absent"]
                return [np.array([[0.0, 0.0, 8.0, 0.0]], dtype=np.float32)]

        # 1. ONNX predicts covered at ~99.9% on dark crop -> MUST return "covered", NEVER "complete"
        engine._ort_session = FakeCoveredSession()
        engine._onnx_model_version = "model-A"
        cls, conf, _color, _ms, model_ver, backend = engine._classify_live_label_roi(
            dark_crop, target_locked=True, contact_seen=False, label_occluded=False
        )
        self.assertEqual(cls, "covered")
        self.assertGreater(conf or 0.0, 0.95)
        self.assertEqual(model_ver, "model-A")
        self.assertIn("model-A", backend)

        # 2. Hot-swap to model-B mid-inference -> prediction still records model-A (BE-10)
        class SwappingSession:
            def run(self, _outputs, _inputs):
                engine._onnx_model_version = "model-B"
                return [np.array([[0.0, 0.0, 8.0, 0.0]], dtype=np.float32)]

        engine._ort_session = SwappingSession()
        engine._onnx_model_version = "model-A"
        cls2, _c2, _col2, _ms2, model_ver2, backend2 = engine._classify_live_label_roi(
            dark_crop, target_locked=True, contact_seen=False, label_occluded=False
        )
        self.assertEqual(cls2, "covered")
        self.assertEqual(model_ver2, "model-A")
        self.assertIn("model-A", backend2)

        # 3. Failing ONNX session falls back to heuristic AND attributes heuristic backend + returns "unknown" on dark crop
        class FailingSession:
            def run(self, _outputs, _inputs):
                raise RuntimeError("ort crash")

        engine._ort_session = FailingSession()
        engine._onnx_model_version = "model-broken"
        cls3, _c3, _col3, _ms3, model_ver3, backend3 = engine._classify_live_label_roi(
            dark_crop, target_locked=True, contact_seen=False, label_occluded=False
        )
        self.assertEqual(cls3, "unknown")
        self.assertEqual(model_ver3, HEURISTIC_MODEL_VERSION)
        self.assertEqual(backend3, HEURISTIC_BACKEND_LABEL)

    def test_BE44_load_onnx_model_verifies_sha256_before_mutating_session(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        sentinel_session = object()
        engine._ort_session = sentinel_session
        engine._onnx_model_version = "active-v1"

        with tempfile.TemporaryDirectory() as tmp:
            fake_onnx = Path(tmp) / "label_classifier.onnx"
            fake_onnx.write_bytes(b"not-a-valid-onnx-payload")
            bad_sha = "0" * 64
            with self.assertRaises(ValueError):
                engine.load_onnx_model(fake_onnx, model_version="v2", expected_sha256=bad_sha)
            # Active session and version untouched!
            self.assertIs(engine._ort_session, sentinel_session)
            self.assertEqual(engine._onnx_model_version, "active-v1")

            # Missing file leaves active model untouched
            with self.assertRaises(FileNotFoundError):
                engine.load_onnx_model(Path(tmp) / "missing.onnx", model_version="v3", expected_sha256=hashlib.sha256(b"x").hexdigest())
            self.assertIs(engine._ort_session, sentinel_session)
            self.assertEqual(engine._onnx_model_version, "active-v1")

    def test_BE06_reject_inflight_observation_after_cycle_reset(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)

        # Hook classification to trigger a cycle reset while frame is in-flight
        def classify_with_reset(*_args, **_kwargs):
            engine.trigger_new_cycle("yellow")
            return "absent", 0.95, "yellow", 1.0, HEURISTIC_MODEL_VERSION, HEURISTIC_BACKEND_LABEL

        engine._classify_live_label_roi = classify_with_reset  # type: ignore[method-assign]
        committed = engine._process_frame(frame, time.monotonic())
        self.assertFalse(committed)
        # Replacement FSM must NOT have transitioned to ALARM from the stale "absent" observation
        self.assertNotEqual(engine.fsm.state, FSMState.ALARM)
        self.assertFalse(engine.get_state()["alarm_latched"])

    def test_BE07_live_calibration_gate_strict_vs_debug(self) -> None:
        uncalibrated_cfg = {
            "config_version": 0,
            "roi_config_version": "ROI_UNCALIBRATED",
            "camera_device_id": "CAM_STATION_01",
            "camera_width": 1408,
            "camera_height": 768,
            "calibrated_at_utc": None,
            "camera_covers_both_ends": True,
            "rois": {},
            "thresholds": {},
        }
        engine = LiveCameraEngine(config_loader=lambda: uncalibrated_cfg)
        frame_1080p = _make_yellow_tube_frame(width=1920, height=1080, with_white_label=True)

        # In strict mode with uncalibrated 1408x768 default -> view_valid=False, camera_settings_locked=False, blocks PASS!
        engine.set_work_mode("strict")
        cfg_ok, locked, cal_reasons = engine._evaluate_live_calibration(uncalibrated_cfg, 1920, 1080, "strict")
        self.assertFalse(cfg_ok)
        self.assertFalse(locked)
        self.assertIn("ROI_UNCALIBRATED", cal_reasons)
        self.assertIn("CALIBRATION_RESOLUTION_MISMATCH", cal_reasons)

        # Drive locked observation in strict mode -> must enter FAULT/blocked, never PASS
        engine._track_dynamic_targets = lambda f, work_mode=None: (  # type: ignore[method-assign]
            (0.18, 0.44, 0.82, 0.56),
            (0.42, 0.45, 0.58, 0.55),
            (0.18, 0.42, 0.28, 0.62),
            (0.72, 0.42, 0.82, 0.62),
            None,
            False,
            False,
            True,
            [],
        )
        engine._process_frame(frame_1080p, time.monotonic())
        st_strict = engine.get_state()
        self.assertFalse(st_strict["view_valid"])
        self.assertFalse(st_strict["camera_settings_locked"])
        self.assertIn("ROI_UNCALIBRATED", st_strict["view_valid_reasons"])
        self.assertNotEqual(st_strict["outcome"], "PASS")

        # In debug desk-testing mode -> allowed to pass validation despite uncalibrated default
        engine.set_work_mode("debug")
        cfg_ok_dbg, locked_dbg, reasons_dbg = engine._evaluate_live_calibration(uncalibrated_cfg, 1920, 1080, "debug")
        self.assertTrue(cfg_ok_dbg)
        self.assertTrue(locked_dbg)
        self.assertEqual(reasons_dbg, [])

        # In strict mode with matching calibrated config -> allowed!
        calibrated_cfg = {
            "config_version": 1,
            "roi_config_version": "ROI_V001",
            "camera_device_id": engine.camera_device_id,
            "camera_width": 1920,
            "camera_height": 1080,
            "calibrated_at_utc": "2026-11-23T10:00:00.000Z",
            "camera_covers_both_ends": True,
            "rois": {
                "tube_roi": [0.16, 0.42, 0.84, 0.58],
                "label_roi": [0.43, 0.45, 0.57, 0.55],
                "label_contact_roi": [0.40, 0.38, 0.60, 0.62],
                "left_grip_zone": [0.16, 0.34, 0.31, 0.82],
                "right_grip_zone": [0.69, 0.34, 0.84, 0.82],
            },
            "thresholds": {},
        }
        cfg_ok_cal, locked_cal, reasons_cal = engine._evaluate_live_calibration(calibrated_cfg, 1920, 1080, "strict")
        self.assertTrue(cfg_ok_cal)
        self.assertTrue(locked_cal)
        self.assertEqual(reasons_cal, [])

        # Resolution mismatch on calibrated config in strict mode -> rejected!
        cfg_ok_mm, locked_mm, reasons_mm = engine._evaluate_live_calibration(calibrated_cfg, 1280, 720, "strict")
        self.assertFalse(cfg_ok_mm)
        self.assertFalse(locked_mm)
        self.assertIn("CALIBRATION_RESOLUTION_MISMATCH", reasons_mm)

    def test_BE08_per_candidate_grip_status_enables_gloved_c_removal(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        engine.set_work_mode("debug")
        engine.fsm = InspectionFSM(
            config=FSMConfig(
                initial_complete_ms=100,
                stable_state_ms=100,
                min_stable_frames=2,
                hand_leave_stable_ms=100,
            )
        )
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)

        state_ctrl = {"contact": False, "cls": "complete"}

        def fake_track(_f, work_mode=None):
            return (
                (0.18, 0.44, 0.82, 0.56),
                (0.42, 0.45, 0.58, 0.55),
                (0.44, 0.42, 0.56, 0.58) if state_ctrl["contact"] else (0.18, 0.42, 0.28, 0.62),
                (0.72, 0.42, 0.82, 0.62),
                None,
                state_ctrl["contact"],
                False,
                True,
                [],
            )

        def fake_cls(*_args, **_kwargs):
            return state_ctrl["cls"], 0.96, "yellow", 1.0, HEURISTIC_MODEL_VERSION, HEURISTIC_BACKEND_LABEL

        engine._track_dynamic_targets = fake_track  # type: ignore[method-assign]
        engine._classify_live_label_roi = fake_cls  # type: ignore[method-assign]

        now_val = [1000.0]
        with mock.patch("igbt_pokayoke.live_camera.time.monotonic", side_effect=lambda: now_val[0]):
            # 1. Confirm initial complete -> READY -> auto start_inspection -> INSPECTING
            for _ in range(3):
                now_val[0] += 0.1
                engine._process_frame(frame, now_val[0])
            self.assertEqual(engine.fsm.state, FSMState.INSPECTING)

            # 2. Gloved hand enters label contact ROI while left_grip_active is True for display
            state_ctrl["contact"] = True
            for _ in range(2):
                now_val[0] += 0.1
                engine._process_frame(frame, now_val[0])
            self.assertTrue(engine.get_state()["left_grip_active"])
            self.assertTrue(engine.fsm._contact_seen)

            # 3. Hand leaves and label is absent for >= 1200ms -> PASS (C_REMOVED)
            state_ctrl["contact"] = False
            state_ctrl["cls"] = "absent"
            for _ in range(20):
                now_val[0] += 0.1
                engine._process_frame(frame, now_val[0])
                if engine.fsm.state == FSMState.PASS:
                    break
            self.assertEqual(engine.fsm.state, FSMState.PASS)
            self.assertEqual(engine.fsm.result_reason, "C_REMOVED")

    def test_BE09_frame_age_measured_at_fsm_consumption_time(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        engine.set_work_mode("debug")
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)
        engine._track_dynamic_targets = lambda _f, work_mode=None: (  # type: ignore[method-assign]
            (0.18, 0.44, 0.82, 0.56),
            (0.42, 0.45, 0.58, 0.55),
            (0.18, 0.42, 0.28, 0.62),
            (0.72, 0.42, 0.82, 0.62),
            None,
            False,
            False,
            True,
            [],
        )
        # Simulate 1.0s inference stall between capture_ts and consumption time
        stale_capture_ts = time.monotonic() - 1.0
        engine._process_frame(frame, stale_capture_ts)
        self.assertEqual(engine.fsm.state, FSMState.VIEW_INVALID)
        self.assertEqual(engine.fsm.result_reason, "CAPTURE_TIMESTAMP_STALE_OR_WRONG_CLOCK_DOMAIN")

    def test_BE11_retain_transitions_on_hud_failure_and_persistence_retry(self) -> None:
        attempts: list[tuple[str, str]] = []

        def flaky_sink(snapshot: dict, _prev_fsm: str | None):
            ev_id = snapshot["transition_event_id"]
            attempts.append((ev_id, snapshot["fsm_state"]))
            if len(attempts) == 1:
                raise RuntimeError("transient sqlite busy")
            if len(attempts) == 2:
                return False
            return True

        engine = LiveCameraEngine(config_loader=lambda: {}, on_cycle_update=flaky_sink)
        engine.set_work_mode("strict")
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)

        # Trigger bare-hand violation ALARM while HUD rendering raises
        engine._track_dynamic_targets = lambda _f, work_mode=None: (  # type: ignore[method-assign]
            (0.18, 0.44, 0.82, 0.56),
            (0.42, 0.45, 0.58, 0.55),
            None,
            None,
            (0.42, 0.44, 0.56, 0.58),
            False,
            True,  # bare_hand_touching_part -> ALARM
            True,
            [],
        )
        engine._draw_hud = mock.Mock(side_effect=RuntimeError("HUD render failure"))  # type: ignore[method-assign]

        # Frame 1: ALARM latches, HUD fails, persistence raises -> transition remains queued
        engine._process_frame(frame, time.monotonic())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0][1], "ALARM")
        self.assertEqual(len(engine._pending_transitions), 1)

        # Frame 2: ALARM already latched (changed=False), retry returns False -> still queued with same event_id
        engine._process_frame(frame, time.monotonic())
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0][0], attempts[1][0])
        self.assertEqual(len(engine._pending_transitions), 1)

        # Frame 3: retry succeeds -> popped cleanly, exact same stable event_id
        engine._process_frame(frame, time.monotonic())
        self.assertEqual(len(attempts), 3)
        self.assertEqual(attempts[0][0], attempts[2][0])
        self.assertEqual(len(engine._pending_transitions), 0)

    def test_BE48_preserve_explicit_zero_overexposed_threshold(self) -> None:
        cfg = {
            "thresholds": {
                "min_laplacian_variance": 5.0,
                "min_brightness": 10.0,
                "max_brightness": 250.0,
                "max_overexposed_fraction": 0,
            }
        }
        engine = LiveCameraEngine(config_loader=lambda: cfg)
        engine.set_work_mode("debug")
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True, with_overexposed_patch=True)
        engine._track_dynamic_targets = lambda _f, work_mode=None: (  # type: ignore[method-assign]
            (0.18, 0.44, 0.82, 0.56),
            (0.42, 0.45, 0.58, 0.55),
            None,
            None,
            None,
            False,
            False,
            True,
            [],
        )
        engine._process_frame(frame, time.monotonic())
        st = engine.get_state()
        self.assertGreater(st["overexposed_fraction"] or 0.0, 0.0)
        self.assertLess(st["overexposed_fraction"] or 1.0, 0.18)
        self.assertIn("OVEREXPOSED", st["view_valid_reasons"])
        self.assertFalse(st["quality_ok"])

    def test_BE49_do_not_latch_alarm_computed_under_previous_work_mode(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        engine.set_work_mode("strict")
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)

        # Frame detects bare_hand_touching_part in strict mode, then user switches to debug before commit
        engine._track_dynamic_targets = lambda _f, work_mode=None: (  # type: ignore[method-assign]
            (0.18, 0.44, 0.82, 0.56),
            (0.42, 0.45, 0.58, 0.55),
            None,
            None,
            (0.44, 0.44, 0.56, 0.58),
            False,
            work_mode == "strict",
            True,
            [],
        )

        def switch_mode_mid_frame(*_args, **_kwargs):
            engine.set_work_mode("debug")
            return "complete", 0.92, "yellow", 1.0, HEURISTIC_MODEL_VERSION, HEURISTIC_BACKEND_LABEL

        engine._classify_live_label_roi = switch_mode_mid_frame  # type: ignore[method-assign]
        committed = engine._process_frame(frame, time.monotonic())
        self.assertFalse(committed)
        st = engine.get_state()
        self.assertEqual(st["work_mode"], "debug")
        self.assertFalse(st["alarm_latched"])
        self.assertFalse(st["bare_hand_violation"])

    def test_BE34_and_BE35_atomic_snapshot_and_condition_wait_helpers(self) -> None:
        engine = LiveCameraEngine(config_loader=lambda: {})
        frame = _make_yellow_tube_frame(width=640, height=360, with_white_label=True)

        # Before any frame: returns (None, None, state)
        f0, c0, s0 = engine.get_atomic_capture_snapshot()
        self.assertIsNone(f0)
        self.assertIsNone(c0)
        self.assertIsInstance(s0, dict)

        # Invalid explicit crop box raises ValueError (BE-57)
        with self.assertRaises(ValueError):
            engine.get_atomic_capture_snapshot(box=[0.7, 0.5, 0.3, 0.8])

        # Process a frame and verify atomic capture snapshot
        engine._process_frame(frame, time.monotonic())
        f_bgr, crop_bgr, snap = engine.get_atomic_capture_snapshot(encode_jpeg=False)
        self.assertIsInstance(f_bgr, np.ndarray)
        self.assertIsInstance(crop_bgr, np.ndarray)
        self.assertEqual(snap["frame_id"], 1)

        f_jpg, crop_bgr2, snap2 = engine.get_atomic_capture_snapshot(encode_jpeg=True)
        self.assertIsInstance(f_jpg, bytes)
        assert isinstance(f_jpg, bytes)
        self.assertTrue(f_jpg.startswith(b"\xff\xd8"))
        self.assertIsInstance(crop_bgr2, np.ndarray)
        self.assertEqual(snap2["frame_id"], 1)

        # Non-blocking cache read + condition wait (BE-34)
        hud_jpg, seq = engine.get_hud_jpeg_with_seq()
        crop_jpg, cseq = engine.get_crop_jpeg_with_seq()
        self.assertTrue(hud_jpg and hud_jpg.startswith(b"\xff\xd8"))
        self.assertTrue(crop_jpg and crop_jpg.startswith(b"\xff\xd8"))
        self.assertEqual(seq, cseq)
        self.assertGreaterEqual(seq, 1)

        # wait_for_frame returns immediately if newer frame already exists
        w_jpg, w_seq = engine.wait_for_frame(last_seq=0, timeout=0.5, kind="hud")
        self.assertEqual(w_seq, seq)
        self.assertEqual(w_jpg, hud_jpg)

        # wait_for_frame wakes promptly when another thread publishes a new frame
        engine._running = True
        woke_result: dict[str, object] = {}

        def waiter():
            jpg, new_seq = engine.wait_for_frame(last_seq=seq, timeout=2.0, kind="crop")
            woke_result["jpg"] = jpg
            woke_result["seq"] = new_seq

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        engine._process_frame(frame, time.monotonic())
        t.join(timeout=2.0)
        engine._running = False
        self.assertFalse(t.is_alive())
        self.assertEqual(woke_result["seq"], seq + 1)
        self.assertIsNotNone(woke_result["jpg"])


if __name__ == "__main__":
    unittest.main()
