"""T01-T25 deterministic FSM scenarios; no camera, model, or wall-clock sleep."""
import unittest

from igbt_pokayoke.logic.fsm import FSMConfig, FSMEvent, InspectionFSM


class FSMScenario(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.fsm = InspectionFSM(
            FSMConfig(
                initial_complete_ms=100,
                stable_state_ms=100,
                min_stable_frames=2,
                c_absent_duration_ms=1200,
                max_frame_gap_ms=200,
                hand_leave_stable_ms=0,
                cycle_timeout_s=30,
                action_timeout_s=20,
            ),
            clock=lambda: self.now,
        )
        self.frame_id = 0

    def send(self, event):
        return self.fsm.process(event, now_monotonic=self.now)

    def obs(self, dt=0.1, **updates):
        self.now += dt
        self.frame_id += 1
        fields = dict(
            kind="observation", capture_monotonic_s=self.now, frame_id=self.frame_id,
            label_class="complete", view_valid=True, quality_ok=True,
            camera_healthy=True, camera_settings_locked=True, config_match=True,
            tube_present=True, label_window_present=True, label_roi_clear=True,
            hand_occluded=False, hand_contact_candidate=False,
            hand_in_contact_roi=False, hand_in_left_grip_zone=False,
            hand_in_right_grip_zone=False, pipe_color="yellow",
        )
        fields.update(updates)
        return self.send(FSMEvent(**fields))

    def prepare(self, pipe_color="yellow"):
        self.send(FSMEvent(kind="new_cycle", cycle_id="cycle-test"))
        self.obs(label_class="complete", pipe_color=pipe_color)
        self.obs(label_class="complete", pipe_color=pipe_color)
        self.send(FSMEvent(kind="start_inspection"))

    def c_contact_and_leave(self):
        self.obs(hand_contact_candidate=True, hand_in_contact_roi=True, label_roi_clear=False,
                 hand_occluded=True)
        self.obs(hand_contact_candidate=False, hand_in_contact_roi=False, label_roi_clear=True,
                 hand_occluded=False, label_class="absent")

    def c_absent(self, duration=1.2, dt=0.1, **kwargs):
        steps = round(duration / dt)
        for _ in range(steps + 1):
            self.obs(dt=dt, label_class="absent", **kwargs)

    def test_T01_initial_complete_then_stable_crossed_out_passes(self):
        self.prepare()
        self.obs(label_class="crossed_out")
        self.obs(label_class="crossed_out")
        self.assertEqual(self.fsm.outcome, "PASS")
        self.assertEqual(self.fsm.result_reason, "A_CROSSED_OUT")

    def test_T02_initial_complete_then_stable_covered_passes(self):
        self.prepare()
        self.obs(label_class="covered")
        self.obs(label_class="covered")
        self.assertEqual(self.fsm.outcome, "PASS")
        self.assertEqual(self.fsm.result_reason, "B_COVERED")

    def test_T03_contact_leave_valid_absent_1200ms_passes(self):
        self.prepare()
        self.c_contact_and_leave()
        self.c_absent()
        self.assertEqual(self.fsm.outcome, "PASS")
        self.assertEqual(self.fsm.result_reason, "C_REMOVED")

    def test_T04_normal_two_end_grips_never_trigger_c(self):
        self.prepare()
        for _ in range(8):
            self.obs(hand_in_left_grip_zone=True, hand_in_right_grip_zone=True)
        self.assertNotIn(self.fsm.state.value, {"PROCESSING", "VERIFY_C_ABSENT"})
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T05_hand_occlusion_never_accumulates_absent_time(self):
        self.prepare()
        self.obs(hand_contact_candidate=True, hand_in_contact_roi=True, hand_occluded=True,
                 label_roi_clear=False, view_valid=False)
        for _ in range(14):
            self.obs(label_class="absent", hand_in_contact_roi=True, hand_occluded=True,
                     label_roi_clear=False, view_valid=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T06_absent_without_contact_candidate_does_not_pass(self):
        self.prepare()
        self.c_absent(duration=1.5)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T07_label_reappearing_cancels_c_timer(self):
        self.prepare(); self.c_contact_and_leave()
        for _ in range(8): self.obs(label_class="absent")
        self.obs(label_class="complete")
        self.c_absent(duration=1.3)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T08_backside_or_flip_view_invalid_never_passes(self):
        self.prepare(); self.c_contact_and_leave()
        for _ in range(16):
            self.obs(label_class="absent", view_valid=False, label_window_present=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T09_part_out_of_view_never_passes(self):
        self.prepare(); self.c_contact_and_leave()
        for _ in range(16):
            self.obs(label_class="absent", tube_present=False, label_window_present=False, view_valid=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T10_hand_detection_with_visible_complete_label_does_not_c_pass(self):
        self.prepare()
        self.obs(hand_contact_candidate=True, hand_in_contact_roi=True)
        self.obs(hand_contact_candidate=False, hand_in_contact_roi=False, label_roi_clear=True)
        for _ in range(15): self.obs(label_class="complete", label_roi_clear=True)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T11_overexposure_reflected_as_quality_failure_blocks_pass(self):
        self.prepare()
        self.obs(label_class="crossed_out", quality_ok=False, view_valid=False)
        self.obs(label_class="crossed_out", quality_ok=False, view_valid=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T12_blurry_frame_blocks_action_confirmation(self):
        self.prepare()
        self.obs(label_class="crossed_out", quality_ok=False)
        self.obs(label_class="crossed_out", quality_ok=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T13_camera_failure_enters_fault(self):
        self.prepare()
        self.send(FSMEvent(kind="fault", reason="CAMERA_DISCONNECTED"))
        self.assertEqual(self.fsm.state.value, "FAULT")
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T14_capture_gap_over_limit_clears_c_timer(self):
        self.prepare(); self.c_contact_and_leave()
        self.obs(label_class="absent")
        self.obs(dt=0.3, label_class="absent")
        for _ in range(5): self.obs(label_class="absent")
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T15_initial_absent_alarms(self):
        self.send(FSMEvent(kind="new_cycle", cycle_id="absent-initial"))
        self.obs(label_class="absent")
        self.assertEqual(self.fsm.outcome, "ALARM")

    def test_T16_yellow_and_blue_pipe_colors_are_not_used_as_label_shortcuts(self):
        for color in ("yellow", "blue"):
            with self.subTest(color=color):
                self.setUp(); self.prepare(pipe_color=color)
                self.assertEqual(self.fsm.state.value, "INSPECTING")
                self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T17_changed_glove_or_unverified_detection_does_not_create_candidate(self):
        self.prepare()
        self.obs(hand_detection_valid=False, hand_contact_candidate=False, hand_in_label_contact_roi=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")
        self.assertNotIn(self.fsm.state.value, {"PROCESSING", "VERIFY_C_ABSENT"})

    def test_T18_unknown_residue_state_never_passes(self):
        self.prepare(); self.c_contact_and_leave()
        self.obs(label_class="unknown")
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T19_restart_with_active_cycle_requires_review(self):
        restarted = InspectionFSM(self.fsm.config, clock=lambda: self.now,
                                  restored_cycle={"cycle_id": "interrupted", "state": "INSPECTING"})
        self.assertEqual(restarted.state.value, "REVIEW_REQUIRED")
        self.assertEqual(restarted.outcome, "ALARM")

    def test_T20_roi_or_camera_configuration_mismatch_blocks_pass(self):
        self.prepare()
        self.obs(label_class="crossed_out", config_match=False)
        self.obs(label_class="crossed_out", config_match=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T21_ambiguous_covered_label_unknown_does_not_pass(self):
        self.prepare()
        self.obs(label_class="unknown")
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T22_end_grip_motion_never_becomes_label_contact(self):
        self.prepare()
        for _ in range(5):
            self.obs(hand_in_left_grip_zone=True, hand_in_right_grip_zone=False,
                     hand_contact_candidate=False, hand_in_contact_roi=False)
        self.assertNotIn(self.fsm.state.value, {"PROCESSING", "VERIFY_C_ABSENT"})

    def test_T23_hand_inspection_without_removal_cancels_candidate(self):
        self.prepare(); self.c_contact_and_leave()
        for _ in range(15): self.obs(label_class="complete")
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T24_marker_without_valid_crossout_does_not_pass(self):
        self.prepare()
        self.obs(marker_detected=True, label_class="complete")
        self.obs(marker_detected=True, label_class="complete")
        self.assertNotEqual(self.fsm.outcome, "PASS")

    def test_T25_camera_settings_unlocked_faults_or_blocks_pass(self):
        self.prepare()
        self.obs(camera_settings_locked=False)
        self.assertNotEqual(self.fsm.outcome, "PASS")
        self.assertIn(self.fsm.state.value, {"FAULT", "ALARM", "VIEW_INVALID"})

    def test_transition_log_records_every_event_and_state_change(self):
        self.prepare()
        self.obs(label_class="crossed_out")
        self.obs(label_class="crossed_out")
        log = self.fsm.transition_log()
        self.assertGreater(len(log), 0)
        self.assertTrue(all("event" in item and "state_after" in item and "capture_monotonic_s" in item for item in log))
        self.assertTrue(any(item["state_after"] == "PASS" for item in log))


if __name__ == "__main__":
    unittest.main()
