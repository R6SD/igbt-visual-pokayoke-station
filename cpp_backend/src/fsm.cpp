#include <igbt/fsm.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace igbt {

std::string fsm_state_to_string(FSMState state) {
    switch (state) {
        case FSMState::BOOT: return "BOOT";
        case FSMState::WAIT_PART: return "WAIT_PART";
        case FSMState::VERIFY_INITIAL: return "VERIFY_INITIAL";
        case FSMState::READY: return "READY";
        case FSMState::INSPECTING: return "INSPECTING";
        case FSMState::PROCESSING: return "PROCESSING";
        case FSMState::VERIFY_C_ABSENT: return "VERIFY_C_ABSENT";
        case FSMState::VIEW_INVALID: return "VIEW_INVALID";
        case FSMState::PASS: return "PASS";
        case FSMState::ALARM: return "ALARM";
        case FSMState::FAULT: return "FAULT";
        case FSMState::REVIEW_REQUIRED: return "REVIEW_REQUIRED";
    }
    return "WAIT_PART";
}

FSMState fsm_state_from_string(const std::string& name) {
    if (name == "BOOT") return FSMState::BOOT;
    if (name == "WAIT_PART") return FSMState::WAIT_PART;
    if (name == "VERIFY_INITIAL") return FSMState::VERIFY_INITIAL;
    if (name == "READY") return FSMState::READY;
    if (name == "INSPECTING") return FSMState::INSPECTING;
    if (name == "PROCESSING") return FSMState::PROCESSING;
    if (name == "VERIFY_C_ABSENT") return FSMState::VERIFY_C_ABSENT;
    if (name == "VIEW_INVALID") return FSMState::VIEW_INVALID;
    if (name == "PASS") return FSMState::PASS;
    if (name == "ALARM") return FSMState::ALARM;
    if (name == "FAULT") return FSMState::FAULT;
    if (name == "REVIEW_REQUIRED") return FSMState::REVIEW_REQUIRED;
    throw std::invalid_argument("Unknown FSMState: " + name);
}

void FSMConfig::validate() const {
    if (c_absent_duration_ms != 1200) {
        throw std::invalid_argument("C removal confirmation is fixed at 1200 ms by the process requirement.");
    }
    if (initial_complete_ms < 0 || stable_state_ms < 0 || min_stable_frames < 1) {
        throw std::invalid_argument("Initial/stable frame requirements are invalid.");
    }
    if (max_frame_gap_ms <= 0 || max_capture_age_ms <= 0) {
        throw std::invalid_argument("Capture timing limits must be positive.");
    }
    if (std::min(cycle_timeout_s, action_timeout_s) <= 0.0) {
        throw std::invalid_argument("Cycle and action timeouts must be positive.");
    }
    if (max_log_records < 1) {
        throw std::invalid_argument("Log retention limit must be positive.");
    }
    if (ood_alarm_frames < 1) {
        throw std::invalid_argument("OOD alarm frame threshold must be positive.");
    }
}

namespace {

template <typename T>
json opt_to_json(const std::optional<T>& opt) {
    if (opt.has_value()) {
        return json(*opt);
    }
    return nullptr;
}

}  // namespace

json FSMEvent::to_json() const {
    return json{
        {"kind", kind},
        {"cycle_id", opt_to_json(cycle_id)},
        {"capture_monotonic_s", opt_to_json(capture_monotonic_s)},
        {"frame_id", opt_to_json(frame_id)},
        {"label_class", label_class},
        {"view_valid", opt_to_json(view_valid)},
        {"quality_ok", opt_to_json(quality_ok)},
        {"camera_healthy", opt_to_json(camera_healthy)},
        {"camera_settings_locked", opt_to_json(camera_settings_locked)},
        {"config_match", opt_to_json(config_match)},
        {"tube_present", opt_to_json(tube_present)},
        {"label_window_present", opt_to_json(label_window_present)},
        {"label_roi_clear", opt_to_json(label_roi_clear)},
        {"hand_occluded", opt_to_json(hand_occluded)},
        {"hand_contact_candidate", hand_contact_candidate},
        {"hand_detection_valid", opt_to_json(hand_detection_valid)},
        {"hand_in_contact_roi", opt_to_json(hand_in_contact_roi)},
        {"hand_in_label_contact_roi", opt_to_json(hand_in_label_contact_roi)},
        {"hand_in_left_grip_zone", hand_in_left_grip_zone},
        {"hand_in_right_grip_zone", hand_in_right_grip_zone},
        {"marker_detected", marker_detected},
        {"ood_rejected", ood_rejected},
        {"ood_energy", opt_to_json(ood_energy)},
        {"ood_mahalanobis", opt_to_json(ood_mahalanobis)},
        {"ood_reason", opt_to_json(ood_reason)},
        {"pipe_color", opt_to_json(pipe_color)},
        {"reason", opt_to_json(reason)},
        {"reset_authorized", reset_authorized},
        {"operator_anon_id", opt_to_json(operator_anon_id)},
    };
}

bool InspectionFSM::is_terminal(FSMState s) {
    return s == FSMState::PASS || s == FSMState::ALARM ||
           s == FSMState::FAULT || s == FSMState::REVIEW_REQUIRED;
}

InspectionFSM::InspectionFSM(
    const FSMConfig& config,
    ClockFn clock,
    std::optional<json> restored_cycle,
    AuditSinkFn audit_sink
)
    : config_(config),
      clock_(clock ? std::move(clock) : ClockFn(monotonic_seconds)),
      audit_sink_(std::move(audit_sink)) {
    config_.validate();

    if (restored_cycle.has_value() && restored_cycle->is_object()) {
        std::string prev_st = restored_cycle->value("state", std::string(""));
        if (!prev_st.empty() &&
            prev_st != "WAIT_PART" &&
            prev_st != "PASS" &&
            prev_st != "ALARM" &&
            prev_st != "FAULT") {
            if (restored_cycle->contains("cycle_id") && (*restored_cycle)["cycle_id"].is_string()) {
                cycle_id_ = (*restored_cycle)["cycle_id"].get<std::string>();
            }
            result_reason_ = "RESTART_DURING_ACTIVE_CYCLE";
            state_ = FSMState::REVIEW_REQUIRED;
            outcome_ = "ALARM";
            append_log(
                json{{"kind", "startup_restore"}},
                clock_(),
                std::nullopt,
                FSMState::WAIT_PART,
                FSMState::REVIEW_REQUIRED,
                *result_reason_
            );
        }
    }
}

double InspectionFSM::c_absent_elapsed_ms() const {
    if (!c_absent_start_.has_value() || !c_absent_last_.has_value()) {
        return 0.0;
    }
    return std::max(0.0, (*c_absent_last_ - *c_absent_start_) * 1000.0);
}

int InspectionFSM::c_timer_ms() const {
    const double elapsed = c_absent_elapsed_ms();
    if (elapsed <= 0.0) {
        return 0;
    }
    const int limit = config_.c_absent_duration_ms;
    if (state_ == FSMState::PASS && result_reason_.has_value() && *result_reason_ == "C_REMOVED") {
        return limit;
    }
    if (elapsed + 1e-6 >= static_cast<double>(limit)) {
        return limit;
    }
    return std::max(0, std::min(limit - 1, static_cast<int>(std::llround(elapsed))));
}

std::size_t InspectionFSM::flush_audit_sink() {
    if (!audit_sink_) {
        pending_audit_records_.clear();
        return 0;
    }
    std::size_t delivered = 0;
    while (!pending_audit_records_.empty()) {
        const json& record = pending_audit_records_.front();
        try {
            audit_sink_(record);
        } catch (...) {
            break;
        }
        pending_audit_records_.pop_front();
        ++delivered;
    }
    return delivered;
}

void InspectionFSM::invalidate_geometry_evidence(
    const std::string& reason,
    std::optional<double> now_monotonic
) {
    if (is_terminal(state_) || state_ == FSMState::WAIT_PART) {
        return;
    }
    const double now = now_monotonic.has_value() ? *now_monotonic : clock_();
    clear_stability();
    clear_c_timer();
    contact_seen_ = false;
    hand_left_start_.reset();
    hand_left_confirmed_ = false;
    resume_state_ = phase_for_resume();
    transition_to(FSMState::VIEW_INVALID, reason, "geometry_reset", now, std::nullopt, std::nullopt);
}

json InspectionFSM::transition_log_json() const {
    json arr = json::array();
    for (const auto& item : transition_records_) {
        arr.push_back(item);
    }
    return arr;
}

FSMState InspectionFSM::process(const FSMEvent& event, std::optional<double> now_monotonic) {
    const double now = now_monotonic.has_value() ? *now_monotonic : clock_();
    const FSMState before = state_;
    std::string reason = "NO_STATE_CHANGE";

    if (event.kind == "new_cycle") {
        reason = handle_new_cycle(event, now);
    } else if (event.kind == "start_inspection") {
        reason = handle_start_inspection(event, now);
    } else if (event.kind == "observation") {
        reason = handle_observation(event, now);
    } else if (event.kind == "fault") {
        reason = event.reason.value_or("FAULT_REPORTED");
        transition_to(FSMState::FAULT, reason, event.kind, now, event.capture_monotonic_s, event.frame_id);
    } else if (event.kind == "reset" || event.kind == "part_removed") {
        reason = handle_reset(event, now);
    } else {
        reason = "UNKNOWN_EVENT_TYPE";
    }

    const FSMState after = state_;
    append_log(
        event.to_json(),
        now,
        event.capture_monotonic_s,
        before,
        after,
        reason,
        "EVENT",
        event.frame_id
    );
    return state_;
}

std::string InspectionFSM::handle_new_cycle(const FSMEvent& event, double now) {
    if (state_ != FSMState::WAIT_PART) {
        return "CYCLE_ALREADY_ACTIVE";
    }
    clear_cycle_data();
    if (event.cycle_id.has_value() && !event.cycle_id->empty()) {
        cycle_id_ = *event.cycle_id;
    } else {
        cycle_id_ = "cycle-" + std::to_string(static_cast<int64_t>(now * 1000.0));
    }
    cycle_started_ = now;
    result_reason_.reset();
    transition_to(FSMState::VERIFY_INITIAL, "NEW_CYCLE", event.kind, now, event.capture_monotonic_s, event.frame_id);
    return "NEW_CYCLE";
}

std::string InspectionFSM::handle_start_inspection(const FSMEvent& event, double now) {
    if (state_ != FSMState::READY || !initial_complete_confirmed_) {
        return "INITIAL_COMPLETE_NOT_CONFIRMED";
    }
    inspection_started_ = now;
    transition_to(FSMState::INSPECTING, "INSPECTION_STARTED", event.kind, now, event.capture_monotonic_s, event.frame_id);
    return "INSPECTION_STARTED";
}

std::string InspectionFSM::handle_reset(const FSMEvent& event, double now) {
    if (event.kind == "part_removed" && state_ == FSMState::PASS) {
        clear_cycle_data();
        cycle_id_.reset();
        transition_to(FSMState::WAIT_PART, "PART_REMOVED", event.kind, now, event.capture_monotonic_s, event.frame_id);
        return "PART_REMOVED";
    }
    if (!event.reset_authorized) {
        return "RESET_NOT_AUTHORIZED";
    }
    if (state_ != FSMState::ALARM &&
        state_ != FSMState::FAULT &&
        state_ != FSMState::REVIEW_REQUIRED &&
        state_ != FSMState::PASS) {
        return "RESET_NOT_ALLOWED_IN_ACTIVE_CYCLE";
    }
    clear_cycle_data();
    cycle_id_.reset();
    const std::string r = event.reason.value_or("AUTHORIZED_RESET");
    transition_to(FSMState::WAIT_PART, r, event.kind, now, event.capture_monotonic_s, event.frame_id);
    return r;
}

FSMState InspectionFSM::phase_for_resume() const {
    if (!initial_complete_confirmed_) {
        return FSMState::VERIFY_INITIAL;
    }
    if (!inspection_started_.has_value()) {
        return FSMState::READY;
    }
    if (contact_seen_) {
        return FSMState::PROCESSING;
    }
    return FSMState::INSPECTING;
}

std::string InspectionFSM::handle_observation(const FSMEvent& event, double now) {
    if (is_terminal(state_) || state_ == FSMState::WAIT_PART) {
        return "OBSERVATION_IGNORED_OUTSIDE_ACTIVE_CYCLE";
    }
    if (cycle_started_.has_value() && (now - *cycle_started_) >= config_.cycle_timeout_s) {
        transition_to(FSMState::ALARM, "CYCLE_TIMEOUT", event.kind, now, event.capture_monotonic_s, event.frame_id);
        return "CYCLE_TIMEOUT";
    }
    if (inspection_started_.has_value() && (now - *inspection_started_) >= config_.action_timeout_s) {
        transition_to(FSMState::ALARM, "ACTION_TIMEOUT", event.kind, now, event.capture_monotonic_s, event.frame_id);
        return "ACTION_TIMEOUT";
    }

    if (event.camera_healthy.has_value() && !(*event.camera_healthy)) {
        transition_to(FSMState::FAULT, "FAULT_CAMERA", event.kind, now, event.capture_monotonic_s, event.frame_id);
        return "FAULT_CAMERA";
    }
    if (event.camera_settings_locked.has_value() && !(*event.camera_settings_locked)) {
        transition_to(FSMState::FAULT, "CAMERA_SETTINGS_NOT_LOCKED", event.kind, now, event.capture_monotonic_s, event.frame_id);
        return "CAMERA_SETTINGS_NOT_LOCKED";
    }

    const auto frame_reason = capture_timing_reason(event, now);
    if (frame_reason.has_value()) {
        clear_stability();
        clear_c_timer();
        hand_left_start_.reset();
        hand_left_confirmed_ = false;
        resume_state_ = phase_for_resume();
        transition_to(FSMState::VIEW_INVALID, *frame_reason, event.kind, now, event.capture_monotonic_s, event.frame_id);
        return *frame_reason;
    }

    if (event.config_match.has_value() && !(*event.config_match)) {
        return invalidate(event, now, "CONFIGURATION_MISMATCH", false);
    }

    const bool inspection_active = initial_complete_confirmed_ && inspection_started_.has_value();
    const bool contact = contact_eligible(event);
    if (contact && inspection_active) {
        contact_seen_ = true;
        hand_left_start_.reset();
        hand_left_confirmed_ = false;
        if (state_ != FSMState::PROCESSING) {
            transition_to(FSMState::PROCESSING, "HAND_CONTACT_CANDIDATE", event.kind, now, event.capture_monotonic_s, event.frame_id);
        }
    } else if (event.hand_in_contact_roi.value_or(false) && inspection_active) {
        hand_left_start_.reset();
        hand_left_confirmed_ = false;
    }

    const auto [valid, invalid_reason] = observation_valid(event);
    if (!valid) {
        const bool preserve = contact_seen_ && event.hand_occluded.value_or(false);
        return invalidate(event, now, invalid_reason, preserve);
    }

    if (state_ == FSMState::VIEW_INVALID) {
        const FSMState target = phase_for_resume();
        transition_to(target, "VIEW_RESTORED", event.kind, now, event.capture_monotonic_s, event.frame_id);
    }

    if (event.ood_rejected || event.label_class == "ood_unknown") {
        clear_stability();
        clear_c_timer();
        ++ood_streak_;
        if (ood_streak_ >= config_.ood_alarm_frames) {
            transition_to(FSMState::ALARM, "STATE_OOD_UNKNOWN", event.kind, now, event.capture_monotonic_s, event.frame_id);
            return "STATE_OOD_UNKNOWN";
        }
        return "OOD_GATE_HOLD";
    }
    ood_streak_ = 0;

    if (event.label_class == "unknown") {
        clear_stability();
        clear_c_timer();
        transition_to(FSMState::ALARM, "ALARM_UNKNOWN_LABEL", event.kind, now, event.capture_monotonic_s, event.frame_id);
        return "ALARM_UNKNOWN_LABEL";
    }

    const double cap_s = event.capture_monotonic_s.value();

    if (state_ == FSMState::VERIFY_INITIAL) {
        if (event.label_class != "complete") {
            const std::string code = (event.label_class == "absent")
                                         ? "INITIAL_LABEL_ABSENT"
                                         : "INITIAL_LABEL_NOT_COMPLETE";
            transition_to(FSMState::ALARM, code, event.kind, now, event.capture_monotonic_s, event.frame_id);
            return code;
        }
        const bool stable = observe_stable("complete", cap_s);
        const double elapsed_ms = stable_elapsed_ms(cap_s);
        if (stable && elapsed_ms + 1e-6 >= static_cast<double>(config_.initial_complete_ms)) {
            initial_complete_confirmed_ = true;
            transition_to(FSMState::READY, "INITIAL_COMPLETE_CONFIRMED", event.kind, now, event.capture_monotonic_s, event.frame_id);
            return "INITIAL_COMPLETE_CONFIRMED";
        }
        return "VERIFYING_INITIAL_COMPLETE";
    }

    if (state_ != FSMState::READY &&
        state_ != FSMState::INSPECTING &&
        state_ != FSMState::PROCESSING &&
        state_ != FSMState::VERIFY_C_ABSENT) {
        return "OBSERVATION_NOT_APPLICABLE";
    }

    if (state_ == FSMState::READY) {
        return "WAITING_FOR_INSPECTION_START";
    }

    if (event.label_class == "crossed_out" || event.label_class == "covered") {
        clear_c_timer();
        if (observe_stable(event.label_class, cap_s) &&
            stable_elapsed_ms(cap_s) + 1e-6 >= static_cast<double>(config_.stable_state_ms)) {
            const std::string code = (event.label_class == "crossed_out") ? "A_CROSSED_OUT" : "B_COVERED";
            transition_to(FSMState::PASS, code, event.kind, now, event.capture_monotonic_s, event.frame_id);
            return code;
        }
        return "VERIFYING_A_OR_B";
    }

    if (event.label_class == "complete" || event.label_class == "absent") {
        clear_stability();
    }

    if (event.label_class == "complete") {
        clear_c_timer();
        if ((state_ == FSMState::VERIFY_C_ABSENT || state_ == FSMState::PROCESSING) &&
            !event.hand_in_contact_roi.value_or(false)) {
            contact_seen_ = false;
            hand_left_confirmed_ = false;
            hand_left_start_.reset();
            transition_to(FSMState::INSPECTING, "LABEL_REAPPEARED_C_CANCELED", event.kind, now, event.capture_monotonic_s, event.frame_id);
        }
        return "LABEL_COMPLETE_NO_C_EVIDENCE";
    }

    if (state_ == FSMState::PROCESSING || state_ == FSMState::VERIFY_C_ABSENT) {
        if (event.hand_in_contact_roi.value_or(false) || event.hand_contact_candidate) {
            hand_left_start_.reset();
            hand_left_confirmed_ = false;
            clear_c_timer();
            if (state_ == FSMState::VERIFY_C_ABSENT) {
                transition_to(FSMState::PROCESSING, "HAND_REENTERED_CONTACT_ROI", event.kind, now, event.capture_monotonic_s, event.frame_id);
            }
            return "WAITING_FOR_HAND_TO_LEAVE";
        }
        if (!contact_seen_) {
            clear_c_timer();
            return "C_REQUIRES_CONTACT_CANDIDATE";
        }
        if (!hand_left_start_.has_value()) {
            hand_left_start_ = cap_s;
        }
        const double left_ms = (cap_s - *hand_left_start_) * 1000.0;
        if (left_ms + 1e-6 >= static_cast<double>(config_.hand_leave_stable_ms)) {
            hand_left_confirmed_ = true;
        }
        if (!hand_left_confirmed_ || !event.label_roi_clear.value_or(false)) {
            clear_c_timer();
            return "WAITING_FOR_CLEAR_ROI_AFTER_HAND_LEAVE";
        }
        if (event.label_class == "absent") {
            if (state_ != FSMState::VERIFY_C_ABSENT) {
                clear_c_timer();
                c_absent_start_ = cap_s;
                c_absent_last_ = cap_s;
                transition_to(FSMState::VERIFY_C_ABSENT, "C_ABSENT_TIMER_STARTED", event.kind, now, event.capture_monotonic_s, event.frame_id);
            } else {
                if (!c_absent_last_.has_value() ||
                    (cap_s - *c_absent_last_) * 1000.0 > static_cast<double>(config_.max_frame_gap_ms)) {
                    clear_c_timer();
                    c_absent_start_ = cap_s;
                    c_absent_last_ = cap_s;
                    return "C_ABSENT_TIMER_RESTARTED_AFTER_GAP";
                }
                c_absent_last_ = cap_s;
                if (c_absent_elapsed_ms() + 1e-6 >= static_cast<double>(config_.c_absent_duration_ms)) {
                    transition_to(FSMState::PASS, "C_REMOVED", event.kind, now, event.capture_monotonic_s, event.frame_id);
                    return "C_REMOVED";
                }
            }
            return "VERIFYING_C_ABSENT";
        }
        clear_c_timer();
    }

    return "OBSERVATION_RECORDED";
}

bool InspectionFSM::contact_eligible(const FSMEvent& event) const {
    const bool hand_in = event.hand_in_contact_roi.value_or(false) ||
                         event.hand_in_label_contact_roi.value_or(false);
    const bool grip = event.hand_in_left_grip_zone || event.hand_in_right_grip_zone;
    const bool det_ok = !event.hand_detection_valid.has_value() || *event.hand_detection_valid;
    return event.hand_contact_candidate && hand_in && !grip && det_ok;
}

std::pair<bool, std::string> InspectionFSM::observation_valid(const FSMEvent& event) const {
    if (!event.view_valid.value_or(false)) {
        if (event.reason.has_value() && !event.reason->empty()) {
            return {false, *event.reason};
        }
        return {false, "VIEW_INVALID"};
    }
    if (!event.quality_ok.value_or(false)) return {false, "IMAGE_QUALITY_INVALID"};
    if (!event.tube_present.value_or(false)) return {false, "TUBE_NOT_PRESENT_OR_UNKNOWN"};
    if (!event.label_window_present.value_or(false)) return {false, "LABEL_WINDOW_INVALID"};
    if (!event.label_roi_clear.value_or(false) ||
        !event.hand_occluded.has_value() || *event.hand_occluded) {
        return {false, "LABEL_ROI_OCCLUDED_OR_UNKNOWN"};
    }
    if (!event.config_match.value_or(false)) return {false, "CONFIGURATION_MISMATCH"};
    if (!event.camera_healthy.value_or(false)) return {false, "CAMERA_HEALTH_UNKNOWN"};
    if (!event.camera_settings_locked.value_or(false)) return {false, "CAMERA_SETTINGS_NOT_VERIFIED"};
    return {true, ""};
}

std::optional<std::string> InspectionFSM::capture_timing_reason(const FSMEvent& event, double now) {
    if (!event.capture_monotonic_s.has_value() || !event.frame_id.has_value()) {
        return "CAPTURE_TIMESTAMP_OR_FRAME_ID_MISSING";
    }
    const double capture = *event.capture_monotonic_s;
    const double age_ms = (now - capture) * 1000.0;
    if (age_ms < -1.0 || age_ms > static_cast<double>(config_.max_capture_age_ms)) {
        return "CAPTURE_TIMESTAMP_STALE_OR_WRONG_CLOCK_DOMAIN";
    }
    if (last_capture_.has_value()) {
        const double delta_ms = (capture - *last_capture_) * 1000.0;
        if (delta_ms <= 0.0) {
            return "CAPTURE_TIMESTAMP_NOT_MONOTONIC";
        }
        if (last_frame_id_.has_value() && *event.frame_id <= *last_frame_id_) {
            return "FRAME_ID_NOT_MONOTONIC";
        }
        if (delta_ms > static_cast<double>(config_.max_frame_gap_ms)) {
            last_capture_ = capture;
            last_frame_id_ = *event.frame_id;
            return "FRAME_GAP_EXCEEDED";
        }
    }
    last_capture_ = capture;
    last_frame_id_ = *event.frame_id;
    return std::nullopt;
}

std::string InspectionFSM::invalidate(
    const FSMEvent& event,
    double now,
    const std::string& reason,
    bool preserve_contact
) {
    clear_stability();
    clear_c_timer();
    hand_left_start_.reset();
    hand_left_confirmed_ = false;
    if (!preserve_contact) {
        contact_seen_ = false;
    }
    resume_state_ = phase_for_resume();
    transition_to(FSMState::VIEW_INVALID, reason, event.kind, now, event.capture_monotonic_s, event.frame_id);
    return reason;
}

bool InspectionFSM::observe_stable(const std::string& label, double capture) {
    if (!stable_class_.has_value() || *stable_class_ != label) {
        stable_class_ = label;
        stable_start_ = capture;
        stable_frames_ = 1;
        return stable_frames_ >= config_.min_stable_frames;
    }
    ++stable_frames_;
    return stable_frames_ >= config_.min_stable_frames;
}

double InspectionFSM::stable_elapsed_ms(double capture) const {
    if (!stable_start_.has_value()) {
        return 0.0;
    }
    return std::max(0.0, (capture - *stable_start_) * 1000.0);
}

void InspectionFSM::clear_stability() {
    stable_class_.reset();
    stable_start_.reset();
    stable_frames_ = 0;
}

void InspectionFSM::clear_c_timer() {
    c_absent_start_.reset();
    c_absent_last_.reset();
}

void InspectionFSM::clear_cycle_data() {
    initial_complete_confirmed_ = false;
    cycle_started_.reset();
    inspection_started_.reset();
    last_capture_.reset();
    last_frame_id_.reset();
    contact_seen_ = false;
    hand_left_start_.reset();
    hand_left_confirmed_ = false;
    resume_state_ = FSMState::VERIFY_INITIAL;
    ood_streak_ = 0;
    clear_stability();
    clear_c_timer();
}

void InspectionFSM::transition_to(
    FSMState next_state,
    const std::string& reason,
    const std::string& event_name,
    double now,
    std::optional<double> capture,
    std::optional<int64_t> frame_id
) {
    const FSMState previous = state_;
    state_ = next_state;
    result_reason_ = reason;

    if (next_state == FSMState::PASS) {
        outcome_ = "PASS";
    } else if (next_state == FSMState::ALARM || next_state == FSMState::REVIEW_REQUIRED) {
        outcome_ = "ALARM";
    } else if (next_state == FSMState::FAULT) {
        outcome_ = "FAULT";
    } else if (next_state == FSMState::VIEW_INVALID) {
        outcome_ = "VIEW_INVALID";
    } else {
        outcome_ = "WAITING";
    }

    if (previous != next_state || (reason != "NO_STATE_CHANGE" && reason != "OBSERVATION_RECORDED")) {
        append_log(
            json{{"kind", event_name}},
            now,
            capture,
            previous,
            next_state,
            reason,
            "STATE_TRANSITION",
            frame_id
        );
    }
}

void InspectionFSM::append_log(
    const json& event_obj,
    double now,
    std::optional<double> capture,
    FSMState before,
    FSMState after,
    const std::string& reason,
    const std::string& record_type,
    std::optional<int64_t> frame_id
) {
    ++log_sequence_;
    json record{
        {"sequence", log_sequence_},
        {"record_type", record_type},
        {"event", event_obj},
        {"now_monotonic_s", now},
        {"capture_monotonic_s", opt_to_json(capture)},
        {"frame_id", opt_to_json(frame_id)},
        {"state_before", fsm_state_to_string(before)},
        {"state_after", fsm_state_to_string(after)},
        {"outcome", outcome_},
        {"reason", reason},
        {"cycle_id", opt_to_json(cycle_id_)},
        {"c_timer_ms", c_timer_ms()},
        {"initial_complete_confirmed", initial_complete_confirmed_},
    };

    if (transition_records_.size() >= config_.max_log_records) {
        transition_records_.pop_front();
    }
    transition_records_.push_back(record);

    if (audit_sink_) {
        pending_audit_records_.push_back(std::move(record));
        flush_audit_sink();
    }
}

ReasonExplanationZh explain_reason_zh(
    const std::optional<std::string>& reason_code,
    const std::string& outcome
) {
    static const std::unordered_map<std::string, ReasonExplanationZh> kTable = {
        {"WAITING_FOR_PART", {"请双手佩戴防静电手套握持料管两端，将标签正对相机检测区", "等待透明料管与防呆标签进入视野"}},
        {"NEW_CYCLE", {"检测到料管入位，正在核验初始完整标签", "新检测周期启动，执行初始完整标签确认"}},
        {"INITIAL_COMPLETE_CONFIRMED", {"初始标签核验通过，请执行划除 / 遮挡 / 移除标签动作", "初始完整标签连续稳定达标，进入作业监测"}},
        {"INSPECTION_STARTED", {"正在监测标签处置动作（A划除 / B遮挡 / C撕除）", "作业监测阶段已激活"}},
        {"HAND_CONTACT_CANDIDATE", {"检测到防静电手套进入标签接触区，请在处置后移开双手", "2D视觉邻近代理触发接触候选，等待手部离开核验"}},
        {"C_ABSENT_TIMER_STARTED", {"检测到标签已移除，请保持平稳 1.2 秒完成确认", "C类脱落确认计时启动（1200 ms 连续窗口）"}},
        {"A_CROSSED_OUT", {"放行通过（A类·标签已划除）", "A类划除状态连续稳定达标，防呆判定通过"}},
        {"B_COVERED", {"放行通过（B类·标签已遮挡）", "B类遮挡状态连续稳定达标，防呆判定通过"}},
        {"C_REMOVED", {"放行通过（C类·标签已移除满 1.2s）", "接触候选后手部离开且标签连续缺失满 1200 ms，判定通过"}},
        {"BARE_HAND_VIOLATION", {"违规报警：检测到裸手触碰料管！请立即佩戴防静电手套并由工程师复位", "车间严格模式下触发裸手接触料管/标签 SOP 闭锁报警"}},
        {"TUBE_SWAP_SUSPECTED", {"防呆闭锁：检测到中途疑似掉包或更换料管！请核查后由工程师复位", "活跃检测周期内检出料管颜色突变或质心非连续跳变，触发防掉包闭锁"}},
        {"BARE_HAND_WARNING_NO_GLOVE", {"提示：检测到未佩戴防静电手套，请穿戴合规手套后再持管作业", "视野内检出裸手皮肤区域，尚未触碰料管"}},
        {"INITIAL_LABEL_ABSENT", {"异常闭锁：入位料管缺失初始标签，禁止作业，请呼叫工程师复核", "VERIFY_INITIAL 阶段检出 absent，触发初始缺标报警"}},
        {"INITIAL_LABEL_NOT_COMPLETE", {"异常闭锁：初始标签状态不完整，请核查来料标签后复位", "VERIFY_INITIAL 阶段标签非 complete，触发来料异常报警"}},
        {"ALARM_UNKNOWN_LABEL", {"异常闭锁：标签状态无法可靠识别，请核查标签与照明后复位", "有效视野下分类置信度不足（unknown），按故障安全原则闭锁"}},
        {"CYCLE_TIMEOUT", {"超时报警：单管作业总时长超限，请复核后重新开始", "检测周期超过 cycle_timeout_s 阈值"}},
        {"ACTION_TIMEOUT", {"超时报警：标签处置动作超时，请确认标签状态后复位", "进入 INSPECTING 后超过 action_timeout_s 未完成有效处置"}},
        {"IMAGE_QUALITY_INVALID", {"画面质量受限：存在反光过曝、暗光或失焦，请调整持管角度", "标签 ROI 光学质量门控未通过（亮度/过曝率/清晰度超限）"}},
        {"LABEL_ROI_OVEREXPOSED", {"强反光提示：透明料管表面反光过强，请微调持管倾角", "标签区域过曝像素占比超过阈值上限"}},
        {"OVEREXPOSED", {"强反光提示：透明料管表面反光过强，请微调持管倾角", "标签区域过曝像素占比超过阈值上限"}},
        {"LOW_SHARPNESS", {"画面轻微模糊：请保持料管平稳处于焦平面内", "拉普拉斯清晰度方差低于设定下限"}},
        {"LABEL_ROI_OCCLUDED_OR_UNKNOWN", {"标签区域被遮挡：请移开中央手部或异物以恢复视野", "标签窗口存在遮挡物重叠，暂停状态判定"}},
        {"CONFIGURATION_MISMATCH", {"工位待标定：严格模式下需先完成 1280×720 ROI 标定保存", "当前相机分辨率/设备ID或 ROI 标定状态与车间严格门控不匹配"}},
        {"ROI_UNCALIBRATED", {"工位待标定：请切换至工程师工作台完成 ROI 区域标定", "ROI_UNCALIBRATED：严格模式要求已标定的 ROI 配置"}},
        {"CAMERA_NOT_CONNECTED", {"相机未连接：请检查 USB 视觉相机连接状态", "UVC 采集设备离线或初始化中"}},
        {"CAMERA_FEED_LOST", {"相机信号中断：正在自动重连 USB 相机", "视频流读取中断，已自动置为 VIEW_INVALID 防止误放行"}},
        {"TUBE_ALIGN_FAIL", {"料管对齐失败：请摆正料管并避开中段强反光（已阻断放行）", "RANSAC 平行壁面拟合连续失败 ≥3 帧或几何门控越界，严禁回退无拉平分类"}},
        {"TUBE_ALIGN_LOW_CONF", {"料管姿态波动：暂借上一帧矩阵，请保持料管平稳以恢复核验", "料管对齐连续失败 ≤2 帧，复用上一有效矩阵并冻结 PASS 达标计时"}},
        {"WHITE_GLOVE_OVEREXPOSED", {"白色手套反光过强：请检查补光角度并确认合规佩戴", "白色 ESD 手套区域 V>=245 过曝占比超过 35%，已触发光学护栏告警"}},
        {"STATE_OOD_UNKNOWN", {"异常闭锁：连续检出未知异常样本(OOD)，已连拍取证，请工程师复核", "连续达到阈值帧数触发 Energy/Mahalanobis OOD 双门控拒识，按故障安全原则闭锁报警"}},
        {"OOD_GATE_HOLD", {"疑似未知异常样本(OOD)，已暂停放行计时并复核中", "单帧触发 OOD 双门控拒识，已清零稳定性与 1.2s 脱落计时"}},
        {"OOD_PROB_LOW", {"标签分类置信度不足，已拒绝放行", "温度缩放后最大类别概率低于阈值 τ_p"}},
        {"OOD_ENERGY_HIGH", {"标签能量分数越界（疑似未见异常或强干扰），已拒识", "LogSumExp Energy Score 高于标定阈值 τ_E"}},
        {"OOD_MAHALANOBIS_HIGH", {"特征空间偏离已知分布（马氏距离超限），已拒识", "倒数第二层特征向量距最近已知类中心的 Mahalanobis 距离高于 τ_M"}},
        {"OOD_DUAL_EXCEEDED", {"Energy 与 Mahalanobis 双门控同时越界，判定未知异常", "Energy Score 与特征马氏距离均超标定阈值"}},
        {"OOD_CALIBRATION_SHA256_MISMATCH", {"OOD 标定参数与当前模型权重哈希不匹配，已拒绝加载", "ood_calibration.json 的 model_sha256 与当前激活模型不一致"}},
        {"OOD_NON_FINITE", {"神经网络推理输出非有限数值，已安全拒识", "检测到 NaN/Inf logits 或特征张量"}},
        {"MES_HEARTBEAT_TIMEOUT", {"MES 断联超时闭锁：请恢复厂区 MES 网络或由 QE 审批受控白名单", "MES 心跳连续超时超过阈值（默认 60s），车规 Fail-Closed 自动停机锁机"}},
        {"MES_OFFLINE_LOT_CHANGE_FORBIDDEN", {"离线状态严禁切换工单或批次（Lot），请联网后重新扫码核验", "断网期间无法在线校验批次防混料与状态，强制拒绝换批操作"}},
        {"MES_LOT_UNBOUND_LOCK", {"尚未在线绑定当前工单/批次（Lot），禁止放行", "已启用 MES 强管控但 mes_active_lot 为空"}},
        {"MES_LOT_HOLD_RECALL", {"MES 批次冻结追回闭锁：请隔离该批次已检料管并由 QE 复核解锁", "MES 对账返回当前 Lot 为 HOLD 状态，立即触发硬闭锁追回"}},
        {"MES_LOT_SCRAP_RECALL", {"MES 批次报废追回闭锁：请立即拦截该批次全部料管并由 QE 解锁", "MES 对账返回当前 Lot 为 SCRAP 状态，立即触发硬闭锁追回"}},
        {"MES_WHITELIST_QUOTA_EXCEEDED", {"受控离线白名单管数配额已耗尽，工位自动锁定", "离线作业管数达到 QE 审批上限（<= 50 管），触发先到先锁"}},
        {"MES_WHITELIST_DURATION_EXCEEDED", {"受控离线白名单时长已超时，工位自动锁定", "离线作业时长达到 QE 审批上限（<= 30 分钟），触发先到先锁"}},
        {"MES_PIPE_COLOR_MISMATCH", {"料管颜色与当前工单配方不符，防混料闭锁！", "视觉检出管色与在线绑定 Lot 预期料管颜色冲突"}},
        {"MES_LOT_SIGNATURE_INVALID", {"离线批次缓存签名校验失败，禁止继续作业", "mes_active_lot 本地 SHA-256 完整性签名校验不通过"}},
        {"MES_LOT_CACHE_EXPIRED", {"离线批次缓存已过期，请恢复 MES 网络重新绑定核验", "当前时间超出活跃 Lot 缓存有效期 expires_at_ms"}},
        {"AUTHORIZED_RESET", {"人工复位完成，请放入下一支待检料管", "已执行授权复位，状态机回到 WAIT_PART"}},
        {"STRICT_MODE_REVERIFY_REQUIRED", {"已切换至车间严格模式，请重新放入料管执行完整核验", "从调试模式切入严格模式，已清除未闭锁调试周期状态"}},
    };

    if (reason_code.has_value() && !reason_code->empty()) {
        auto it = kTable.find(*reason_code);
        if (it != kTable.end()) {
            return it->second;
        }
    }
    if (outcome == "PASS") {
        return {"防呆核验通过，允许进入下道工序", "状态机判定 PASS"};
    }
    if (outcome == "ALARM") {
        return {"防呆报警闭锁，请联系工程师复核复位", reason_code.value_or("ALARM")};
    }
    if (outcome == "VIEW_INVALID") {
        return {"当前视野未就绪，请将料管平稳置于检测框内", reason_code.value_or("VIEW_INVALID")};
    }
    if (outcome == "FAULT") {
        return {"设备状态异常，请检查相机与工位配置", reason_code.value_or("FAULT")};
    }
    return {"等待待检料管入位", reason_code.value_or("WAITING")};
}

}  // namespace igbt
