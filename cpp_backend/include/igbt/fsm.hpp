#pragma once

#include <igbt/common.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace igbt {

enum class FSMState {
    BOOT,
    WAIT_PART,
    VERIFY_INITIAL,
    READY,
    INSPECTING,
    PROCESSING,
    VERIFY_C_ABSENT,
    VIEW_INVALID,
    PASS,
    ALARM,
    FAULT,
    REVIEW_REQUIRED
};

std::string fsm_state_to_string(FSMState state);
FSMState fsm_state_from_string(const std::string& name);

struct FSMConfig {
    int initial_complete_ms{800};
    int stable_state_ms{500};
    int min_stable_frames{5};
    int c_absent_duration_ms{1200};
    int max_frame_gap_ms{200};
    int max_capture_age_ms{500};
    int hand_leave_stable_ms{250};
    double cycle_timeout_s{60.0};
    double action_timeout_s{40.0};
    std::size_t max_log_records{1000};

    void validate() const;
};

struct FSMEvent {
    std::string kind;
    std::optional<std::string> cycle_id;
    std::optional<double> capture_monotonic_s;
    std::optional<int64_t> frame_id;
    std::string label_class{"unknown"};
    std::optional<bool> view_valid;
    std::optional<bool> quality_ok;
    std::optional<bool> camera_healthy;
    std::optional<bool> camera_settings_locked;
    std::optional<bool> config_match;
    std::optional<bool> tube_present;
    std::optional<bool> label_window_present;
    std::optional<bool> label_roi_clear;
    std::optional<bool> hand_occluded;
    bool hand_contact_candidate{false};
    std::optional<bool> hand_detection_valid;
    std::optional<bool> hand_in_contact_roi;
    std::optional<bool> hand_in_label_contact_roi;
    bool hand_in_left_grip_zone{false};
    bool hand_in_right_grip_zone{false};
    bool marker_detected{false};
    std::optional<std::string> pipe_color;
    std::optional<std::string> reason;
    bool reset_authorized{false};
    std::optional<std::string> operator_anon_id;

    json to_json() const;
};

class InspectionFSM {
public:
    using ClockFn = std::function<double()>;
    using AuditSinkFn = std::function<void(const json&)>;

    explicit InspectionFSM(
        const FSMConfig& config = FSMConfig{},
        ClockFn clock = monotonic_seconds,
        std::optional<json> restored_cycle = std::nullopt,
        AuditSinkFn audit_sink = nullptr
    );

    FSMState process(const FSMEvent& event, std::optional<double> now_monotonic = std::nullopt);

    void invalidate_geometry_evidence(
        const std::string& reason = "CAMERA_GEOMETRY_CHANGED",
        std::optional<double> now_monotonic = std::nullopt
    );

    std::size_t flush_audit_sink();
    std::size_t pending_audit_count() const { return pending_audit_records_.size(); }

    FSMState state() const { return state_; }
    std::string state_str() const { return fsm_state_to_string(state_); }
    const std::string& outcome() const { return outcome_; }
    const std::optional<std::string>& result_reason() const { return result_reason_; }
    const std::optional<std::string>& cycle_id() const { return cycle_id_; }
    bool initial_complete_confirmed() const { return initial_complete_confirmed_; }
    bool contact_seen() const { return contact_seen_; }
    int c_timer_ms() const;
    const FSMConfig& config() const { return config_; }

    const std::deque<json>& transition_records() const { return transition_records_; }
    json transition_log_json() const;

    static bool is_terminal(FSMState s);

private:
    double c_absent_elapsed_ms() const;
    std::string handle_new_cycle(const FSMEvent& event, double now);
    std::string handle_start_inspection(const FSMEvent& event, double now);
    std::string handle_reset(const FSMEvent& event, double now);
    std::string handle_observation(const FSMEvent& event, double now);

    FSMState phase_for_resume() const;
    bool contact_eligible(const FSMEvent& event) const;
    std::pair<bool, std::string> observation_valid(const FSMEvent& event) const;
    std::optional<std::string> capture_timing_reason(const FSMEvent& event, double now);
    std::string invalidate(const FSMEvent& event, double now, const std::string& reason, bool preserve_contact);
    bool observe_stable(const std::string& label, double capture);
    double stable_elapsed_ms(double capture) const;
    void clear_stability();
    void clear_c_timer();
    void clear_cycle_data();
    void transition_to(
        FSMState next_state,
        const std::string& reason,
        const std::string& event_name,
        double now,
        std::optional<double> capture,
        std::optional<int64_t> frame_id = std::nullopt
    );
    void append_log(
        const json& event_obj,
        double now,
        std::optional<double> capture,
        FSMState before,
        FSMState after,
        const std::string& reason,
        const std::string& record_type = "STATE_TRANSITION",
        std::optional<int64_t> frame_id = std::nullopt
    );

    FSMConfig config_;
    ClockFn clock_;
    AuditSinkFn audit_sink_;

    FSMState state_{FSMState::WAIT_PART};
    std::string outcome_{"WAITING"};
    std::optional<std::string> result_reason_;
    std::optional<std::string> cycle_id_;
    bool initial_complete_confirmed_{false};

    std::deque<json> transition_records_;
    std::deque<json> pending_audit_records_;
    uint64_t log_sequence_{0};

    std::optional<double> cycle_started_;
    std::optional<double> inspection_started_;
    std::optional<double> last_capture_;
    std::optional<int64_t> last_frame_id_;

    std::optional<std::string> stable_class_;
    std::optional<double> stable_start_;
    int stable_frames_{0};

    bool contact_seen_{false};
    std::optional<double> hand_left_start_;
    bool hand_left_confirmed_{false};

    std::optional<double> c_absent_start_;
    std::optional<double> c_absent_last_;
    FSMState resume_state_{FSMState::VERIFY_INITIAL};
};

// Operator & engineer Chinese diagnostic text lookup compatible with React UI
struct ReasonExplanationZh {
    std::string operator_message;
    std::string engineer_message;
};

ReasonExplanationZh explain_reason_zh(
    const std::optional<std::string>& reason_code,
    const std::string& outcome = "WAITING"
);

}  // namespace igbt
