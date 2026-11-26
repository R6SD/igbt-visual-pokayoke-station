#pragma once

#include "igbt/common.hpp"
#include "igbt/database.hpp"
#include "igbt/industrial_hal.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace igbt {

enum class RedRabbitState {
    PROD_READY,
    CHALLENGE_REQUIRED,
    CHALLENGE_HARD_LOCK
};

std::string red_rabbit_state_to_string(RedRabbitState state);
int64_t default_wall_clock_ms();

struct RedRabbitPolicy {
    bool strict_enabled{true};
    int required_ok_count{1};
    int required_ng_count{2};
    int64_t interval_ms{4LL * 3600 * 1000};
    int shift_hour_day{8};
    int shift_hour_night{20};
    int64_t shift_grace_ms{5LL * 60 * 1000};
    int64_t session_timeout_ms{10LL * 60 * 1000};
};

struct RedRabbitPlanStep {
    int step_index{0};
    std::string sample_uid;
    std::string sample_type;
    std::string expected_verdict;
    std::string product_family;
    bool completed{false};
    std::optional<bool> is_match;
    std::optional<std::string> actual_verdict;
};

class RedRabbitController {
public:
    using WallClockMsFn = std::function<int64_t()>;
    using InterlockCallback = std::function<void(InterlockSignal)>;

    explicit RedRabbitController(
        StationDatabase& db,
        std::string station_id = "STATION_01",
        RedRabbitPolicy policy = {},
        WallClockMsFn wall_clock_ms = default_wall_clock_ms,
        bool crash_dump_detected = false
    );

    void set_interlock_callback(InterlockCallback cb);
    void set_wall_clock_for_test(WallClockMsFn fn);

    RedRabbitState state() const;
    bool is_blocking() const;

    void evaluate_periodic_triggers();

    json trigger_challenge(
        const std::string& reason,
        const std::string& lot_id = "",
        const std::string& actor = "SYSTEM"
    );

    json start_session(
        const std::string& operator_id,
        const std::string& lot_id,
        const std::string& model_sha256
    );

    json arm_sample(const std::string& session_id, const std::string& sample_uid);

    json record_attempt(
        const std::string& session_id,
        const std::string& sample_uid,
        const std::string& actual_verdict,
        const std::string& evidence_dir = "",
        std::optional<double> ood_energy = std::nullopt,
        std::optional<double> ood_mahalanobis = std::nullopt
    );

    bool try_consume_live_verdict(
        const std::string& outcome,
        const std::string& evidence_dir = "",
        std::optional<double> ood_energy = std::nullopt,
        std::optional<double> ood_mahalanobis = std::nullopt
    );

    json unlock_hard_lock(
        const std::string& actor_id,
        const std::string& actor_role,
        const std::string& reason_code,
        const std::string& comment
    );

    json status_json();
    json audit_summary_json(int limit = 30) const;

private:
    struct ActiveSession {
        std::string session_id;
        std::string trigger_reason;
        std::string operator_id;
        std::string lot_id;
        std::string model_sha256;
        int64_t started_at_ms{0};
        std::vector<RedRabbitPlanStep> plan;
        std::size_t current_step_index{0};
        std::optional<std::string> armed_sample_uid;
        std::unordered_set<std::string> attempted_uids;
    };

    int64_t now_ms_locked() const;
    std::optional<int64_t> latest_shift_boundary_ms(int64_t now_ms) const;
    std::optional<InterlockSignal> evaluate_periodic_triggers_locked(int64_t now_ms);
    void transition_to_challenge_locked(const std::string& reason);
    std::optional<InterlockSignal> fail_session_to_hard_lock_locked(
        const std::string& session_id,
        const std::string& session_status,
        const std::string& reason_code,
        const std::string& comment,
        int64_t now_ms
    );
    void validate_step_target_locked(
        const std::string& session_id,
        const std::string& sample_uid,
        int64_t now_ms,
        std::optional<InterlockSignal>& pending_sig
    );
    json build_status_json_locked(int64_t now_ms) const;
    void emit_interlock(const std::optional<InterlockSignal>& sig) const;

    StationDatabase& db_;
    std::string station_id_;
    RedRabbitPolicy policy_;
    WallClockMsFn wall_clock_ms_;
    InterlockCallback interlock_cb_;

    mutable std::mutex mutex_;
    RedRabbitState state_{RedRabbitState::CHALLENGE_REQUIRED};
    std::string trigger_reason_{"COLD_BOOT"};
    std::string active_lot_id_;
    std::optional<ActiveSession> active_session_;
    std::optional<json> last_finished_session_;
    std::optional<json> last_attempt_;
    std::optional<json> last_lock_event_;
    std::string last_passed_session_id_;
    int64_t last_passed_at_ms_{0};
};

}  // namespace igbt
