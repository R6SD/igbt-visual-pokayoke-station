#include "igbt/red_rabbit.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <random>
#include <stdexcept>

namespace igbt {

namespace {

std::string trim_rr(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

bool is_allowed_trigger_reason(const std::string& reason) {
    return reason == "SHIFT_CHANGE" || reason == "INTERVAL_EXPIRED" ||
           reason == "LOT_CHANGE" || reason == "COLD_BOOT" ||
           reason == "CRASH_RECOVERY" || reason == "MODEL_OR_OPTICS_CHANGE" ||
           reason == "MANUAL_AUDIT";
}

uint64_t seed_from_rng() {
    const std::string hex = random_token_hex(8);
    uint64_t val = 0;
    for (char c : hex) {
        val <<= 4;
        if (c >= '0' && c <= '9') val |= static_cast<uint64_t>(c - '0');
        else if (c >= 'a' && c <= 'f') val |= static_cast<uint64_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') val |= static_cast<uint64_t>(c - 'A' + 10);
    }
    return val;
}

json plan_step_to_json(const RedRabbitPlanStep& s) {
    return json{
        {"step_index", s.step_index},
        {"sample_uid", s.sample_uid},
        {"sample_type", s.sample_type},
        {"expected_verdict", s.expected_verdict},
        {"product_family", s.product_family},
        {"completed", s.completed},
        {"is_match", s.is_match.has_value() ? json(*s.is_match) : json(nullptr)},
        {"actual_verdict", s.actual_verdict.has_value() ? json(*s.actual_verdict) : json(nullptr)},
    };
}

}  // namespace

std::string red_rabbit_state_to_string(RedRabbitState state) {
    switch (state) {
        case RedRabbitState::PROD_READY: return "PROD_READY";
        case RedRabbitState::CHALLENGE_REQUIRED: return "CHALLENGE_REQUIRED";
        case RedRabbitState::CHALLENGE_HARD_LOCK: return "CHALLENGE_HARD_LOCK";
    }
    return "CHALLENGE_REQUIRED";
}

int64_t default_wall_clock_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

RedRabbitController::RedRabbitController(
    StationDatabase& db,
    std::string station_id,
    RedRabbitPolicy policy,
    WallClockMsFn wall_clock_ms,
    bool crash_dump_detected
)
    : db_(db),
      station_id_(std::move(station_id)),
      policy_(policy),
      wall_clock_ms_(wall_clock_ms ? std::move(wall_clock_ms) : default_wall_clock_ms) {
    if (station_id_.empty()) {
        station_id_ = "STATION_01";
    }
    if (auto last_passed = db_.rr_last_passed_session(); last_passed.has_value()) {
        last_passed_session_id_ = last_passed->value("session_id", "");
        if (last_passed->contains("completed_at_ms") && (*last_passed)["completed_at_ms"].is_number_integer()) {
            last_passed_at_ms_ = (*last_passed)["completed_at_ms"].get<int64_t>();
        }
    }
    // AEC-Q101 strict baseline: cold boot or crash recovery ALWAYS requires challenge
    state_ = RedRabbitState::CHALLENGE_REQUIRED;
    trigger_reason_ = crash_dump_detected ? "CRASH_RECOVERY" : "COLD_BOOT";
}

void RedRabbitController::set_interlock_callback(InterlockCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    interlock_cb_ = std::move(cb);
}

void RedRabbitController::set_wall_clock_for_test(WallClockMsFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    wall_clock_ms_ = fn ? std::move(fn) : default_wall_clock_ms;
}

RedRabbitState RedRabbitController::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool RedRabbitController::is_blocking() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ != RedRabbitState::PROD_READY;
}

int64_t RedRabbitController::now_ms_locked() const {
    return wall_clock_ms_ ? wall_clock_ms_() : default_wall_clock_ms();
}

std::optional<int64_t> RedRabbitController::latest_shift_boundary_ms(int64_t now_ms) const {
    if (now_ms <= 0) return std::nullopt;
    const std::time_t now_sec = static_cast<std::time_t>(now_ms / 1000);
    std::tm local_tm{};
    if (localtime_s(&local_tm, &now_sec) != 0) {
        return std::nullopt;
    }

    auto make_boundary_ms = [&](int day_offset, int hour) -> std::optional<int64_t> {
        std::tm cand = local_tm;
        cand.tm_mday += day_offset;
        cand.tm_hour = hour;
        cand.tm_min = 0;
        cand.tm_sec = 0;
        cand.tm_isdst = -1;
        const std::time_t t = std::mktime(&cand);
        if (t < 0) return std::nullopt;
        return static_cast<int64_t>(t) * 1000;
    };

    const std::optional<int64_t> candidates[] = {
        make_boundary_ms(-1, policy_.shift_hour_night),
        make_boundary_ms(0, policy_.shift_hour_day),
        make_boundary_ms(0, policy_.shift_hour_night),
    };

    std::optional<int64_t> latest;
    for (const auto& b : candidates) {
        if (!b.has_value()) continue;
        if (*b + policy_.shift_grace_ms <= now_ms) {
            if (!latest.has_value() || *b > *latest) {
                latest = *b;
            }
        }
    }
    return latest;
}

std::optional<InterlockSignal> RedRabbitController::fail_session_to_hard_lock_locked(
    const std::string& session_id,
    const std::string& session_status,
    const std::string& reason_code,
    const std::string& comment,
    int64_t now_ms
) {
    db_.rr_update_session_status(session_id, session_status, now_ms);
    if (active_session_.has_value() && active_session_->session_id == session_id) {
        json plan_arr = json::array();
        for (const auto& s : active_session_->plan) {
            plan_arr.push_back(plan_step_to_json(s));
        }
        last_finished_session_ = json{
            {"session_id", active_session_->session_id},
            {"trigger_reason", active_session_->trigger_reason},
            {"operator_id", active_session_->operator_id},
            {"lot_id", active_session_->lot_id},
            {"status", session_status},
            {"started_at_ms", active_session_->started_at_ms},
            {"completed_at_ms", now_ms},
            {"required_plan", std::move(plan_arr)},
            {"current_step_index", active_session_->current_step_index},
        };
        active_session_.reset();
    }

    state_ = RedRabbitState::CHALLENGE_HARD_LOCK;
    trigger_reason_ = reason_code;
    last_lock_event_ = db_.rr_append_lock_audit(
        "CHALLENGE_FAIL_LOCK",
        "SYSTEM",
        "SYSTEM",
        reason_code,
        comment,
        session_id,
        now_ms
    );
    return InterlockSignal::ALARM_LATCHED;
}

void RedRabbitController::transition_to_challenge_locked(const std::string& reason) {
    if (state_ == RedRabbitState::CHALLENGE_HARD_LOCK) {
        return;
    }
    const int64_t now_ms = now_ms_locked();
    if (active_session_.has_value()) {
        db_.rr_update_session_status(active_session_->session_id, "ABORTED", now_ms);
        active_session_.reset();
    }
    state_ = RedRabbitState::CHALLENGE_REQUIRED;
    trigger_reason_ = reason;
}

std::optional<InterlockSignal> RedRabbitController::evaluate_periodic_triggers_locked(int64_t now_ms) {
    if (!policy_.strict_enabled) {
        return std::nullopt;
    }
    if (state_ == RedRabbitState::CHALLENGE_REQUIRED && active_session_.has_value()) {
        if (policy_.session_timeout_ms > 0 &&
            (now_ms - active_session_->started_at_ms) > policy_.session_timeout_ms) {
            return fail_session_to_hard_lock_locked(
                active_session_->session_id,
                "TIMEOUT",
                "SESSION_TIMEOUT",
                "红兔挑战会话超时未完成，工位自动进入硬闭锁",
                now_ms
            );
        }
    }

    if (state_ == RedRabbitState::PROD_READY && last_passed_at_ms_ > 0) {
        if (policy_.interval_ms > 0 && (now_ms - last_passed_at_ms_) >= policy_.interval_ms) {
            transition_to_challenge_locked("INTERVAL_EXPIRED");
            return InterlockSignal::LOCKED;
        }
        if (auto boundary = latest_shift_boundary_ms(now_ms); boundary.has_value()) {
            if (last_passed_at_ms_ < *boundary) {
                transition_to_challenge_locked("SHIFT_CHANGE");
                return InterlockSignal::LOCKED;
            }
        }
    }
    return std::nullopt;
}

void RedRabbitController::emit_interlock(const std::optional<InterlockSignal>& sig) const {
    if (!sig.has_value()) return;
    InterlockCallback cb;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cb = interlock_cb_;
    }
    if (cb) {
        cb(*sig);
    }
}

void RedRabbitController::evaluate_periodic_triggers() {
    std::optional<InterlockSignal> sig;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sig = evaluate_periodic_triggers_locked(now_ms_locked());
    }
    emit_interlock(sig);
}

json RedRabbitController::trigger_challenge(
    const std::string& reason,
    const std::string& lot_id,
    const std::string& actor
) {
    if (!is_allowed_trigger_reason(reason)) {
        throw DatabaseHttpError(422, "INVALID_RED_RABBIT_TRIGGER_REASON");
    }

    std::optional<InterlockSignal> sig;
    json out;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t now_ms = now_ms_locked();
        const std::string clean_lot = trim_rr(lot_id);
        if (!clean_lot.empty()) {
            active_lot_id_ = clean_lot;
        }
        if (state_ == RedRabbitState::CHALLENGE_HARD_LOCK) {
            // Hard lock cannot be cleared or downgraded by triggering a new challenge.
            out = build_status_json_locked(now_ms);
        } else {
            // Preserve COLD_BOOT/CRASH_RECOVERY root reason if cold boot hasn't passed yet and trigger is incidental
            const bool keep_existing_cold_reason =
                state_ == RedRabbitState::CHALLENGE_REQUIRED &&
                !active_session_.has_value() &&
                (trigger_reason_ == "COLD_BOOT" || trigger_reason_ == "CRASH_RECOVERY") &&
                reason == "MODEL_OR_OPTICS_CHANGE" &&
                actor == "AUTOLOAD";
            if (!keep_existing_cold_reason) {
                transition_to_challenge_locked(reason);
                sig = InterlockSignal::LOCKED;
            }
            out = build_status_json_locked(now_ms);
        }
    }
    emit_interlock(sig);
    return out;
}

json RedRabbitController::start_session(
    const std::string& operator_id,
    const std::string& lot_id,
    const std::string& model_sha256
) {
    std::optional<InterlockSignal> sig;
    json out;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t now_ms = now_ms_locked();
        sig = evaluate_periodic_triggers_locked(now_ms);

        if (state_ == RedRabbitState::CHALLENGE_HARD_LOCK) {
            throw DatabaseHttpError(409, "RED_RABBIT_HARD_LOCK_UNLOCK_REQUIRED");
        }

        if (state_ == RedRabbitState::PROD_READY) {
            transition_to_challenge_locked("MANUAL_AUDIT");
            sig = InterlockSignal::LOCKED;
        } else if (active_session_.has_value()) {
            db_.rr_update_session_status(active_session_->session_id, "ABORTED", now_ms);
            active_session_.reset();
        }

        const std::string clean_op = trim_rr(operator_id).empty() ? "OP_DEFAULT" : trim_rr(operator_id);
        const std::string clean_lot = trim_rr(lot_id).empty()
                                          ? (active_lot_id_.empty() ? "LOT_DEFAULT" : active_lot_id_)
                                          : trim_rr(lot_id);
        active_lot_id_ = clean_lot;

        const json all_active = db_.rr_list_samples(true);
        std::vector<RedRabbitPlanStep> ok_pool;
        std::vector<RedRabbitPlanStep> ng_pool;
        for (const auto& item : all_active) {
            const int64_t exp = item.value("expires_at_ms", int64_t{0});
            if (exp <= now_ms) continue;
            RedRabbitPlanStep step;
            step.sample_uid = item.value("sample_uid", "");
            step.sample_type = item.value("sample_type", "");
            step.expected_verdict = item.value("expected_verdict", "");
            step.product_family = item.value("product_family", "IGBT_TO247");
            if (step.sample_type == "OK_GOLDEN" && step.expected_verdict == "PASS") {
                ok_pool.push_back(std::move(step));
            } else if (step.expected_verdict == "ALARM") {
                ng_pool.push_back(std::move(step));
            }
        }

        if (static_cast<int>(ok_pool.size()) < policy_.required_ok_count ||
            static_cast<int>(ng_pool.size()) < policy_.required_ng_count) {
            throw DatabaseHttpError(422, "RED_RABBIT_INSUFFICIENT_GOLDEN_SAMPLES");
        }

        std::mt19937_64 rng(seed_from_rng());
        std::shuffle(ok_pool.begin(), ok_pool.end(), rng);
        std::shuffle(ng_pool.begin(), ng_pool.end(), rng);

        std::vector<RedRabbitPlanStep> chosen;
        chosen.reserve(static_cast<std::size_t>(policy_.required_ok_count + policy_.required_ng_count));
        for (int i = 0; i < policy_.required_ok_count; ++i) {
            chosen.push_back(ok_pool[static_cast<std::size_t>(i)]);
        }
        for (int i = 0; i < policy_.required_ng_count; ++i) {
            chosen.push_back(ng_pool[static_cast<std::size_t>(i)]);
        }
        std::shuffle(chosen.begin(), chosen.end(), rng);
        for (std::size_t i = 0; i < chosen.size(); ++i) {
            chosen[i].step_index = static_cast<int>(i);
        }

        json plan_json = json::array();
        for (const auto& s : chosen) {
            plan_json.push_back(json{
                {"step_index", s.step_index},
                {"sample_uid", s.sample_uid},
                {"sample_type", s.sample_type},
                {"expected_verdict", s.expected_verdict},
                {"product_family", s.product_family},
            });
        }

        const std::string session_id = "RR_" + random_token_hex(8);
        const std::string clean_sha = model_sha256.empty() ? "heuristic" : model_sha256;
        db_.rr_create_session(json{
            {"session_id", session_id},
            {"station_id", station_id_},
            {"trigger_reason", trigger_reason_.empty() ? "MANUAL_AUDIT" : trigger_reason_},
            {"operator_id", clean_op},
            {"lot_id", clean_lot},
            {"required_plan", plan_json},
            {"status", "IN_PROGRESS"},
            {"started_at_ms", now_ms},
            {"completed_at_ms", nullptr},
            {"model_sha256", clean_sha},
        });

        ActiveSession sess;
        sess.session_id = session_id;
        sess.trigger_reason = trigger_reason_.empty() ? "MANUAL_AUDIT" : trigger_reason_;
        sess.operator_id = clean_op;
        sess.lot_id = clean_lot;
        sess.model_sha256 = clean_sha;
        sess.started_at_ms = now_ms;
        sess.plan = std::move(chosen);
        active_session_ = std::move(sess);
        last_attempt_.reset();

        out = build_status_json_locked(now_ms);
    } catch (...) {
        emit_interlock(sig);
        throw;
    }
    emit_interlock(sig);
    return out;
}

void RedRabbitController::validate_step_target_locked(
    const std::string& session_id,
    const std::string& sample_uid,
    int64_t now_ms,
    std::optional<InterlockSignal>& pending_sig
) {
    if (state_ == RedRabbitState::CHALLENGE_HARD_LOCK) {
        throw DatabaseHttpError(409, "RED_RABBIT_HARD_LOCK_UNLOCK_REQUIRED");
    }
    if (state_ != RedRabbitState::CHALLENGE_REQUIRED || !active_session_.has_value()) {
        throw DatabaseHttpError(409, "RED_RABBIT_NO_ACTIVE_SESSION");
    }
    if (active_session_->session_id != session_id) {
        throw DatabaseHttpError(409, "RED_RABBIT_SESSION_MISMATCH");
    }
    if (policy_.session_timeout_ms > 0 &&
        (now_ms - active_session_->started_at_ms) > policy_.session_timeout_ms) {
        pending_sig = fail_session_to_hard_lock_locked(
            active_session_->session_id,
            "TIMEOUT",
            "SESSION_TIMEOUT",
            "红兔挑战会话超时未完成，工位自动进入硬闭锁",
            now_ms
        );
        throw DatabaseHttpError(409, "RED_RABBIT_SESSION_TIMEOUT");
    }
    if (active_session_->current_step_index >= active_session_->plan.size()) {
        throw DatabaseHttpError(409, "RED_RABBIT_SESSION_ALREADY_COMPLETE");
    }

    const std::string clean_uid = trim_rr(sample_uid);
    if (clean_uid.empty()) {
        throw DatabaseHttpError(422, "MISSING_SAMPLE_UID");
    }
    if (active_session_->attempted_uids.find(clean_uid) != active_session_->attempted_uids.end()) {
        throw DatabaseHttpError(409, "RED_RABBIT_DUPLICATE_SAMPLE_UID");
    }

    const auto& step = active_session_->plan[active_session_->current_step_index];
    if (step.sample_uid != clean_uid) {
        throw DatabaseHttpError(422, "RED_RABBIT_WRONG_SAMPLE_UID_FOR_STEP");
    }

    auto sample_row = db_.rr_get_sample(clean_uid);
    if (!sample_row.has_value() || !sample_row->value("is_active", false)) {
        throw DatabaseHttpError(422, "RED_RABBIT_SAMPLE_INACTIVE");
    }
    if (sample_row->value("expires_at_ms", int64_t{0}) <= now_ms) {
        throw DatabaseHttpError(422, "RED_RABBIT_SAMPLE_EXPIRED");
    }
}

json RedRabbitController::arm_sample(const std::string& session_id, const std::string& sample_uid) {
    std::optional<InterlockSignal> pending_sig;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t now_ms = now_ms_locked();
        validate_step_target_locked(session_id, sample_uid, now_ms, pending_sig);
        active_session_->armed_sample_uid = trim_rr(sample_uid);
        return build_status_json_locked(now_ms);
    } catch (...) {
        emit_interlock(pending_sig);
        throw;
    }
}

json RedRabbitController::record_attempt(
    const std::string& session_id,
    const std::string& sample_uid,
    const std::string& actual_verdict,
    const std::string& evidence_dir,
    std::optional<double> ood_energy,
    std::optional<double> ood_mahalanobis
) {
    const std::string clean_verdict = trim_rr(actual_verdict);
    if (clean_verdict != "PASS" && clean_verdict != "ALARM") {
        throw DatabaseHttpError(422, "INVALID_ACTUAL_VERDICT");
    }

    std::optional<InterlockSignal> pending_sig;
    json out;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t now_ms = now_ms_locked();
        validate_step_target_locked(session_id, sample_uid, now_ms, pending_sig);

        auto& step = active_session_->plan[active_session_->current_step_index];
        const bool is_match = (clean_verdict == step.expected_verdict);

        step.completed = true;
        step.is_match = is_match;
        step.actual_verdict = clean_verdict;
        active_session_->attempted_uids.insert(step.sample_uid);
        active_session_->armed_sample_uid.reset();

        last_attempt_ = db_.rr_record_attempt(json{
            {"session_id", active_session_->session_id},
            {"step_index", step.step_index},
            {"sample_uid", step.sample_uid},
            {"expected_verdict", step.expected_verdict},
            {"actual_verdict", clean_verdict},
            {"ood_energy", ood_energy.has_value() ? json(*ood_energy) : json(nullptr)},
            {"ood_mahalanobis", ood_mahalanobis.has_value() ? json(*ood_mahalanobis) : json(nullptr)},
            {"is_match", is_match},
            {"evidence_dir", evidence_dir},
            {"recorded_at_ms", now_ms},
        });

        if (!is_match) {
            const std::string reason_code =
                (step.expected_verdict == "ALARM" && clean_verdict == "PASS")
                    ? "CRITICAL_NG_MISSED_AS_PASS"
                    : "GOLDEN_SAMPLE_MISMATCH";
            const std::string comment =
                "金样 " + step.sample_uid + " (" + step.sample_type + ") 期望 " +
                step.expected_verdict + "，实测 " + clean_verdict;
            pending_sig = fail_session_to_hard_lock_locked(
                active_session_->session_id,
                "FAILED",
                reason_code,
                comment,
                now_ms
            );
        } else {
            ++active_session_->current_step_index;
            if (active_session_->current_step_index >= active_session_->plan.size()) {
                db_.rr_update_session_status(active_session_->session_id, "PASSED", now_ms);
                last_passed_session_id_ = active_session_->session_id;
                last_passed_at_ms_ = now_ms;

                json plan_arr = json::array();
                for (const auto& s : active_session_->plan) {
                    plan_arr.push_back(plan_step_to_json(s));
                }
                last_finished_session_ = json{
                    {"session_id", active_session_->session_id},
                    {"trigger_reason", active_session_->trigger_reason},
                    {"operator_id", active_session_->operator_id},
                    {"lot_id", active_session_->lot_id},
                    {"status", "PASSED"},
                    {"started_at_ms", active_session_->started_at_ms},
                    {"completed_at_ms", now_ms},
                    {"required_plan", std::move(plan_arr)},
                    {"current_step_index", active_session_->current_step_index},
                };
                active_session_.reset();
                state_ = RedRabbitState::PROD_READY;
                trigger_reason_.clear();
                pending_sig = InterlockSignal::LOCKED;
            }
        }

        out = build_status_json_locked(now_ms);
    } catch (...) {
        emit_interlock(pending_sig);
        throw;
    }

    emit_interlock(pending_sig);
    return out;
}

bool RedRabbitController::try_consume_live_verdict(
    const std::string& outcome,
    const std::string& evidence_dir
) {
    if (outcome != "PASS" && outcome != "ALARM") {
        return false;
    }
    std::string session_id;
    std::string armed_uid;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != RedRabbitState::CHALLENGE_REQUIRED || !active_session_.has_value()) {
            return false;
        }
        if (!active_session_->armed_sample_uid.has_value() || active_session_->armed_sample_uid->empty()) {
            return false;
        }
        session_id = active_session_->session_id;
        armed_uid = *active_session_->armed_sample_uid;
    }

    try {
        record_attempt(session_id, armed_uid, outcome, evidence_dir);
        return true;
    } catch (...) {
        return false;
    }
}

json RedRabbitController::unlock_hard_lock(
    const std::string& actor_id,
    const std::string& actor_role,
    const std::string& reason_code,
    const std::string& comment
) {
    const std::string clean_actor = trim_rr(actor_id);
    const std::string clean_role = trim_rr(actor_role);
    const std::string clean_reason = trim_rr(reason_code);
    const std::string clean_comment = trim_rr(comment);

    if (clean_role != "ENGINEER" && clean_role != "QE") {
        throw DatabaseHttpError(422, "RED_RABBIT_UNLOCK_ROLE_INVALID");
    }
    if (clean_actor.empty()) {
        throw DatabaseHttpError(422, "RED_RABBIT_UNLOCK_ACTOR_REQUIRED");
    }
    if (clean_reason.empty()) {
        throw DatabaseHttpError(422, "RED_RABBIT_UNLOCK_REASON_REQUIRED");
    }
    if (clean_comment.size() < 6) {
        throw DatabaseHttpError(422, "RED_RABBIT_UNLOCK_COMMENT_TOO_SHORT");
    }

    std::optional<InterlockSignal> sig;
    json out;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != RedRabbitState::CHALLENGE_HARD_LOCK) {
            throw DatabaseHttpError(409, "RED_RABBIT_NOT_HARD_LOCKED");
        }
        const int64_t now_ms = now_ms_locked();
        std::optional<std::string> rel_sess;
        if (last_finished_session_.has_value()) {
            rel_sess = last_finished_session_->value("session_id", "");
        }
        last_lock_event_ = db_.rr_append_lock_audit(
            "ENGINEER_UNLOCK",
            clean_actor,
            clean_role,
            clean_reason,
            clean_comment,
            rel_sess,
            now_ms
        );
        // AEC-Q101 strict: unlocking hard lock MUST return to CHALLENGE_REQUIRED, never directly PROD_READY
        state_ = RedRabbitState::CHALLENGE_REQUIRED;
        trigger_reason_ = "MANUAL_AUDIT";
        active_session_.reset();
        sig = InterlockSignal::LOCKED;
        out = build_status_json_locked(now_ms);
    } catch (...) {
        emit_interlock(sig);
        throw;
    }
    emit_interlock(sig);
    return out;
}

json RedRabbitController::build_status_json_locked(int64_t now_ms) const {
    json session_json = nullptr;
    if (active_session_.has_value()) {
        json plan_arr = json::array();
        for (const auto& s : active_session_->plan) {
            plan_arr.push_back(plan_step_to_json(s));
        }
        json current_step = nullptr;
        if (active_session_->current_step_index < active_session_->plan.size()) {
            current_step = plan_step_to_json(active_session_->plan[active_session_->current_step_index]);
        }
        const int64_t elapsed = std::max<int64_t>(0, now_ms - active_session_->started_at_ms);
        const int64_t remaining = std::max<int64_t>(0, policy_.session_timeout_ms - elapsed);
        session_json = json{
            {"session_id", active_session_->session_id},
            {"trigger_reason", active_session_->trigger_reason},
            {"operator_id", active_session_->operator_id},
            {"lot_id", active_session_->lot_id},
            {"status", "IN_PROGRESS"},
            {"started_at_ms", active_session_->started_at_ms},
            {"timeout_remaining_ms", remaining},
            {"current_step_index", active_session_->current_step_index},
            {"total_steps", active_session_->plan.size()},
            {"armed_sample_uid", active_session_->armed_sample_uid.has_value() ? json(*active_session_->armed_sample_uid) : json(nullptr)},
            {"current_step", std::move(current_step)},
            {"required_plan", std::move(plan_arr)},
        };
    } else if (last_finished_session_.has_value()) {
        session_json = *last_finished_session_;
    }

    json next_due_ms = nullptr;
    if (state_ == RedRabbitState::PROD_READY && last_passed_at_ms_ > 0 && policy_.interval_ms > 0) {
        next_due_ms = last_passed_at_ms_ + policy_.interval_ms;
    }

    return json{
        {"station_id", station_id_},
        {"strict_enabled", policy_.strict_enabled},
        {"state", red_rabbit_state_to_string(state_)},
        {"blocking", state_ != RedRabbitState::PROD_READY},
        {"hard_locked", state_ == RedRabbitState::CHALLENGE_HARD_LOCK},
        {"trigger_reason", trigger_reason_.empty() ? json(nullptr) : json(trigger_reason_)},
        {"active_lot_id", active_lot_id_.empty() ? json(nullptr) : json(active_lot_id_)},
        {"required_ok_count", policy_.required_ok_count},
        {"required_ng_count", policy_.required_ng_count},
        {"interval_ms", policy_.interval_ms},
        {"session_timeout_ms", policy_.session_timeout_ms},
        {"last_passed_session_id", last_passed_session_id_.empty() ? json(nullptr) : json(last_passed_session_id_)},
        {"last_passed_at_ms", last_passed_at_ms_ > 0 ? json(last_passed_at_ms_) : json(nullptr)},
        {"next_due_at_ms", next_due_ms},
        {"now_ms", now_ms},
        {"session", std::move(session_json)},
        {"last_attempt", last_attempt_.has_value() ? *last_attempt_ : json(nullptr)},
        {"last_lock_event", last_lock_event_.has_value() ? *last_lock_event_ : json(nullptr)},
    };
}

json RedRabbitController::status_json() {
    std::optional<InterlockSignal> sig;
    json out;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t now_ms = now_ms_locked();
        sig = evaluate_periodic_triggers_locked(now_ms);
        out = build_status_json_locked(now_ms);
    } catch (...) {
        emit_interlock(sig);
        throw;
    }
    emit_interlock(sig);
    return out;
}

json RedRabbitController::audit_summary_json(int limit) const {
    return json{
        {"chain_verification", db_.rr_verify_lock_audit_chain()},
        {"lock_audit", db_.rr_list_lock_audit(limit)},
        {"recent_sessions", db_.rr_list_sessions(std::min(limit, 20))},
        {"samples", db_.rr_list_samples(false)},
    };
}

}  // namespace igbt
