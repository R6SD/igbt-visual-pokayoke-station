#include "igbt/database.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace igbt {

namespace {

class RRStmtGuard {
public:
    RRStmtGuard(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
            std::string err = db ? sqlite3_errmsg(db) : "null sqlite3 handle";
            throw std::runtime_error("SQLite prepare failed: " + err);
        }
    }
    ~RRStmtGuard() {
        if (stmt_) sqlite3_finalize(stmt_);
    }
    RRStmtGuard(const RRStmtGuard&) = delete;
    RRStmtGuard& operator=(const RRStmtGuard&) = delete;

    sqlite3_stmt* get() const noexcept { return stmt_; }

private:
    sqlite3_stmt* stmt_{nullptr};
};

class RRTransactionGuard {
public:
    explicit RRTransactionGuard(sqlite3* db) : db_(db) {
        char* errmsg = nullptr;
        if (sqlite3_exec(db_, "BEGIN IMMEDIATE TRANSACTION;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::string msg = errmsg ? errmsg : "BEGIN IMMEDIATE failed";
            if (errmsg) sqlite3_free(errmsg);
            throw std::runtime_error(msg);
        }
    }
    ~RRTransactionGuard() {
        if (!committed_ && db_) {
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        }
    }
    void commit() {
        if (committed_) return;
        char* errmsg = nullptr;
        if (sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::string msg = errmsg ? errmsg : "COMMIT failed";
            if (errmsg) sqlite3_free(errmsg);
            throw std::runtime_error(msg);
        }
        committed_ = true;
    }

private:
    sqlite3* db_{nullptr};
    bool committed_{false};
};

void rr_bind_text(sqlite3_stmt* stmt, int idx, const std::string& val) {
    sqlite3_bind_text(stmt, idx, val.c_str(), static_cast<int>(val.size()), SQLITE_TRANSIENT);
}

void rr_bind_text_or_null(sqlite3_stmt* stmt, int idx, const std::optional<std::string>& val) {
    if (val.has_value() && !val->empty()) {
        rr_bind_text(stmt, idx, *val);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

void rr_bind_double_or_null(sqlite3_stmt* stmt, int idx, const std::optional<double>& val) {
    if (val.has_value() && std::isfinite(*val)) {
        sqlite3_bind_double(stmt, idx, *val);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

void rr_bind_int64_or_null(sqlite3_stmt* stmt, int idx, const std::optional<int64_t>& val) {
    if (val.has_value()) {
        sqlite3_bind_int64(stmt, idx, *val);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

std::string rr_col_text(sqlite3_stmt* stmt, int idx) {
    const unsigned char* txt = sqlite3_column_text(stmt, idx);
    if (!txt) return {};
    int n = sqlite3_column_bytes(stmt, idx);
    return std::string(reinterpret_cast<const char*>(txt), static_cast<std::size_t>(std::max(0, n)));
}

json rr_col_text_or_null(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return rr_col_text(stmt, idx);
}

json rr_col_double_or_null(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return sqlite3_column_double(stmt, idx);
}

json rr_col_int64_or_null(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return static_cast<int64_t>(sqlite3_column_int64(stmt, idx));
}

int64_t rr_wall_clock_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

std::string trim_copy(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

bool is_valid_sample_uid(const std::string& uid) {
    if (uid.empty() || uid.size() > 64) return false;
    for (unsigned char c : uid) {
        if (!(std::isalnum(c) || c == '-' || c == '_' || c == '.')) {
            return false;
        }
    }
    return true;
}

bool is_allowed_sample_type(const std::string& t) {
    return t == "OK_GOLDEN" || t == "NG_UNMARKED" || t == "NG_PARTIAL" ||
           t == "NG_REVERSED" || t == "NG_OOD";
}

bool is_allowed_verdict(const std::string& v) {
    return v == "PASS" || v == "ALARM";
}

bool is_allowed_audit_event_type(const std::string& t) {
    return t == "CHALLENGE_FAIL_LOCK" || t == "WATCHDOG_FAULT_LOCK" ||
           t == "MES_OFFLINE_LOCK" || t == "ENGINEER_UNLOCK" || t == "POLICY_OVERRIDE";
}

bool is_allowed_actor_role(const std::string& r) {
    return r == "SYSTEM" || r == "OPERATOR" || r == "ENGINEER" || r == "QE";
}

const std::string& genesis_hash_64() {
    static const std::string kGenesis(64, '0');
    return kGenesis;
}

std::string compute_audit_row_hash(
    const std::string& prev_hash,
    const std::string& event_id,
    const std::string& event_type,
    const std::string& actor_id,
    const std::string& actor_role,
    const std::string& reason_code,
    const std::string& comment,
    int64_t created_at_ms
) {
    const std::string canonical =
        prev_hash + "|" + event_id + "|" + event_type + "|" + actor_id + "|" +
        actor_role + "|" + reason_code + "|" + comment + "|" + std::to_string(created_at_ms);
    return sha256_hex(canonical);
}

json read_sample_row(sqlite3_stmt* stmt) {
    return json{
        {"sample_uid", rr_col_text(stmt, 0)},
        {"sample_type", rr_col_text(stmt, 1)},
        {"expected_verdict", rr_col_text(stmt, 2)},
        {"product_family", rr_col_text(stmt, 3)},
        {"is_active", sqlite3_column_int(stmt, 4) != 0},
        {"calibrated_by", rr_col_text(stmt, 5)},
        {"expires_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 6))},
        {"created_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 7))},
    };
}

json read_session_row(sqlite3_stmt* stmt) {
    json plan = json::array();
    const std::string plan_raw = rr_col_text(stmt, 5);
    try {
        json parsed = json::parse(plan_raw);
        if (parsed.is_array()) plan = std::move(parsed);
    } catch (...) {
    }
    return json{
        {"session_id", rr_col_text(stmt, 0)},
        {"station_id", rr_col_text(stmt, 1)},
        {"trigger_reason", rr_col_text(stmt, 2)},
        {"operator_id", rr_col_text(stmt, 3)},
        {"lot_id", rr_col_text(stmt, 4)},
        {"required_plan", std::move(plan)},
        {"status", rr_col_text(stmt, 6)},
        {"started_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 7))},
        {"completed_at_ms", rr_col_int64_or_null(stmt, 8)},
        {"model_sha256", rr_col_text(stmt, 9)},
    };
}

json read_audit_row(sqlite3_stmt* stmt) {
    return json{
        {"audit_seq", static_cast<int64_t>(sqlite3_column_int64(stmt, 0))},
        {"event_id", rr_col_text(stmt, 1)},
        {"event_type", rr_col_text(stmt, 2)},
        {"actor_id", rr_col_text(stmt, 3)},
        {"actor_role", rr_col_text(stmt, 4)},
        {"reason_code", rr_col_text(stmt, 5)},
        {"comment", rr_col_text(stmt, 6)},
        {"related_session", rr_col_text_or_null(stmt, 7)},
        {"created_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 8))},
        {"prev_hash", rr_col_text(stmt, 9)},
        {"row_hash", rr_col_text(stmt, 10)},
    };
}

}  // namespace

json StationDatabase::rr_list_samples(bool active_only) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = active_only
        ? "SELECT sample_uid, sample_type, expected_verdict, product_family, is_active, calibrated_by, expires_at_ms, created_at_ms "
          "FROM red_rabbit_samples WHERE is_active=1 ORDER BY sample_type ASC, sample_uid ASC;"
        : "SELECT sample_uid, sample_type, expected_verdict, product_family, is_active, calibrated_by, expires_at_ms, created_at_ms "
          "FROM red_rabbit_samples ORDER BY is_active DESC, sample_type ASC, sample_uid ASC;";
    RRStmtGuard stmt(db_, sql);
    json out = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        out.push_back(read_sample_row(stmt.get()));
    }
    return out;
}

std::optional<json> StationDatabase::rr_get_sample(const std::string& sample_uid) const {
    std::lock_guard<std::mutex> lock(mutex_);
    RRStmtGuard stmt(
        db_,
        "SELECT sample_uid, sample_type, expected_verdict, product_family, is_active, calibrated_by, expires_at_ms, created_at_ms "
        "FROM red_rabbit_samples WHERE sample_uid=?;"
    );
    rr_bind_text(stmt.get(), 1, trim_copy(sample_uid));
    if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
        return std::nullopt;
    }
    return read_sample_row(stmt.get());
}

json StationDatabase::rr_upsert_sample(const json& payload, const std::string& actor) {
    if (!payload.is_object()) {
        throw DatabaseHttpError(422, "INVALID_GOLDEN_SAMPLE_PAYLOAD");
    }

    const std::string sample_uid = trim_copy(payload.value("sample_uid", ""));
    if (!is_valid_sample_uid(sample_uid)) {
        throw DatabaseHttpError(422, "INVALID_SAMPLE_UID");
    }

    const std::string sample_type = trim_copy(payload.value("sample_type", ""));
    if (!is_allowed_sample_type(sample_type)) {
        throw DatabaseHttpError(422, "INVALID_SAMPLE_TYPE");
    }

    const std::string derived_verdict = (sample_type == "OK_GOLDEN") ? "PASS" : "ALARM";
    std::string expected_verdict = trim_copy(payload.value("expected_verdict", derived_verdict));
    if (!is_allowed_verdict(expected_verdict)) {
        throw DatabaseHttpError(422, "INVALID_EXPECTED_VERDICT");
    }
    if (expected_verdict != derived_verdict) {
        throw DatabaseHttpError(422, "SAMPLE_TYPE_VERDICT_MISMATCH");
    }

    std::string product_family = trim_copy(payload.value("product_family", "IGBT_TO247"));
    if (product_family.empty()) product_family = "IGBT_TO247";

    bool is_active = true;
    if (payload.contains("is_active")) {
        if (payload["is_active"].is_boolean()) {
            is_active = payload["is_active"].get<bool>();
        } else if (payload["is_active"].is_number_integer()) {
            is_active = payload["is_active"].get<int64_t>() != 0;
        } else {
            throw DatabaseHttpError(422, "INVALID_IS_ACTIVE");
        }
    }

    const int64_t now_ms = rr_wall_clock_ms();
    int64_t expires_at_ms = now_ms + 365LL * 24 * 3600 * 1000;
    if (payload.contains("expires_at_ms") && !payload["expires_at_ms"].is_null()) {
        if (!payload["expires_at_ms"].is_number_integer()) {
            throw DatabaseHttpError(422, "INVALID_EXPIRES_AT_MS");
        }
        expires_at_ms = payload["expires_at_ms"].get<int64_t>();
        if (expires_at_ms <= 0) {
            throw DatabaseHttpError(422, "INVALID_EXPIRES_AT_MS");
        }
    }

    const std::string calibrated_by = trim_copy(actor).empty() ? "engineer" : trim_copy(actor);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        RRTransactionGuard tx(db_);
        RRStmtGuard ins(
            db_,
            "INSERT INTO red_rabbit_samples(sample_uid, sample_type, expected_verdict, product_family, is_active, calibrated_by, expires_at_ms, created_at_ms) "
            "VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(sample_uid) DO UPDATE SET "
            "sample_type=excluded.sample_type, "
            "expected_verdict=excluded.expected_verdict, "
            "product_family=excluded.product_family, "
            "is_active=excluded.is_active, "
            "calibrated_by=excluded.calibrated_by, "
            "expires_at_ms=excluded.expires_at_ms;"
        );
        rr_bind_text(ins.get(), 1, sample_uid);
        rr_bind_text(ins.get(), 2, sample_type);
        rr_bind_text(ins.get(), 3, expected_verdict);
        rr_bind_text(ins.get(), 4, product_family);
        sqlite3_bind_int(ins.get(), 5, is_active ? 1 : 0);
        rr_bind_text(ins.get(), 6, calibrated_by);
        sqlite3_bind_int64(ins.get(), 7, expires_at_ms);
        sqlite3_bind_int64(ins.get(), 8, now_ms);
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to upsert red_rabbit_sample: ") + sqlite3_errmsg(db_));
        }
        tx.commit();
    }

    auto saved = rr_get_sample(sample_uid);
    if (!saved.has_value()) {
        throw std::runtime_error("Upserted golden sample missing on readback");
    }
    return *saved;
}

json StationDatabase::rr_create_session(const json& session_obj) {
    if (!session_obj.is_object()) {
        throw std::invalid_argument("session_obj must be a JSON object");
    }

    const std::string session_id = session_obj.value("session_id", "");
    const std::string station_id = session_obj.value("station_id", "STATION_01");
    const std::string trigger_reason = session_obj.value("trigger_reason", "COLD_BOOT");
    const std::string operator_id = session_obj.value("operator_id", "OP_DEFAULT");
    const std::string lot_id = session_obj.value("lot_id", "UNBOUND_LOT");
    const json required_plan = session_obj.contains("required_plan") && session_obj["required_plan"].is_array()
                                   ? session_obj["required_plan"]
                                   : json::array();
    const std::string status = session_obj.value("status", "IN_PROGRESS");
    const int64_t started_at_ms = session_obj.value("started_at_ms", rr_wall_clock_ms());
    std::optional<int64_t> completed_at_ms;
    if (session_obj.contains("completed_at_ms") && session_obj["completed_at_ms"].is_number_integer()) {
        completed_at_ms = session_obj["completed_at_ms"].get<int64_t>();
    }
    const std::string model_sha256 = session_obj.value("model_sha256", "heuristic");

    {
        std::lock_guard<std::mutex> lock(mutex_);
        RRTransactionGuard tx(db_);
        RRStmtGuard ins(
            db_,
            "INSERT INTO red_rabbit_sessions(session_id, station_id, trigger_reason, operator_id, lot_id, required_plan, status, started_at_ms, completed_at_ms, model_sha256) "
            "VALUES(?,?,?,?,?,?,?,?,?,?);"
        );
        rr_bind_text(ins.get(), 1, session_id);
        rr_bind_text(ins.get(), 2, station_id);
        rr_bind_text(ins.get(), 3, trigger_reason);
        rr_bind_text(ins.get(), 4, operator_id);
        rr_bind_text(ins.get(), 5, lot_id);
        rr_bind_text(ins.get(), 6, required_plan.dump());
        rr_bind_text(ins.get(), 7, status);
        sqlite3_bind_int64(ins.get(), 8, started_at_ms);
        rr_bind_int64_or_null(ins.get(), 9, completed_at_ms);
        rr_bind_text(ins.get(), 10, model_sha256);
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to insert red_rabbit_session: ") + sqlite3_errmsg(db_));
        }
        tx.commit();
    }

    auto created = rr_get_session(session_id);
    return created.has_value() ? *created : session_obj;
}

std::optional<json> StationDatabase::rr_get_session(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    RRStmtGuard stmt(
        db_,
        "SELECT session_id, station_id, trigger_reason, operator_id, lot_id, required_plan, status, started_at_ms, completed_at_ms, model_sha256 "
        "FROM red_rabbit_sessions WHERE session_id=?;"
    );
    rr_bind_text(stmt.get(), 1, session_id);
    if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
        return std::nullopt;
    }
    return read_session_row(stmt.get());
}

json StationDatabase::rr_list_sessions(int limit) const {
    const int safe_limit = std::clamp(limit, 1, 200);
    std::lock_guard<std::mutex> lock(mutex_);
    RRStmtGuard stmt(
        db_,
        "SELECT session_id, station_id, trigger_reason, operator_id, lot_id, required_plan, status, started_at_ms, completed_at_ms, model_sha256 "
        "FROM red_rabbit_sessions ORDER BY started_at_ms DESC LIMIT ?;"
    );
    sqlite3_bind_int(stmt.get(), 1, safe_limit);
    json out = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        out.push_back(read_session_row(stmt.get()));
    }
    return out;
}

void StationDatabase::rr_update_session_status(
    const std::string& session_id,
    const std::string& status,
    std::optional<int64_t> completed_at_ms
) {
    std::lock_guard<std::mutex> lock(mutex_);
    RRTransactionGuard tx(db_);
    RRStmtGuard upd(
        db_,
        "UPDATE red_rabbit_sessions SET status=?, completed_at_ms=? WHERE session_id=?;"
    );
    rr_bind_text(upd.get(), 1, status);
    rr_bind_int64_or_null(upd.get(), 2, completed_at_ms);
    rr_bind_text(upd.get(), 3, session_id);
    if (sqlite3_step(upd.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("Failed to update red_rabbit_session: ") + sqlite3_errmsg(db_));
    }
    tx.commit();
}

json StationDatabase::rr_record_attempt(const json& attempt_obj) {
    if (!attempt_obj.is_object()) {
        throw std::invalid_argument("attempt_obj must be a JSON object");
    }

    const std::string session_id = attempt_obj.value("session_id", "");
    const int step_index = attempt_obj.value("step_index", 0);
    const std::string sample_uid = attempt_obj.value("sample_uid", "");
    const std::string expected_verdict = attempt_obj.value("expected_verdict", "");
    const std::string actual_verdict = attempt_obj.value("actual_verdict", "");
    std::optional<double> ood_energy;
    if (attempt_obj.contains("ood_energy") && attempt_obj["ood_energy"].is_number()) {
        ood_energy = attempt_obj["ood_energy"].get<double>();
    }
    std::optional<double> ood_mahalanobis;
    if (attempt_obj.contains("ood_mahalanobis") && attempt_obj["ood_mahalanobis"].is_number()) {
        ood_mahalanobis = attempt_obj["ood_mahalanobis"].get<double>();
    }
    const bool is_match = attempt_obj.value("is_match", false);
    const std::string evidence_dir = attempt_obj.value("evidence_dir", "");
    const int64_t recorded_at_ms = attempt_obj.value("recorded_at_ms", rr_wall_clock_ms());

    int64_t assigned_id = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RRTransactionGuard tx(db_);
        RRStmtGuard ins(
            db_,
            "INSERT INTO red_rabbit_attempts(session_id, step_index, sample_uid, expected_verdict, actual_verdict, ood_energy, ood_mahalanobis, is_match, evidence_dir, recorded_at_ms) "
            "VALUES(?,?,?,?,?,?,?,?,?,?);"
        );
        rr_bind_text(ins.get(), 1, session_id);
        sqlite3_bind_int(ins.get(), 2, step_index);
        rr_bind_text(ins.get(), 3, sample_uid);
        rr_bind_text(ins.get(), 4, expected_verdict);
        rr_bind_text(ins.get(), 5, actual_verdict);
        rr_bind_double_or_null(ins.get(), 6, ood_energy);
        rr_bind_double_or_null(ins.get(), 7, ood_mahalanobis);
        sqlite3_bind_int(ins.get(), 8, is_match ? 1 : 0);
        rr_bind_text(ins.get(), 9, evidence_dir);
        sqlite3_bind_int64(ins.get(), 10, recorded_at_ms);
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to insert red_rabbit_attempt: ") + sqlite3_errmsg(db_));
        }
        assigned_id = static_cast<int64_t>(sqlite3_last_insert_rowid(db_));
        tx.commit();
    }

    return json{
        {"attempt_id", assigned_id},
        {"session_id", session_id},
        {"step_index", step_index},
        {"sample_uid", sample_uid},
        {"expected_verdict", expected_verdict},
        {"actual_verdict", actual_verdict},
        {"ood_energy", ood_energy.has_value() ? json(*ood_energy) : json(nullptr)},
        {"ood_mahalanobis", ood_mahalanobis.has_value() ? json(*ood_mahalanobis) : json(nullptr)},
        {"is_match", is_match},
        {"evidence_dir", evidence_dir},
        {"recorded_at_ms", recorded_at_ms},
    };
}

json StationDatabase::rr_list_attempts(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    RRStmtGuard stmt(
        db_,
        "SELECT attempt_id, session_id, step_index, sample_uid, expected_verdict, actual_verdict, ood_energy, ood_mahalanobis, is_match, evidence_dir, recorded_at_ms "
        "FROM red_rabbit_attempts WHERE session_id=? ORDER BY step_index ASC, attempt_id ASC;"
    );
    rr_bind_text(stmt.get(), 1, session_id);
    json out = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        out.push_back(json{
            {"attempt_id", static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 0))},
            {"session_id", rr_col_text(stmt.get(), 1)},
            {"step_index", sqlite3_column_int(stmt.get(), 2)},
            {"sample_uid", rr_col_text(stmt.get(), 3)},
            {"expected_verdict", rr_col_text(stmt.get(), 4)},
            {"actual_verdict", rr_col_text(stmt.get(), 5)},
            {"ood_energy", rr_col_double_or_null(stmt.get(), 6)},
            {"ood_mahalanobis", rr_col_double_or_null(stmt.get(), 7)},
            {"is_match", sqlite3_column_int(stmt.get(), 8) != 0},
            {"evidence_dir", rr_col_text(stmt.get(), 9)},
            {"recorded_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 10))},
        });
    }
    return out;
}

std::optional<json> StationDatabase::rr_last_passed_session() const {
    std::lock_guard<std::mutex> lock(mutex_);
    RRStmtGuard stmt(
        db_,
        "SELECT session_id, station_id, trigger_reason, operator_id, lot_id, required_plan, status, started_at_ms, completed_at_ms, model_sha256 "
        "FROM red_rabbit_sessions WHERE status='PASSED' ORDER BY COALESCE(completed_at_ms, started_at_ms) DESC LIMIT 1;"
    );
    if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
        return std::nullopt;
    }
    return read_session_row(stmt.get());
}

json StationDatabase::rr_append_lock_audit(
    const std::string& event_type,
    const std::string& actor_id,
    const std::string& actor_role,
    const std::string& reason_code,
    const std::string& comment,
    const std::optional<std::string>& related_session,
    int64_t created_at_ms
) {
    if (!is_allowed_audit_event_type(event_type)) {
        throw DatabaseHttpError(422, "INVALID_AUDIT_EVENT_TYPE");
    }
    if (!is_allowed_actor_role(actor_role)) {
        throw DatabaseHttpError(422, "INVALID_AUDIT_ACTOR_ROLE");
    }
    const std::string clean_actor = trim_copy(actor_id);
    const std::string clean_reason = trim_copy(reason_code);
    const std::string clean_comment = trim_copy(comment);
    if (clean_actor.empty() || clean_reason.empty()) {
        throw DatabaseHttpError(422, "MISSING_AUDIT_ACTOR_OR_REASON");
    }

    const int64_t ts_ms = (created_at_ms > 0) ? created_at_ms : rr_wall_clock_ms();
    const std::string event_id = "AUD_" + random_token_hex(8);

    std::lock_guard<std::mutex> lock(mutex_);
    RRTransactionGuard tx(db_);

    std::string prev_hash = genesis_hash_64();
    {
        RRStmtGuard prev_stmt(db_, "SELECT row_hash FROM station_lock_audit ORDER BY audit_seq DESC LIMIT 1;");
        if (sqlite3_step(prev_stmt.get()) == SQLITE_ROW) {
            const std::string last_hash = rr_col_text(prev_stmt.get(), 0);
            if (!last_hash.empty()) {
                prev_hash = last_hash;
            }
        }
    }

    const std::string row_hash = compute_audit_row_hash(
        prev_hash,
        event_id,
        event_type,
        clean_actor,
        actor_role,
        clean_reason,
        clean_comment,
        ts_ms
    );

    RRStmtGuard ins(
        db_,
        "INSERT INTO station_lock_audit(event_id, event_type, actor_id, actor_role, reason_code, comment, related_session, created_at_ms, prev_hash, row_hash) "
        "VALUES(?,?,?,?,?,?,?,?,?,?);"
    );
    rr_bind_text(ins.get(), 1, event_id);
    rr_bind_text(ins.get(), 2, event_type);
    rr_bind_text(ins.get(), 3, clean_actor);
    rr_bind_text(ins.get(), 4, actor_role);
    rr_bind_text(ins.get(), 5, clean_reason);
    rr_bind_text(ins.get(), 6, clean_comment);
    rr_bind_text_or_null(ins.get(), 7, related_session);
    sqlite3_bind_int64(ins.get(), 8, ts_ms);
    rr_bind_text(ins.get(), 9, prev_hash);
    rr_bind_text(ins.get(), 10, row_hash);
    if (sqlite3_step(ins.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("Failed to append station_lock_audit: ") + sqlite3_errmsg(db_));
    }

    const int64_t audit_seq = static_cast<int64_t>(sqlite3_last_insert_rowid(db_));
    tx.commit();

    return json{
        {"audit_seq", audit_seq},
        {"event_id", event_id},
        {"event_type", event_type},
        {"actor_id", clean_actor},
        {"actor_role", actor_role},
        {"reason_code", clean_reason},
        {"comment", clean_comment},
        {"related_session", (related_session.has_value() && !related_session->empty()) ? json(*related_session) : json(nullptr)},
        {"created_at_ms", ts_ms},
        {"prev_hash", prev_hash},
        {"row_hash", row_hash},
    };
}

json StationDatabase::rr_list_lock_audit(int limit) const {
    const int safe_limit = std::clamp(limit, 1, 200);
    std::lock_guard<std::mutex> lock(mutex_);
    RRStmtGuard stmt(
        db_,
        "SELECT audit_seq, event_id, event_type, actor_id, actor_role, reason_code, comment, related_session, created_at_ms, prev_hash, row_hash "
        "FROM station_lock_audit ORDER BY audit_seq DESC LIMIT ?;"
    );
    sqlite3_bind_int(stmt.get(), 1, safe_limit);
    json out = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        out.push_back(read_audit_row(stmt.get()));
    }
    return out;
}

json StationDatabase::rr_verify_lock_audit_chain() const {
    std::lock_guard<std::mutex> lock(mutex_);
    RRStmtGuard stmt(
        db_,
        "SELECT audit_seq, event_id, event_type, actor_id, actor_role, reason_code, comment, related_session, created_at_ms, prev_hash, row_hash "
        "FROM station_lock_audit ORDER BY audit_seq ASC;"
    );

    std::string expected_prev = genesis_hash_64();
    std::string latest_hash = expected_prev;
    int checked_rows = 0;
    bool valid = true;
    json broken_seq = nullptr;

    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        ++checked_rows;
        const int64_t seq = static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 0));
        const std::string event_id = rr_col_text(stmt.get(), 1);
        const std::string event_type = rr_col_text(stmt.get(), 2);
        const std::string actor_id = rr_col_text(stmt.get(), 3);
        const std::string actor_role = rr_col_text(stmt.get(), 4);
        const std::string reason_code = rr_col_text(stmt.get(), 5);
        const std::string comment = rr_col_text(stmt.get(), 6);
        const int64_t created_at_ms = static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 8));
        const std::string prev_hash = rr_col_text(stmt.get(), 9);
        const std::string row_hash = rr_col_text(stmt.get(), 10);

        const std::string recomputed = compute_audit_row_hash(
            prev_hash,
            event_id,
            event_type,
            actor_id,
            actor_role,
            reason_code,
            comment,
            created_at_ms
        );

        if (prev_hash != expected_prev || recomputed != row_hash) {
            if (valid) {
                valid = false;
                broken_seq = seq;
            }
        }
        expected_prev = row_hash;
        latest_hash = row_hash;
    }

    return json{
        {"valid", valid},
        {"checked_rows", checked_rows},
        {"broken_seq", broken_seq},
        {"latest_hash", latest_hash},
    };
}

}  // namespace igbt
