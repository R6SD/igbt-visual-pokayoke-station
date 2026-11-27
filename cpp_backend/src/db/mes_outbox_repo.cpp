#include "igbt/database.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace igbt {

namespace {

class MesStmtGuard {
public:
    MesStmtGuard(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
            std::string err = db ? sqlite3_errmsg(db) : "null sqlite3 handle";
            throw std::runtime_error("SQLite prepare failed: " + err);
        }
    }
    ~MesStmtGuard() {
        if (stmt_) sqlite3_finalize(stmt_);
    }
    MesStmtGuard(const MesStmtGuard&) = delete;
    MesStmtGuard& operator=(const MesStmtGuard&) = delete;

    sqlite3_stmt* get() const noexcept { return stmt_; }

private:
    sqlite3_stmt* stmt_{nullptr};
};

class MesTransactionGuard {
public:
    explicit MesTransactionGuard(sqlite3* db) : db_(db) {
        char* errmsg = nullptr;
        if (sqlite3_exec(db_, "BEGIN IMMEDIATE TRANSACTION;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::string msg = errmsg ? errmsg : "BEGIN IMMEDIATE failed";
            if (errmsg) sqlite3_free(errmsg);
            throw std::runtime_error(msg);
        }
    }
    ~MesTransactionGuard() {
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

void mes_bind_text(sqlite3_stmt* stmt, int idx, const std::string& val) {
    sqlite3_bind_text(stmt, idx, val.c_str(), static_cast<int>(val.size()), SQLITE_TRANSIENT);
}

void mes_bind_text_or_null(sqlite3_stmt* stmt, int idx, const std::optional<std::string>& val) {
    if (val.has_value() && !val->empty()) {
        mes_bind_text(stmt, idx, *val);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

void mes_bind_int64_or_null(sqlite3_stmt* stmt, int idx, const std::optional<int64_t>& val) {
    if (val.has_value()) {
        sqlite3_bind_int64(stmt, idx, *val);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

std::string mes_col_text(sqlite3_stmt* stmt, int idx) {
    const unsigned char* txt = sqlite3_column_text(stmt, idx);
    if (!txt) return {};
    int n = sqlite3_column_bytes(stmt, idx);
    return std::string(reinterpret_cast<const char*>(txt), static_cast<std::size_t>(std::max(0, n)));
}

json mes_col_text_or_null(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return mes_col_text(stmt, idx);
}

json mes_col_int_or_null(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return sqlite3_column_int(stmt, idx);
}

json mes_col_int64_or_null(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return static_cast<int64_t>(sqlite3_column_int64(stmt, idx));
}

int64_t mes_wall_clock_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

bool is_allowed_outbox_status(const std::string& status) {
    return status == "PENDING" || status == "IN_FLIGHT" || status == "ACKED" || status == "DEAD_LETTER";
}

std::string normalize_lot_status(const std::string& status) {
    if (status == "HOLD" || status == "SCRAP" || status == "ACTIVE") return status;
    return "ACTIVE";
}

json read_outbox_row(sqlite3_stmt* stmt) {
    json payload = json::object();
    const std::string payload_raw = mes_col_text(stmt, 7);
    try {
        json parsed = json::parse(payload_raw);
        if (parsed.is_object()) payload = std::move(parsed);
    } catch (...) {
    }
    return json{
        {"id", static_cast<int64_t>(sqlite3_column_int64(stmt, 0))},
        {"idempotency_key", mes_col_text(stmt, 1)},
        {"cycle_id", mes_col_text(stmt, 2)},
        {"work_order_id", mes_col_text(stmt, 3)},
        {"lot_id", mes_col_text(stmt, 4)},
        {"tube_seq", static_cast<int64_t>(sqlite3_column_int64(stmt, 5))},
        {"endpoint_path", mes_col_text(stmt, 6)},
        {"payload", std::move(payload)},
        {"payload_json", payload_raw},
        {"status", mes_col_text(stmt, 8)},
        {"retry_count", sqlite3_column_int(stmt, 9)},
        {"next_retry_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 10))},
        {"last_http_code", mes_col_int_or_null(stmt, 11)},
        {"last_error", mes_col_text_or_null(stmt, 12)},
        {"response_body", mes_col_text_or_null(stmt, 13)},
        {"created_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 14))},
        {"acked_at_ms", mes_col_int64_or_null(stmt, 15)},
    };
}

json read_active_lot_row(sqlite3_stmt* stmt) {
    return json{
        {"work_order_id", mes_col_text(stmt, 0)},
        {"lot_id", mes_col_text(stmt, 1)},
        {"part_number", mes_col_text(stmt, 2)},
        {"expected_pipe_color", mes_col_text(stmt, 3)},
        {"lot_status", mes_col_text(stmt, 4)},
        {"signature_sha256", mes_col_text(stmt, 5)},
        {"validated_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 6))},
        {"expires_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 7))},
        {"tube_seq", static_cast<int64_t>(sqlite3_column_int64(stmt, 8))},
        {"offline_used_count", sqlite3_column_int(stmt, 9)},
        {"offline_started_at_ms", mes_col_int64_or_null(stmt, 10)},
        {"updated_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt, 11))},
    };
}

json default_runtime_policy_json() {
    return json{
        {"whitelist_approved", false},
        {"whitelist_actor_id", ""},
        {"whitelist_actor_role", ""},
        {"whitelist_reason", ""},
        {"whitelist_max_tubes", 50},
        {"whitelist_max_duration_ms", 1800000},
        {"whitelist_approved_at_ms", 0},
        {"recall_locked", false},
        {"recall_status", ""},
        {"recall_lot_id", ""},
        {"recall_work_order_id", ""},
        {"recall_reason", ""},
        {"recall_affected_keys", json::array()},
        {"recall_triggered_at_ms", 0},
        {"updated_at_ms", 0},
    };
}

}  // namespace

void StationDatabase::mes_insert_outbox_unlocked(const MesOutboxEnqueueRequest& req) {
    if (req.idempotency_key.empty()) {
        throw std::invalid_argument("mes_insert_outbox_unlocked: idempotency_key cannot be empty");
    }
    const int64_t ts_ms = (req.created_at_ms > 0) ? req.created_at_ms : mes_wall_clock_ms();
    const std::string endpoint = req.endpoint_path.empty() ? "/api/mes/v1/pass_records" : req.endpoint_path;
    const std::string payload_str = req.payload.dump();

    MesStmtGuard ins(
        db_,
        "INSERT OR IGNORE INTO mes_outbox_queue("
        "idempotency_key, cycle_id, work_order_id, lot_id, tube_seq, "
        "endpoint_path, payload_json, status, retry_count, next_retry_at_ms, created_at_ms"
        ") VALUES(?,?,?,?,?,?,?,'PENDING',0,?,?);"
    );
    mes_bind_text(ins.get(), 1, req.idempotency_key);
    mes_bind_text(ins.get(), 2, req.cycle_id);
    mes_bind_text(ins.get(), 3, req.work_order_id);
    mes_bind_text(ins.get(), 4, req.lot_id);
    sqlite3_bind_int64(ins.get(), 5, req.tube_seq);
    mes_bind_text(ins.get(), 6, endpoint);
    mes_bind_text(ins.get(), 7, payload_str);
    sqlite3_bind_int64(ins.get(), 8, ts_ms);
    sqlite3_bind_int64(ins.get(), 9, ts_ms);
    if (sqlite3_step(ins.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("Failed to insert mes_outbox_queue: ") + sqlite3_errmsg(db_));
    }

    if (sqlite3_changes(db_) > 0 && !req.lot_id.empty()) {
        if (req.increment_offline_used) {
            const int64_t offline_anchor = req.offline_started_at_ms.value_or(ts_ms);
            MesStmtGuard upd(
                db_,
                "UPDATE mes_active_lot "
                "SET tube_seq = MAX(tube_seq, ?), "
                "    offline_used_count = offline_used_count + 1, "
                "    offline_started_at_ms = COALESCE(offline_started_at_ms, ?), "
                "    updated_at_ms = ? "
                "WHERE singleton_id=1 AND lot_id=?;"
            );
            sqlite3_bind_int64(upd.get(), 1, req.tube_seq);
            sqlite3_bind_int64(upd.get(), 2, offline_anchor);
            sqlite3_bind_int64(upd.get(), 3, ts_ms);
            mes_bind_text(upd.get(), 4, req.lot_id);
            sqlite3_step(upd.get());
        } else {
            MesStmtGuard upd(
                db_,
                "UPDATE mes_active_lot "
                "SET tube_seq = MAX(tube_seq, ?), "
                "    updated_at_ms = ? "
                "WHERE singleton_id=1 AND lot_id=?;"
            );
            sqlite3_bind_int64(upd.get(), 1, req.tube_seq);
            sqlite3_bind_int64(upd.get(), 2, ts_ms);
            mes_bind_text(upd.get(), 3, req.lot_id);
            sqlite3_step(upd.get());
        }
    }
}

json StationDatabase::mes_enqueue_outbox(const MesOutboxEnqueueRequest& req) {
    std::lock_guard<std::mutex> lock(mutex_);
    {
        MesTransactionGuard tx(db_);
        mes_insert_outbox_unlocked(req);
        tx.commit();
    }

    MesStmtGuard sel(
        db_,
        "SELECT id, idempotency_key, cycle_id, work_order_id, lot_id, tube_seq, endpoint_path, "
        "payload_json, status, retry_count, next_retry_at_ms, last_http_code, last_error, "
        "response_body, created_at_ms, acked_at_ms "
        "FROM mes_outbox_queue WHERE idempotency_key=?;"
    );
    mes_bind_text(sel.get(), 1, req.idempotency_key);
    if (sqlite3_step(sel.get()) != SQLITE_ROW) {
        throw std::runtime_error("Enqueued MES outbox row missing on readback");
    }
    return read_outbox_row(sel.get());
}

std::optional<json> StationDatabase::mes_claim_next_pending(int64_t now_ms, int64_t stale_inflight_ms) {
    const int64_t lease_until = now_ms + std::max<int64_t>(1000, stale_inflight_ms);
    std::lock_guard<std::mutex> lock(mutex_);
    MesTransactionGuard tx(db_);

    std::optional<json> row;
    {
        MesStmtGuard sel(
            db_,
            "SELECT id, idempotency_key, cycle_id, work_order_id, lot_id, tube_seq, endpoint_path, "
            "payload_json, status, retry_count, next_retry_at_ms, last_http_code, last_error, "
            "response_body, created_at_ms, acked_at_ms "
            "FROM mes_outbox_queue "
            "WHERE (status='PENDING' AND next_retry_at_ms <= ?) "
            "   OR (status='IN_FLIGHT' AND next_retry_at_ms <= ?) "
            "ORDER BY id ASC LIMIT 1;"
        );
        sqlite3_bind_int64(sel.get(), 1, now_ms);
        sqlite3_bind_int64(sel.get(), 2, now_ms);
        if (sqlite3_step(sel.get()) == SQLITE_ROW) {
            row = read_outbox_row(sel.get());
        }
    }
    if (!row.has_value()) {
        return std::nullopt;
    }

    const int64_t row_id = (*row)["id"].get<int64_t>();
    {
        MesStmtGuard upd(
            db_,
            "UPDATE mes_outbox_queue SET status='IN_FLIGHT', next_retry_at_ms=? WHERE id=?;"
        );
        sqlite3_bind_int64(upd.get(), 1, lease_until);
        sqlite3_bind_int64(upd.get(), 2, row_id);
        if (sqlite3_step(upd.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to claim MES outbox row: ") + sqlite3_errmsg(db_));
        }
    }
    tx.commit();

    (*row)["status"] = "IN_FLIGHT";
    (*row)["next_retry_at_ms"] = lease_until;
    return row;
}

void StationDatabase::mes_mark_acked(
    int64_t row_id,
    int http_code,
    const std::string& response_body,
    int64_t acked_at_ms
) {
    const int64_t ts_ms = (acked_at_ms > 0) ? acked_at_ms : mes_wall_clock_ms();
    std::lock_guard<std::mutex> lock(mutex_);
    MesTransactionGuard tx(db_);
    MesStmtGuard upd(
        db_,
        "UPDATE mes_outbox_queue "
        "SET status='ACKED', last_http_code=?, last_error=NULL, response_body=?, acked_at_ms=? "
        "WHERE id=?;"
    );
    sqlite3_bind_int(upd.get(), 1, http_code);
    mes_bind_text_or_null(upd.get(), 2, response_body.empty() ? std::nullopt : std::optional<std::string>(response_body));
    sqlite3_bind_int64(upd.get(), 3, ts_ms);
    sqlite3_bind_int64(upd.get(), 4, row_id);
    if (sqlite3_step(upd.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("Failed to mark MES outbox ACKED: ") + sqlite3_errmsg(db_));
    }
    tx.commit();
}

void StationDatabase::mes_mark_retry(
    int64_t row_id,
    int http_code,
    const std::string& error_msg,
    int64_t next_retry_at_ms,
    bool dead_letter
) {
    std::lock_guard<std::mutex> lock(mutex_);
    MesTransactionGuard tx(db_);
    MesStmtGuard upd(
        db_,
        "UPDATE mes_outbox_queue "
        "SET status=?, retry_count=retry_count+1, next_retry_at_ms=?, last_http_code=?, last_error=? "
        "WHERE id=?;"
    );
    mes_bind_text(upd.get(), 1, dead_letter ? "DEAD_LETTER" : "PENDING");
    sqlite3_bind_int64(upd.get(), 2, next_retry_at_ms);
    if (http_code > 0) {
        sqlite3_bind_int(upd.get(), 3, http_code);
    } else {
        sqlite3_bind_null(upd.get(), 3);
    }
    mes_bind_text_or_null(upd.get(), 4, error_msg.empty() ? std::nullopt : std::optional<std::string>(error_msg));
    sqlite3_bind_int64(upd.get(), 5, row_id);
    if (sqlite3_step(upd.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("Failed to mark MES outbox retry: ") + sqlite3_errmsg(db_));
    }
    tx.commit();
}

json StationDatabase::mes_requeue_dead_letter(int64_t row_id, int64_t now_ms) {
    const int64_t ts_ms = (now_ms > 0) ? now_ms : mes_wall_clock_ms();
    std::lock_guard<std::mutex> lock(mutex_);
    MesTransactionGuard tx(db_);

    std::string cur_status;
    {
        MesStmtGuard chk(db_, "SELECT status FROM mes_outbox_queue WHERE id=?;");
        sqlite3_bind_int64(chk.get(), 1, row_id);
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            throw DatabaseHttpError(404, "OUTBOX_ROW_NOT_FOUND");
        }
        cur_status = mes_col_text(chk.get(), 0);
    }
    if (cur_status != "DEAD_LETTER") {
        throw DatabaseHttpError(409, "OUTBOX_ROW_NOT_DEAD_LETTER");
    }

    {
        MesStmtGuard upd(
            db_,
            "UPDATE mes_outbox_queue "
            "SET status='PENDING', retry_count=0, next_retry_at_ms=?, last_error=NULL "
            "WHERE id=?;"
        );
        sqlite3_bind_int64(upd.get(), 1, ts_ms);
        sqlite3_bind_int64(upd.get(), 2, row_id);
        if (sqlite3_step(upd.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to requeue MES outbox row: ") + sqlite3_errmsg(db_));
        }
    }
    tx.commit();

    MesStmtGuard sel(
        db_,
        "SELECT id, idempotency_key, cycle_id, work_order_id, lot_id, tube_seq, endpoint_path, "
        "payload_json, status, retry_count, next_retry_at_ms, last_http_code, last_error, "
        "response_body, created_at_ms, acked_at_ms "
        "FROM mes_outbox_queue WHERE id=?;"
    );
    sqlite3_bind_int64(sel.get(), 1, row_id);
    if (sqlite3_step(sel.get()) != SQLITE_ROW) {
        throw DatabaseHttpError(404, "OUTBOX_ROW_NOT_FOUND");
    }
    return read_outbox_row(sel.get());
}

json StationDatabase::mes_list_outbox(int limit, const std::string& status_filter) const {
    const int safe_limit = std::clamp(limit, 1, 200);
    const bool has_filter = is_allowed_outbox_status(status_filter);
    std::lock_guard<std::mutex> lock(mutex_);
    const char* sql = has_filter
        ? "SELECT id, idempotency_key, cycle_id, work_order_id, lot_id, tube_seq, endpoint_path, "
          "payload_json, status, retry_count, next_retry_at_ms, last_http_code, last_error, "
          "response_body, created_at_ms, acked_at_ms "
          "FROM mes_outbox_queue WHERE status=? ORDER BY id DESC LIMIT ?;"
        : "SELECT id, idempotency_key, cycle_id, work_order_id, lot_id, tube_seq, endpoint_path, "
          "payload_json, status, retry_count, next_retry_at_ms, last_http_code, last_error, "
          "response_body, created_at_ms, acked_at_ms "
          "FROM mes_outbox_queue ORDER BY id DESC LIMIT ?;";
    MesStmtGuard stmt(db_, sql);
    if (has_filter) {
        mes_bind_text(stmt.get(), 1, status_filter);
        sqlite3_bind_int(stmt.get(), 2, safe_limit);
    } else {
        sqlite3_bind_int(stmt.get(), 1, safe_limit);
    }
    json out = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        out.push_back(read_outbox_row(stmt.get()));
    }
    return out;
}

json StationDatabase::mes_outbox_counts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int pending = 0;
    int in_flight = 0;
    int acked = 0;
    int dead_letter = 0;
    MesStmtGuard stmt(db_, "SELECT status, COUNT(*) FROM mes_outbox_queue GROUP BY status;");
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        const std::string st = mes_col_text(stmt.get(), 0);
        const int cnt = sqlite3_column_int(stmt.get(), 1);
        if (st == "PENDING") pending = cnt;
        else if (st == "IN_FLIGHT") in_flight = cnt;
        else if (st == "ACKED") acked = cnt;
        else if (st == "DEAD_LETTER") dead_letter = cnt;
    }
    return json{
        {"pending", pending},
        {"in_flight", in_flight},
        {"acked", acked},
        {"dead_letter", dead_letter},
        {"total", pending + in_flight + acked + dead_letter},
    };
}

std::vector<std::string> StationDatabase::mes_list_lot_outbox_keys(const std::string& lot_id, int limit) const {
    std::vector<std::string> keys;
    if (lot_id.empty()) return keys;
    const int safe_limit = std::clamp(limit, 1, 500);
    std::lock_guard<std::mutex> lock(mutex_);
    MesStmtGuard stmt(
        db_,
        "SELECT idempotency_key FROM mes_outbox_queue WHERE lot_id=? ORDER BY id DESC LIMIT ?;"
    );
    mes_bind_text(stmt.get(), 1, lot_id);
    sqlite3_bind_int(stmt.get(), 2, safe_limit);
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        keys.push_back(mes_col_text(stmt.get(), 0));
    }
    return keys;
}

std::optional<json> StationDatabase::mes_get_active_lot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MesStmtGuard stmt(
        db_,
        "SELECT work_order_id, lot_id, part_number, expected_pipe_color, lot_status, signature_sha256, "
        "validated_at_ms, expires_at_ms, tube_seq, offline_used_count, offline_started_at_ms, updated_at_ms "
        "FROM mes_active_lot WHERE singleton_id=1;"
    );
    if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
        return std::nullopt;
    }
    return read_active_lot_row(stmt.get());
}

json StationDatabase::mes_save_active_lot(const json& lot_obj) {
    if (!lot_obj.is_object()) {
        throw std::invalid_argument("mes_save_active_lot expects object");
    }
    const std::string work_order_id = lot_obj.value("work_order_id", "");
    const std::string lot_id = lot_obj.value("lot_id", "");
    const std::string part_number = lot_obj.value("part_number", "");
    const std::string expected_pipe_color = lot_obj.value("expected_pipe_color", "");
    const std::string lot_status = normalize_lot_status(lot_obj.value("lot_status", "ACTIVE"));
    const std::string signature_sha256 = lot_obj.value("signature_sha256", "");
    const int64_t validated_at_ms = lot_obj.value("validated_at_ms", mes_wall_clock_ms());
    const int64_t expires_at_ms = lot_obj.value("expires_at_ms", validated_at_ms + 24LL * 3600 * 1000);
    const int64_t tube_seq = lot_obj.value("tube_seq", int64_t{0});
    const int offline_used_count = lot_obj.value("offline_used_count", 0);
    std::optional<int64_t> offline_started_at_ms;
    if (lot_obj.contains("offline_started_at_ms") && lot_obj["offline_started_at_ms"].is_number_integer()) {
        offline_started_at_ms = lot_obj["offline_started_at_ms"].get<int64_t>();
    }
    const int64_t updated_at_ms = lot_obj.value("updated_at_ms", mes_wall_clock_ms());

    {
        std::lock_guard<std::mutex> lock(mutex_);
        MesTransactionGuard tx(db_);
        MesStmtGuard ins(
            db_,
            "INSERT INTO mes_active_lot("
            "singleton_id, work_order_id, lot_id, part_number, expected_pipe_color, lot_status, "
            "signature_sha256, validated_at_ms, expires_at_ms, tube_seq, offline_used_count, "
            "offline_started_at_ms, updated_at_ms"
            ") VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(singleton_id) DO UPDATE SET "
            "work_order_id=excluded.work_order_id, "
            "lot_id=excluded.lot_id, "
            "part_number=excluded.part_number, "
            "expected_pipe_color=excluded.expected_pipe_color, "
            "lot_status=excluded.lot_status, "
            "signature_sha256=excluded.signature_sha256, "
            "validated_at_ms=excluded.validated_at_ms, "
            "expires_at_ms=excluded.expires_at_ms, "
            "tube_seq=excluded.tube_seq, "
            "offline_used_count=excluded.offline_used_count, "
            "offline_started_at_ms=excluded.offline_started_at_ms, "
            "updated_at_ms=excluded.updated_at_ms;"
        );
        mes_bind_text(ins.get(), 1, work_order_id);
        mes_bind_text(ins.get(), 2, lot_id);
        mes_bind_text(ins.get(), 3, part_number);
        mes_bind_text(ins.get(), 4, expected_pipe_color);
        mes_bind_text(ins.get(), 5, lot_status);
        mes_bind_text(ins.get(), 6, signature_sha256);
        sqlite3_bind_int64(ins.get(), 7, validated_at_ms);
        sqlite3_bind_int64(ins.get(), 8, expires_at_ms);
        sqlite3_bind_int64(ins.get(), 9, tube_seq);
        sqlite3_bind_int(ins.get(), 10, offline_used_count);
        mes_bind_int64_or_null(ins.get(), 11, offline_started_at_ms);
        sqlite3_bind_int64(ins.get(), 12, updated_at_ms);
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to save mes_active_lot: ") + sqlite3_errmsg(db_));
        }
        tx.commit();
    }

    auto saved = mes_get_active_lot();
    return saved.has_value() ? *saved : lot_obj;
}

void StationDatabase::mes_update_active_lot_status(const std::string& lot_status, int64_t now_ms) {
    const std::string normalized = normalize_lot_status(lot_status);
    const int64_t ts_ms = (now_ms > 0) ? now_ms : mes_wall_clock_ms();
    std::lock_guard<std::mutex> lock(mutex_);
    MesTransactionGuard tx(db_);
    MesStmtGuard upd(db_, "UPDATE mes_active_lot SET lot_status=?, updated_at_ms=? WHERE singleton_id=1;");
    mes_bind_text(upd.get(), 1, normalized);
    sqlite3_bind_int64(upd.get(), 2, ts_ms);
    sqlite3_step(upd.get());
    tx.commit();
}

void StationDatabase::mes_reset_offline_usage(int64_t now_ms) {
    const int64_t ts_ms = (now_ms > 0) ? now_ms : mes_wall_clock_ms();
    std::lock_guard<std::mutex> lock(mutex_);
    MesTransactionGuard tx(db_);
    MesStmtGuard upd(
        db_,
        "UPDATE mes_active_lot "
        "SET offline_used_count=0, offline_started_at_ms=NULL, updated_at_ms=? "
        "WHERE singleton_id=1 AND (offline_used_count != 0 OR offline_started_at_ms IS NOT NULL);"
    );
    sqlite3_bind_int64(upd.get(), 1, ts_ms);
    sqlite3_step(upd.get());
    tx.commit();
}

int64_t StationDatabase::mes_peek_next_tube_seq() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t lot_seq = 0;
    {
        MesStmtGuard s1(db_, "SELECT COALESCE((SELECT tube_seq FROM mes_active_lot WHERE singleton_id=1), 0);");
        if (sqlite3_step(s1.get()) == SQLITE_ROW) {
            lot_seq = static_cast<int64_t>(sqlite3_column_int64(s1.get(), 0));
        }
    }
    int64_t max_q = 0;
    {
        MesStmtGuard s2(db_, "SELECT COALESCE((SELECT MAX(tube_seq) FROM mes_outbox_queue), 0);");
        if (sqlite3_step(s2.get()) == SQLITE_ROW) {
            max_q = static_cast<int64_t>(sqlite3_column_int64(s2.get(), 0));
        }
    }
    return std::max(lot_seq, max_q) + 1;
}

json StationDatabase::mes_get_runtime_policy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MesStmtGuard stmt(
        db_,
        "SELECT whitelist_approved, whitelist_actor_id, whitelist_actor_role, whitelist_reason, "
        "whitelist_max_tubes, whitelist_max_duration_ms, whitelist_approved_at_ms, "
        "recall_locked, recall_status, recall_lot_id, recall_work_order_id, recall_reason, "
        "recall_affected_keys, recall_triggered_at_ms, updated_at_ms "
        "FROM mes_runtime_policy WHERE singleton_id=1;"
    );
    if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
        return default_runtime_policy_json();
    }
    // Note: SQL has 15 columns (indices 0..14). Read directly cleanly.
    json affected_keys = json::array();
    try {
        json parsed = json::parse(mes_col_text(stmt.get(), 12));
        if (parsed.is_array()) affected_keys = std::move(parsed);
    } catch (...) {
    }
    return json{
        {"whitelist_approved", sqlite3_column_int(stmt.get(), 0) != 0},
        {"whitelist_actor_id", mes_col_text(stmt.get(), 1)},
        {"whitelist_actor_role", mes_col_text(stmt.get(), 2)},
        {"whitelist_reason", mes_col_text(stmt.get(), 3)},
        {"whitelist_max_tubes", sqlite3_column_int(stmt.get(), 4)},
        {"whitelist_max_duration_ms", static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 5))},
        {"whitelist_approved_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 6))},
        {"recall_locked", sqlite3_column_int(stmt.get(), 7) != 0},
        {"recall_status", mes_col_text(stmt.get(), 8)},
        {"recall_lot_id", mes_col_text(stmt.get(), 9)},
        {"recall_work_order_id", mes_col_text(stmt.get(), 10)},
        {"recall_reason", mes_col_text(stmt.get(), 11)},
        {"recall_affected_keys", std::move(affected_keys)},
        {"recall_triggered_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 13))},
        {"updated_at_ms", static_cast<int64_t>(sqlite3_column_int64(stmt.get(), 14))},
    };
}

json StationDatabase::mes_save_runtime_policy(const json& policy_obj) {
    if (!policy_obj.is_object()) {
        throw std::invalid_argument("mes_save_runtime_policy expects object");
    }
    const bool wl_approved = policy_obj.value("whitelist_approved", false);
    const std::string wl_actor = policy_obj.value("whitelist_actor_id", "");
    const std::string wl_role = policy_obj.value("whitelist_actor_role", "");
    const std::string wl_reason = policy_obj.value("whitelist_reason", "");
    const int wl_tubes = std::clamp(policy_obj.value("whitelist_max_tubes", 50), 1, 50);
    const int64_t wl_dur_ms = std::clamp<int64_t>(policy_obj.value("whitelist_max_duration_ms", int64_t{1800000}), 60000, 1800000);
    const int64_t wl_at_ms = policy_obj.value("whitelist_approved_at_ms", int64_t{0});

    const bool rc_locked = policy_obj.value("recall_locked", false);
    const std::string rc_status = policy_obj.value("recall_status", "");
    const std::string rc_lot = policy_obj.value("recall_lot_id", "");
    const std::string rc_wo = policy_obj.value("recall_work_order_id", "");
    const std::string rc_reason = policy_obj.value("recall_reason", "");
    const json rc_keys = (policy_obj.contains("recall_affected_keys") && policy_obj["recall_affected_keys"].is_array())
                             ? policy_obj["recall_affected_keys"]
                             : json::array();
    const int64_t rc_at_ms = policy_obj.value("recall_triggered_at_ms", int64_t{0});
    const int64_t upd_ms = policy_obj.value("updated_at_ms", mes_wall_clock_ms());

    {
        std::lock_guard<std::mutex> lock(mutex_);
        MesTransactionGuard tx(db_);
        MesStmtGuard ins(
            db_,
            "INSERT INTO mes_runtime_policy("
            "singleton_id, whitelist_approved, whitelist_actor_id, whitelist_actor_role, whitelist_reason, "
            "whitelist_max_tubes, whitelist_max_duration_ms, whitelist_approved_at_ms, "
            "recall_locked, recall_status, recall_lot_id, recall_work_order_id, recall_reason, "
            "recall_affected_keys, recall_triggered_at_ms, updated_at_ms"
            ") VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(singleton_id) DO UPDATE SET "
            "whitelist_approved=excluded.whitelist_approved, "
            "whitelist_actor_id=excluded.whitelist_actor_id, "
            "whitelist_actor_role=excluded.whitelist_actor_role, "
            "whitelist_reason=excluded.whitelist_reason, "
            "whitelist_max_tubes=excluded.whitelist_max_tubes, "
            "whitelist_max_duration_ms=excluded.whitelist_max_duration_ms, "
            "whitelist_approved_at_ms=excluded.whitelist_approved_at_ms, "
            "recall_locked=excluded.recall_locked, "
            "recall_status=excluded.recall_status, "
            "recall_lot_id=excluded.recall_lot_id, "
            "recall_work_order_id=excluded.recall_work_order_id, "
            "recall_reason=excluded.recall_reason, "
            "recall_affected_keys=excluded.recall_affected_keys, "
            "recall_triggered_at_ms=excluded.recall_triggered_at_ms, "
            "updated_at_ms=excluded.updated_at_ms;"
        );
        sqlite3_bind_int(ins.get(), 1, wl_approved ? 1 : 0);
        mes_bind_text(ins.get(), 2, wl_actor);
        mes_bind_text(ins.get(), 3, wl_role);
        mes_bind_text(ins.get(), 4, wl_reason);
        sqlite3_bind_int(ins.get(), 5, wl_tubes);
        sqlite3_bind_int64(ins.get(), 6, wl_dur_ms);
        sqlite3_bind_int64(ins.get(), 7, wl_at_ms);
        sqlite3_bind_int(ins.get(), 8, rc_locked ? 1 : 0);
        mes_bind_text(ins.get(), 9, rc_status);
        mes_bind_text(ins.get(), 10, rc_lot);
        mes_bind_text(ins.get(), 11, rc_wo);
        mes_bind_text(ins.get(), 12, rc_reason);
        mes_bind_text(ins.get(), 13, rc_keys.dump());
        sqlite3_bind_int64(ins.get(), 14, rc_at_ms);
        sqlite3_bind_int64(ins.get(), 15, upd_ms);
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to save mes_runtime_policy: ") + sqlite3_errmsg(db_));
        }
        tx.commit();
    }

    return mes_get_runtime_policy();
}

}  // namespace igbt
