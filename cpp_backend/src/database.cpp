#include "igbt/database.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <regex>
#include <set>
#include <sstream>

namespace igbt {

namespace {

class StmtGuard {
public:
    StmtGuard(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
            std::string err = db ? sqlite3_errmsg(db) : "null sqlite3 handle";
            throw std::runtime_error("SQLite prepare failed: " + err);
        }
    }
    ~StmtGuard() {
        if (stmt_) sqlite3_finalize(stmt_);
    }
    StmtGuard(const StmtGuard&) = delete;
    StmtGuard& operator=(const StmtGuard&) = delete;

    sqlite3_stmt* get() const noexcept { return stmt_; }

private:
    sqlite3_stmt* stmt_{nullptr};
};

class TransactionGuard {
public:
    explicit TransactionGuard(sqlite3* db) : db_(db) {
        char* errmsg = nullptr;
        if (sqlite3_exec(db_, "BEGIN IMMEDIATE TRANSACTION;", nullptr, nullptr, &errmsg) != SQLITE_OK) {
            std::string msg = errmsg ? errmsg : "BEGIN IMMEDIATE failed";
            if (errmsg) sqlite3_free(errmsg);
            throw std::runtime_error(msg);
        }
    }
    ~TransactionGuard() {
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

// Exclusive file creation ('xb' semantics via CREATE_NEW) that cleans up on failure unless committed
class ExclusiveEvidenceFileGuard {
public:
    ExclusiveEvidenceFileGuard(
        const std::filesystem::path& evidence_root,
        const std::filesystem::path& target_path,
        const void* data,
        std::size_t size
    ) : path_(target_path) {
        std::error_code ec;
        std::filesystem::create_directories(evidence_root, ec);
        if (target_path.has_parent_path()) {
            std::filesystem::create_directories(target_path.parent_path(), ec);
        }
        if (!is_path_within(evidence_root, target_path)) {
            throw DatabaseHttpError(400, "INVALID_EVIDENCE_PATH");
        }

        const std::wstring wpath = target_path.wstring();
        HANDLE hFile = CreateFileW(
            wpath.c_str(),
            GENERIC_WRITE,
            0,
            nullptr,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (hFile == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            if (err == ERROR_FILE_EXISTS || err == ERROR_ALREADY_EXISTS) {
                throw DatabaseHttpError(409, "EVIDENCE_NAME_COLLISION_RETRY");
            }
            throw std::runtime_error("CreateFileW failed for evidence file, error=" + std::to_string(err));
        }

        created_ = true;
        const auto* ptr = static_cast<const uint8_t*>(data);
        std::size_t remaining = size;
        while (remaining > 0) {
            DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(remaining, 1 << 20));
            DWORD written = 0;
            BOOL ok = WriteFile(hFile, ptr, chunk, &written, nullptr);
            if (!ok || written == 0) {
                CloseHandle(hFile);
                cleanup();
                throw std::runtime_error("WriteFile failed for evidence file");
            }
            ptr += written;
            remaining -= written;
        }
        FlushFileBuffers(hFile);
        CloseHandle(hFile);
    }

    ~ExclusiveEvidenceFileGuard() {
        if (created_ && !committed_) {
            cleanup();
        }
    }

    void commit() noexcept { committed_ = true; }

private:
    void cleanup() noexcept {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    std::filesystem::path path_;
    bool created_{false};
    bool committed_{false};
};

void bind_text_or_null(sqlite3_stmt* stmt, int idx, const std::optional<std::string>& val) {
    if (val.has_value()) {
        sqlite3_bind_text(stmt, idx, val->c_str(), static_cast<int>(val->size()), SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

void bind_text(sqlite3_stmt* stmt, int idx, const std::string& val) {
    sqlite3_bind_text(stmt, idx, val.c_str(), static_cast<int>(val.size()), SQLITE_TRANSIENT);
}

void bind_double_or_null(sqlite3_stmt* stmt, int idx, const std::optional<double>& val) {
    if (val.has_value() && std::isfinite(*val)) {
        sqlite3_bind_double(stmt, idx, *val);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

void bind_int64_or_null(sqlite3_stmt* stmt, int idx, const std::optional<int64_t>& val) {
    if (val.has_value()) {
        sqlite3_bind_int64(stmt, idx, *val);
    } else {
        sqlite3_bind_null(stmt, idx);
    }
}

std::string col_text(sqlite3_stmt* stmt, int idx) {
    const unsigned char* txt = sqlite3_column_text(stmt, idx);
    if (!txt) return {};
    int n = sqlite3_column_bytes(stmt, idx);
    return std::string(reinterpret_cast<const char*>(txt), static_cast<std::size_t>(std::max(0, n)));
}

json col_text_or_null_json(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return col_text(stmt, idx);
}

json col_double_or_null_json(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return sqlite3_column_double(stmt, idx);
}

json col_int64_or_null_json(sqlite3_stmt* stmt, int idx) {
    if (sqlite3_column_type(stmt, idx) == SQLITE_NULL) return nullptr;
    return static_cast<int64_t>(sqlite3_column_int64(stmt, idx));
}

std::optional<std::string> opt_json_string(const json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) return std::nullopt;
    if (obj[key].is_string()) return obj[key].get<std::string>();
    if (obj[key].is_number_integer()) return std::to_string(obj[key].get<int64_t>());
    return std::nullopt;
}

std::optional<double> opt_json_double(const json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null() || obj[key].is_boolean()) {
        return std::nullopt;
    }
    if (obj[key].is_number()) {
        double v = obj[key].get<double>();
        if (std::isfinite(v)) return v;
    }
    return std::nullopt;
}

std::optional<int64_t> opt_json_int64(const json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null() || obj[key].is_boolean()) {
        return std::nullopt;
    }
    if (obj[key].is_number_integer()) {
        return obj[key].get<int64_t>();
    }
    return std::nullopt;
}

std::string utc_iso_offset_ms(int64_t delta_ms) {
    using namespace std::chrono;
    auto tp = system_clock::now() + milliseconds(delta_ms);
    auto total_ms = duration_cast<milliseconds>(tp.time_since_epoch()).count();
    std::time_t secs = static_cast<std::time_t>(total_ms / 1000);
    int64_t rem_ms = total_ms % 1000;
    if (rem_ms < 0) {
        rem_ms += 1000;
        --secs;
    }
    std::tm gmt{};
    gmtime_s(&gmt, &secs);
    std::ostringstream oss;
    oss << std::put_time(&gmt, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setw(3) << std::setfill('0') << rem_ms << "Z";
    return oss.str();
}

// Parse ISO-8601 UTC timestamp to epoch milliseconds for monotonic/UTC ordering checks
std::optional<int64_t> parse_iso_utc_epoch_ms(const std::string& iso) {
    if (iso.size() < 19) return std::nullopt;
    int y = 0, m = 0, d = 0, hh = 0, mm = 0, ss = 0;
    if (std::sscanf(iso.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &m, &d, &hh, &mm, &ss) != 6) {
        return std::nullopt;
    }
    std::tm tm_val{};
    tm_val.tm_year = y - 1900;
    tm_val.tm_mon = m - 1;
    tm_val.tm_mday = d;
    tm_val.tm_hour = hh;
    tm_val.tm_min = mm;
    tm_val.tm_sec = ss;
    std::time_t epoch_s = _mkgmtime(&tm_val);
    if (epoch_s < 0) return std::nullopt;

    int ms = 0;
    std::size_t dot = iso.find('.', 18);
    if (dot != std::string::npos) {
        std::size_t i = dot + 1;
        int digits = 0;
        while (i < iso.size() && std::isdigit(static_cast<unsigned char>(iso[i])) && digits < 3) {
            ms = ms * 10 + (iso[i] - '0');
            ++i;
            ++digits;
        }
        while (digits > 0 && digits < 3) {
            ms *= 10;
            ++digits;
        }
    }

    // Optional timezone offset (+HH:MM / -HH:MM)
    std::size_t tz_pos = iso.find_first_of("+-Z", 19);
    int64_t offset_ms = 0;
    if (tz_pos != std::string::npos && (iso[tz_pos] == '+' || iso[tz_pos] == '-')) {
        int tz_h = 0, tz_m = 0;
        if (std::sscanf(iso.c_str() + tz_pos + 1, "%2d:%2d", &tz_h, &tz_m) >= 1) {
            offset_ms = static_cast<int64_t>(tz_h * 3600 + tz_m * 60) * 1000;
            if (iso[tz_pos] == '+') {
                offset_ms = -offset_ms;
            }
        }
    }
    return static_cast<int64_t>(epoch_s) * 1000 + ms + offset_ms;
}

std::pair<int, uint64_t> directory_file_usage(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return {0, 0};
    int count = 0;
    uint64_t bytes = 0;
    for (auto it = std::filesystem::recursive_directory_iterator(
             dir, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        std::error_code fec;
        if (it->is_regular_file(fec)) {
            auto sz = it->file_size(fec);
            if (!fec) {
                ++count;
                bytes += sz;
            }
        }
    }
    return {count, bytes};
}

std::string to_upper_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

}  // namespace

StationDatabase::StationDatabase(
    std::filesystem::path data_root,
    std::optional<std::filesystem::path> explicit_db_path
) {
    std::error_code ec;
    data_root_ = std::filesystem::absolute(data_root, ec);
    if (ec) data_root_ = data_root;
    evidence_root_ = data_root_ / "evidence";

    if (explicit_db_path.has_value()) {
        db_path_ = std::filesystem::absolute(*explicit_db_path, ec);
        if (ec) db_path_ = *explicit_db_path;
    } else {
        auto sqlite3_candidate = data_root_ / "station.sqlite3";
        auto db_candidate = data_root_ / "station.db";
        if (std::filesystem::exists(sqlite3_candidate, ec)) {
            db_path_ = sqlite3_candidate;
        } else if (std::filesystem::exists(db_candidate, ec)) {
            db_path_ = db_candidate;
        } else {
            db_path_ = sqlite3_candidate;
        }
    }

    open_and_initialize();
}

StationDatabase::~StationDatabase() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_) {
        try {
            exec_sql_unlocked("PRAGMA wal_checkpoint(TRUNCATE);");
        } catch (...) {
        }
        sqlite3_close_v2(db_);
        db_ = nullptr;
    }
}

json StationDatabase::default_config() {
    return json{
        {"config_version", 0},
        {"roi_config_version", "ROI_UNCALIBRATED"},
        {"camera_device_id", "CAM_STATION_01"},
        {"camera_width", kDefaultCameraWidth},
        {"camera_height", kDefaultCameraHeight},
        {"calibrated_at_utc", nullptr},
        {"camera_covers_both_ends", true},
        {"rois", json::object()},
        {"thresholds", json{
            {"hand_glove_confidence", 0.70},
            {"label_min_confidence", 0.85},
            {"label_min_margin", 0.15},
            {"stable_state_ms", 500},
            {"min_stable_frames", 5},
            {"min_laplacian_variance", 120.0},
            {"min_brightness", 55.0},
            {"max_brightness", 225.0},
            {"max_overexposed_fraction", 0.08}
        }},
        {"mode", "shadow"},
        {"interlock_output_enabled", false}
    };
}

json StationDatabase::default_station_state() {
    return json{
        {"mode", "shadow"},
        {"interlock_output_enabled", false},
        {"camera_status", "NOT_CONNECTED"},
        {"camera_name", nullptr},
        {"camera_device_id", nullptr},
        {"backend", nullptr},
        {"resolution", nullptr},
        {"camera_width", kDefaultCameraWidth},
        {"camera_height", kDefaultCameraHeight},
        {"fps", nullptr},
        {"camera_settings_locked", false},
        {"label_status", "unknown"},
        {"label_confidence", nullptr},
        {"view_valid", false},
        {"view_valid_reasons", json::array({"RUNTIME_NOT_CONNECTED"})},
        {"quality_ok", false},
        {"laplacian_variance", nullptr},
        {"brightness", nullptr},
        {"overexposed_fraction", nullptr},
        {"underexposed_fraction", nullptr},
        {"left_grip_active", nullptr},
        {"right_grip_active", nullptr},
        {"label_contact_active", nullptr},
        {"hand_contact_candidate", false},
        {"contact_evidence_type", "2d_visual_proximity_proxy"},
        {"proves_physical_contact", false},
        {"fsm_state", "WAIT_PART"},
        {"outcome", "WAITING"},
        {"result_reason", "RUNTIME_NOT_CONNECTED"},
        {"c_timer_ms", 0},
        {"cycle_id", nullptr},
        {"pipe_color", nullptr},
        {"operator_anon_id", nullptr},
        {"frame_id", nullptr},
        {"capture_monotonic_s", nullptr},
        {"model_version", nullptr},
        {"config_version", nullptr},
        {"mirror_horizontal", true},
        {"updated_at_utc", utc_now_iso()}
    };
}

std::string StationDatabase::validate_cycle_id(const std::string& raw_cycle_id) {
    static const std::regex kRe("^[A-Za-z0-9_-]{1,80}$");
    if (!std::regex_match(raw_cycle_id, kRe)) {
        throw DatabaseHttpError(400, "INVALID_CYCLE_ID");
    }
    return raw_cycle_id;
}

json StationDatabase::validate_config_update(const json& payload) {
    if (!payload.is_object()) {
        throw DatabaseHttpError(422, "INVALID_CONFIG_PAYLOAD");
    }
    static const std::set<std::string> kAllowedKeys = {
        "camera_device_id", "camera_width", "camera_height",
        "camera_covers_both_ends", "rois", "thresholds"
    };
    for (auto it = payload.begin(); it != payload.end(); ++it) {
        if (kAllowedKeys.find(it.key()) == kAllowedKeys.end()) {
            throw DatabaseHttpError(422, "EXTRA_FIELD_FORBIDDEN:" + it.key());
        }
    }

    if (!payload.contains("camera_device_id") || !payload["camera_device_id"].is_string()) {
        throw DatabaseHttpError(422, "INVALID_CAMERA_DEVICE_ID");
    }
    const std::string cam_id = payload["camera_device_id"].get<std::string>();
    if (cam_id.empty() || cam_id.size() > 256) {
        throw DatabaseHttpError(422, "INVALID_CAMERA_DEVICE_ID");
    }

    if (!payload.contains("camera_width") || !payload["camera_width"].is_number_integer() || payload["camera_width"].is_boolean()) {
        throw DatabaseHttpError(422, "INVALID_CAMERA_WIDTH");
    }
    if (!payload.contains("camera_height") || !payload["camera_height"].is_number_integer() || payload["camera_height"].is_boolean()) {
        throw DatabaseHttpError(422, "INVALID_CAMERA_HEIGHT");
    }
    const int cam_w = payload["camera_width"].get<int>();
    const int cam_h = payload["camera_height"].get<int>();
    if (cam_w < 1 || cam_w > 16384 || cam_h < 1 || cam_h > 16384) {
        throw DatabaseHttpError(422, "CAMERA_DIMENSIONS_OUT_OF_RANGE");
    }

    bool covers_both = true;
    if (payload.contains("camera_covers_both_ends")) {
        if (!payload["camera_covers_both_ends"].is_boolean()) {
            throw DatabaseHttpError(422, "INVALID_CAMERA_COVERS_BOTH_ENDS");
        }
        covers_both = payload["camera_covers_both_ends"].get<bool>();
    }

    if (!payload.contains("rois") || !payload["rois"].is_object()) {
        throw DatabaseHttpError(422, "INVALID_ROIS");
    }
    if (!payload.contains("thresholds") || !payload["thresholds"].is_object()) {
        throw DatabaseHttpError(422, "INVALID_THRESHOLDS");
    }

    std::set<std::string> required_rois = {"tube_roi", "label_roi", "label_contact_roi"};
    if (covers_both) {
        required_rois.insert("left_grip_zone");
        required_rois.insert("right_grip_zone");
    }
    std::set<std::string> allowed_rois = required_rois;
    allowed_rois.insert("quality_roi");

    const json& rois = payload["rois"];
    std::vector<std::string> missing;
    for (const auto& req_name : required_rois) {
        if (!rois.contains(req_name)) missing.push_back(req_name);
    }
    if (!missing.empty()) {
        throw DatabaseHttpError(422, json{{"missing_rois", missing}}, "MISSING_ROIS");
    }

    json clean_rois = json::object();
    for (auto it = rois.begin(); it != rois.end(); ++it) {
        const std::string& name = it.key();
        const json& box = it.value();
        if (allowed_rois.find(name) == allowed_rois.end() || !box.is_array() || box.size() != 4) {
            throw DatabaseHttpError(422, "INVALID_ROI:" + name);
        }
        std::array<double, 4> coords{};
        for (std::size_t i = 0; i < 4; ++i) {
            if (!box[i].is_number() || box[i].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_ROI:" + name);
            }
            coords[i] = box[i].get<double>();
            if (!std::isfinite(coords[i])) {
                throw DatabaseHttpError(422, "INVALID_ROI:" + name);
            }
        }
        if (!(0.0 <= coords[0] && coords[0] < coords[2] && coords[2] <= 1.0 &&
              0.0 <= coords[1] && coords[1] < coords[3] && coords[3] <= 1.0)) {
            throw DatabaseHttpError(422, "ROI_OUT_OF_NORMALIZED_BOUNDS:" + name);
        }
        clean_rois[name] = json::array({coords[0], coords[1], coords[2], coords[3]});
    }

    if (covers_both) {
        const auto c = clean_rois["label_contact_roi"];
        const double cx1 = c[0].get<double>(), cy1 = c[1].get<double>();
        const double cx2 = c[2].get<double>(), cy2 = c[3].get<double>();
        const double area = (cx2 - cx1) * (cy2 - cy1);
        for (const char* gname : {"left_grip_zone", "right_grip_zone"}) {
            const auto g = clean_rois[gname];
            const double gx1 = g[0].get<double>(), gy1 = g[1].get<double>();
            const double gx2 = g[2].get<double>(), gy2 = g[3].get<double>();
            const double iw = std::max(0.0, std::min(cx2, gx2) - std::max(cx1, gx1));
            const double ih = std::max(0.0, std::min(cy2, gy2) - std::max(cy1, gy1));
            if (area > 0.0 && (iw * ih) / area > 0.01) {
                throw DatabaseHttpError(422, "CONTACT_ROI_OVERLAPS_GRIP_ZONE");
            }
        }
    }

    json clean_thresholds = json::object();
    for (auto it = payload["thresholds"].begin(); it != payload["thresholds"].end(); ++it) {
        if (it.value().is_null()) {
            clean_thresholds[it.key()] = nullptr;
        } else if (it.value().is_number() && !it.value().is_boolean()) {
            double v = it.value().get<double>();
            if (!std::isfinite(v)) {
                throw DatabaseHttpError(422, "INVALID_THRESHOLD:" + it.key());
            }
            clean_thresholds[it.key()] = v;
        } else {
            throw DatabaseHttpError(422, "INVALID_THRESHOLD:" + it.key());
        }
    }

    auto get_thr = [&](const char* k) -> std::optional<double> {
        if (!clean_thresholds.contains(k) || clean_thresholds[k].is_null()) return std::nullopt;
        return clean_thresholds[k].get<double>();
    };

    for (const char* key : {"hand_glove_confidence", "label_min_confidence", "label_min_margin"}) {
        if (auto v = get_thr(key); v.has_value() && (*v < 0.0 || *v > 1.0)) {
            throw DatabaseHttpError(422, std::string("THRESHOLD_OUT_OF_RANGE:") + key);
        }
    }
    for (const char* key : {"stable_state_ms", "min_stable_frames", "min_laplacian_variance",
                            "min_brightness", "max_brightness", "max_overexposed_fraction"}) {
        if (auto v = get_thr(key); v.has_value() && *v < 0.0) {
            throw DatabaseHttpError(422, std::string("THRESHOLD_MUST_BE_NONNEGATIVE:") + key);
        }
    }
    for (const char* key : {"min_brightness", "max_brightness"}) {
        if (auto v = get_thr(key); v.has_value() && *v > 255.0) {
            throw DatabaseHttpError(422, std::string("BRIGHTNESS_THRESHOLD_OUT_OF_RANGE:") + key);
        }
    }
    if (auto v = get_thr("max_overexposed_fraction"); v.has_value() && *v > 1.0) {
        throw DatabaseHttpError(422, "OVEREXPOSED_FRACTION_MUST_BE_AT_MOST_ONE");
    }
    auto min_b = get_thr("min_brightness");
    auto max_b = get_thr("max_brightness");
    if (min_b.has_value() && max_b.has_value() && *min_b >= *max_b) {
        throw DatabaseHttpError(422, "MIN_BRIGHTNESS_MUST_BE_LOWER_THAN_MAX");
    }

    return json{
        {"camera_device_id", cam_id},
        {"camera_width", cam_w},
        {"camera_height", cam_h},
        {"camera_covers_both_ends", covers_both},
        {"rois", clean_rois},
        {"thresholds", clean_thresholds},
        {"mode", "shadow"},
        {"interlock_output_enabled", false}
    };
}

void StationDatabase::open_and_initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::error_code ec;
    std::filesystem::create_directories(data_root_, ec);
    std::filesystem::create_directories(evidence_root_, ec);
    if (db_path_.has_parent_path()) {
        std::filesystem::create_directories(db_path_.parent_path(), ec);
    }

    const std::string db_utf8 = path_to_utf8(db_path_);
    int rc = sqlite3_open_v2(
        db_utf8.c_str(),
        &db_,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        nullptr
    );
    if (rc != SQLITE_OK) {
        std::string err = db_ ? sqlite3_errmsg(db_) : "sqlite3_open_v2 failed";
        if (db_) {
            sqlite3_close_v2(db_);
            db_ = nullptr;
        }
        throw std::runtime_error("Failed to open SQLite database '" + db_utf8 + "': " + err);
    }

    sqlite3_busy_timeout(db_, 10000);
    exec_sql_unlocked("PRAGMA journal_mode=WAL;");
    exec_sql_unlocked("PRAGMA foreign_keys=ON;");
    exec_sql_unlocked("PRAGMA synchronous=FULL;");
    exec_sql_unlocked("PRAGMA journal_size_limit=67108864;");
    exec_sql_unlocked("PRAGMA wal_autocheckpoint=1000;");

    exec_sql_unlocked(R"SQL(
        CREATE TABLE IF NOT EXISTS cycles (
            cycle_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            initial_label TEXT,
            final_outcome TEXT NOT NULL DEFAULT 'WAITING',
            result_reason TEXT,
            pipe_color TEXT,
            operator_anon_id TEXT,
            model_version TEXT,
            config_version TEXT,
            fsm_state TEXT NOT NULL DEFAULT 'WAIT_PART',
            source TEXT NOT NULL DEFAULT 'station'
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            cycle_id TEXT,
            created_at_utc TEXT NOT NULL,
            capture_monotonic_s REAL,
            frame_id INTEGER,
            event_type TEXT NOT NULL,
            state_before TEXT,
            state_after TEXT,
            outcome TEXT,
            reason TEXT,
            payload_json TEXT NOT NULL,
            FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id)
        );
        CREATE INDEX IF NOT EXISTS events_cycle_idx ON events(cycle_id, created_at_utc);
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            cycle_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            kind TEXT NOT NULL,
            file_relpath TEXT NOT NULL UNIQUE,
            mime_type TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            frame_id INTEGER,
            capture_monotonic_s REAL,
            metadata_json TEXT NOT NULL,
            FOREIGN KEY(cycle_id) REFERENCES cycles(cycle_id),
            FOREIGN KEY(event_id) REFERENCES events(event_id)
        );
        CREATE TABLE IF NOT EXISTS configuration_history (
            config_version INTEGER PRIMARY KEY,
            roi_config_version TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            changed_by TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS config_current (
            singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
            config_version INTEGER NOT NULL,
            roi_config_version TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            changed_by TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            license_status TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            registered_by TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS station_state (
            singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
            updated_at_utc TEXT NOT NULL,
            state_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS station_purge_state (
            singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
            purge_generation INTEGER NOT NULL DEFAULT 0,
            purged_at_utc TEXT,
            purged_monotonic_s REAL
        );
    )SQL");

    if (!column_exists_unlocked("cycles", "source")) {
        TransactionGuard tx(db_);
        exec_sql_unlocked("ALTER TABLE cycles ADD COLUMN source TEXT NOT NULL DEFAULT 'station';");
        backfill_legacy_cycle_sources_unlocked();
        tx.commit();
    }

    const std::string now = utc_now_iso();
    // Seed configuration_history & config_current
    {
        StmtGuard chk(db_, "SELECT config_version, roi_config_version, created_at_utc, changed_by, config_json FROM configuration_history ORDER BY config_version DESC LIMIT 1;");
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            const json def_cfg = default_config();
            const std::string cfg_str = def_cfg.dump();
            StmtGuard ins_hist(db_, "INSERT INTO configuration_history VALUES(0, 'ROI_UNCALIBRATED', ?, 'system', ?);");
            bind_text(ins_hist.get(), 1, now);
            bind_text(ins_hist.get(), 2, cfg_str);
            sqlite3_step(ins_hist.get());

            StmtGuard ins_cur(db_, "INSERT OR REPLACE INTO config_current VALUES(1, 0, 'ROI_UNCALIBRATED', ?, 'system', ?);");
            bind_text(ins_cur.get(), 1, now);
            bind_text(ins_cur.get(), 2, cfg_str);
            sqlite3_step(ins_cur.get());
        } else {
            int ver = sqlite3_column_int(chk.get(), 0);
            std::string roi_ver = col_text(chk.get(), 1);
            std::string created = col_text(chk.get(), 2);
            std::string actor = col_text(chk.get(), 3);
            std::string cfg_str = col_text(chk.get(), 4);

            // Upgrade uncalibrated v0 legacy 1408x768 default to 1280x720 while preserving calibrated configs
            if (ver == 0 && roi_ver == "ROI_UNCALIBRATED") {
                try {
                    json parsed = json::parse(cfg_str);
                    if (parsed.is_object() &&
                        parsed.value("camera_width", 0) != kDefaultCameraWidth &&
                        (!parsed.contains("rois") || parsed["rois"].empty())) {
                        parsed["camera_width"] = kDefaultCameraWidth;
                        parsed["camera_height"] = kDefaultCameraHeight;
                        cfg_str = parsed.dump();
                        StmtGuard upd_hist(db_, "UPDATE configuration_history SET config_json=? WHERE config_version=0;");
                        bind_text(upd_hist.get(), 1, cfg_str);
                        sqlite3_step(upd_hist.get());
                    }
                } catch (...) {
                }
            }

            StmtGuard chk_cur(db_, "SELECT 1 FROM config_current WHERE singleton_id=1;");
            if (sqlite3_step(chk_cur.get()) != SQLITE_ROW) {
                StmtGuard ins_cur(db_, "INSERT INTO config_current VALUES(1, ?, ?, ?, ?, ?);");
                sqlite3_bind_int(ins_cur.get(), 1, ver);
                bind_text(ins_cur.get(), 2, roi_ver);
                bind_text(ins_cur.get(), 3, created.empty() ? now : created);
                bind_text(ins_cur.get(), 4, actor.empty() ? "system" : actor);
                bind_text(ins_cur.get(), 5, cfg_str);
                sqlite3_step(ins_cur.get());
            }
        }
    }

    // Seed station_state
    {
        StmtGuard chk(db_, "SELECT 1 FROM station_state WHERE singleton_id=1;");
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            const std::string init_str = default_station_state().dump();
            StmtGuard ins(db_, "INSERT INTO station_state VALUES(1, ?, ?);");
            bind_text(ins.get(), 1, now);
            bind_text(ins.get(), 2, init_str);
            sqlite3_step(ins.get());
        }
    }

    // Seed station_purge_state
    {
        StmtGuard chk(db_, "SELECT 1 FROM station_purge_state WHERE singleton_id=1;");
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            exec_sql_unlocked("INSERT INTO station_purge_state(singleton_id, purge_generation, purged_at_utc, purged_monotonic_s) VALUES(1, 0, NULL, NULL);");
        }
    }
}

void StationDatabase::exec_sql_unlocked(const std::string& sql) const {
    char* errmsg = nullptr;
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errmsg) != SQLITE_OK) {
        std::string err = errmsg ? errmsg : "SQLite exec error";
        if (errmsg) sqlite3_free(errmsg);
        throw std::runtime_error(err);
    }
}

bool StationDatabase::column_exists_unlocked(const std::string& table_name, const std::string& col_name) const {
    const std::string pragma = "PRAGMA table_info(" + table_name + ");";
    StmtGuard stmt(db_, pragma.c_str());
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        if (col_text(stmt.get(), 1) == col_name) return true;
    }
    return false;
}

void StationDatabase::backfill_legacy_cycle_sources_unlocked() {
    static const std::set<std::string> kSimEvents = {
        "SCENARIO_A_PASS", "SCENARIO_B_PASS", "SCENARIO_C_PASS",
        "SCENARIO_ALARM", "SCENARIO_INITIAL_ALARM", "SCENARIO_BARE_HAND_ALARM",
        "SCENARIO_GLARE_INVALID", "SCENARIO_RESET"
    };

    std::vector<std::string> cycle_ids;
    {
        StmtGuard stmt(db_, "SELECT cycle_id FROM cycles;");
        while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            cycle_ids.push_back(col_text(stmt.get(), 0));
        }
    }

    for (const auto& cid : cycle_ids) {
        bool has_sim_evidence = false;
        bool has_live_marker = false;
        bool has_station_marker = false;
        int ev_count = 0;
        bool all_events_are_sim_transitions = false;

        {
            StmtGuard evd(db_, "SELECT metadata_json FROM evidence WHERE cycle_id=?;");
            bind_text(evd.get(), 1, cid);
            while (sqlite3_step(evd.get()) == SQLITE_ROW) {
                try {
                    json meta = json::parse(col_text(evd.get(), 0));
                    if (!meta.is_object()) {
                        has_station_marker = true;
                        continue;
                    }
                    std::string op_ev = meta.value("operation_event", "");
                    if (kSimEvents.find(op_ev) != kSimEvents.end()) {
                        has_sim_evidence = true;
                    } else if (op_ev == "LIVE_CAMERA_MANUAL_CAPTURE" ||
                               op_ev == "ALARM_AUTO_CAPTURE" ||
                               op_ev.rfind("DATASET_SAMPLE_", 0) == 0) {
                        has_live_marker = true;
                    } else {
                        has_station_marker = true;
                    }
                } catch (...) {
                    has_station_marker = true;
                }
            }
        }

        {
            StmtGuard evs(db_, "SELECT event_type, payload_json FROM events WHERE cycle_id=?;");
            bind_text(evs.get(), 1, cid);
            bool seen_any = false;
            bool all_trans = true;
            while (sqlite3_step(evs.get()) == SQLITE_ROW) {
                seen_any = true;
                ++ev_count;
                std::string ev_type = col_text(evs.get(), 0);
                try {
                    json payload = json::parse(col_text(evs.get(), 1));
                    if (!payload.is_object()) {
                        all_trans = false;
                        has_station_marker = true;
                        continue;
                    }
                    if (payload.value("record_type", "") == "STATE_TRANSITION") {
                        continue;
                    }
                    all_trans = false;
                    if (ev_type == "LIVE_CAPTURE" ||
                        ev_type.rfind("SAMPLE_", 0) == 0 ||
                        payload.value("live_stream_url", "") == "/api/camera/stream" ||
                        payload.value("state_source", "") == "live") {
                        has_live_marker = true;
                    } else if (ev_type.rfind("EVIDENCE_", 0) == 0 &&
                               kSimEvents.find(payload.value("operation_event", "")) != kSimEvents.end()) {
                        has_sim_evidence = true;
                    } else {
                        has_station_marker = true;
                    }
                } catch (...) {
                    all_trans = false;
                    has_station_marker = true;
                }
            }
            all_events_are_sim_transitions = seen_any && all_trans && ev_count > 0;
        }

        if ((has_sim_evidence || all_events_are_sim_transitions) && !has_station_marker && !has_live_marker) {
            StmtGuard upd(db_, "UPDATE cycles SET source='simulated' WHERE cycle_id=?;");
            bind_text(upd.get(), 1, cid);
            sqlite3_step(upd.get());
        } else if (has_live_marker && !has_station_marker && !has_sim_evidence) {
            StmtGuard upd(db_, "UPDATE cycles SET source='live' WHERE cycle_id=?;");
            bind_text(upd.get(), 1, cid);
            sqlite3_step(upd.get());
        }
    }
}

json StationDatabase::latest_config_unlocked() const {
    {
        StmtGuard stmt(db_, "SELECT config_json FROM config_current WHERE singleton_id=1;");
        if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            try {
                return json::parse(col_text(stmt.get(), 0));
            } catch (...) {
            }
        }
    }
    {
        StmtGuard stmt(db_, "SELECT config_json FROM configuration_history ORDER BY config_version DESC LIMIT 1;");
        if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            try {
                return json::parse(col_text(stmt.get(), 0));
            } catch (...) {
            }
        }
    }
    return default_config();
}

json StationDatabase::cycle_summary_all_unlocked() const {
    int64_t total = 0, pass = 0, alarm = 0, view_invalid = 0, waiting = 0, fault = 0;
    StmtGuard stmt(db_, "SELECT final_outcome, COUNT(*) FROM cycles GROUP BY final_outcome;");
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        int64_t cnt = sqlite3_column_int64(stmt.get(), 1);
        total += cnt;
        std::string outcome = to_upper_ascii(col_text(stmt.get(), 0));
        if (outcome == "PASS") pass += cnt;
        else if (outcome == "ALARM") alarm += cnt;
        else if (outcome == "VIEW_INVALID") view_invalid += cnt;
        else if (outcome == "FAULT") fault += cnt;
        else waiting += cnt;
    }
    return json{
        {"total", total},
        {"pass", pass},
        {"alarm", alarm},
        {"view_invalid", view_invalid},
        {"waiting", waiting},
        {"fault", fault}
    };
}

json StationDatabase::attach_latest_evidence_and_summary_unlocked(json state) const {
    if (!state.is_object()) state = json::object();
    StmtGuard stmt(db_, "SELECT evidence_id, kind FROM evidence ORDER BY created_at_utc DESC LIMIT 1;");
    if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        std::string eid = col_text(stmt.get(), 0);
        std::string kind = col_text(stmt.get(), 1);
        state["latest_evidence_url"] = "/api/evidence/" + eid + "/file";
        state["latest_evidence_kind"] = kind;
    } else {
        state.erase("latest_evidence_url");
        state.erase("latest_evidence_kind");
    }
    state["cycle_summary_all"] = cycle_summary_all_unlocked();
    return state;
}

bool StationDatabase::is_stale_live_transition_unlocked(const json& state_data) const {
    if (auto cid = opt_json_string(state_data, "cycle_id"); cid.has_value()) {
        if (deleted_cycles_.find(*cid) != deleted_cycles_.end()) {
            return true;
        }
    }

    StmtGuard stmt(db_, "SELECT purge_generation, purged_at_utc FROM station_purge_state WHERE singleton_id=1;");
    if (sqlite3_step(stmt.get()) != SQLITE_ROW || sqlite3_column_type(stmt.get(), 1) == SQLITE_NULL) {
        return false;
    }

    const int64_t purged_gen = sqlite3_column_int64(stmt.get(), 0);
    const std::string purged_utc = col_text(stmt.get(), 1);
    if (purged_utc.empty()) return false;

    if (auto state_gen = opt_json_int64(state_data, "purge_generation"); state_gen.has_value()) {
        if (*state_gen < purged_gen) return true;
    }

    auto cap_mono = opt_json_double(state_data, "capture_monotonic_s");
    std::optional<int64_t> state_ms;
    if (auto raw_utc = opt_json_string(state_data, "updated_at_utc"); raw_utc.has_value()) {
        state_ms = parse_iso_utc_epoch_ms(*raw_utc);
    }
    auto purged_ms = parse_iso_utc_epoch_ms(purged_utc);

    if (cap_mono.has_value() && purge_cutoff_monotonic_.has_value()) {
        const double cutoff = *purge_cutoff_monotonic_;
        if (*cap_mono < cutoff - 0.001) return true;
        if (*cap_mono <= cutoff && (!state_ms.has_value() || !purged_ms.has_value() || *state_ms <= *purged_ms)) {
            return true;
        }
        if (state_ms.has_value() && purged_ms.has_value() && *state_ms < *purged_ms) {
            return true;
        }
        return false;
    }

    if (state_ms.has_value() && purged_ms.has_value()) {
        return *state_ms <= *purged_ms;
    }
    return false;
}

std::string StationDatabase::insert_event_unlocked(
    const std::optional<std::string>& cycle_id,
    const std::string& event_type,
    const json& payload,
    const std::optional<std::string>& state_before,
    const std::optional<std::string>& state_after,
    const std::optional<std::string>& outcome,
    const std::optional<std::string>& reason,
    std::optional<double> capture_monotonic_s,
    std::optional<int64_t> frame_id,
    const std::optional<std::string>& explicit_event_id,
    const std::optional<std::string>& explicit_created_utc
) {
    const std::string ev_id = (explicit_event_id.has_value() && !explicit_event_id->empty())
                                  ? *explicit_event_id
                                  : random_token_hex(16);
    if (explicit_event_id.has_value() && !explicit_event_id->empty()) {
        StmtGuard chk(db_, "SELECT event_id FROM events WHERE event_id=?;");
        bind_text(chk.get(), 1, ev_id);
        if (sqlite3_step(chk.get()) == SQLITE_ROW) {
            return ev_id;
        }
    }

    const std::string created = (explicit_created_utc.has_value() && !explicit_created_utc->empty())
                                    ? *explicit_created_utc
                                    : utc_now_iso();
    const std::string payload_str = payload.dump();

    StmtGuard ins(db_, "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?);");
    bind_text(ins.get(), 1, ev_id);
    bind_text_or_null(ins.get(), 2, cycle_id);
    bind_text(ins.get(), 3, created);
    bind_double_or_null(ins.get(), 4, capture_monotonic_s);
    bind_int64_or_null(ins.get(), 5, frame_id);
    bind_text(ins.get(), 6, event_type);
    bind_text_or_null(ins.get(), 7, state_before);
    bind_text_or_null(ins.get(), 8, state_after);
    bind_text_or_null(ins.get(), 9, outcome);
    bind_text_or_null(ins.get(), 10, reason);
    bind_text(ins.get(), 11, payload_str);
    if (sqlite3_step(ins.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("Failed to insert event: ") + sqlite3_errmsg(db_));
    }
    return ev_id;
}

bool StationDatabase::safe_unlink_evidence_file(const std::string& file_relpath) const {
    if (file_relpath.empty()) return false;
    std::error_code ec;
    auto candidate = data_root_ / path_from_utf8(file_relpath);
    if (!is_path_within(evidence_root_, candidate)) {
        return false;
    }
    if (std::filesystem::is_regular_file(candidate, ec)) {
        bool removed = std::filesystem::remove(candidate, ec);
        auto parent = candidate.parent_path();
        auto abs_ev_root = std::filesystem::weakly_canonical(evidence_root_, ec);
        auto abs_parent = std::filesystem::weakly_canonical(parent, ec);
        if (!ec && abs_parent != abs_ev_root && is_path_within(evidence_root_, parent)) {
            std::filesystem::remove(parent, ec);  // only removes if empty
        }
        return removed;
    }
    return false;
}

std::vector<std::filesystem::path> StationDatabase::snapshot_evidence_files() const {
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    if (!std::filesystem::is_directory(evidence_root_, ec)) return files;
    for (auto it = std::filesystem::recursive_directory_iterator(
             evidence_root_, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        std::error_code fec;
        if (it->is_regular_file(fec) && is_path_within(evidence_root_, it->path())) {
            files.push_back(it->path());
        }
    }
    return files;
}

int StationDatabase::safe_clean_evidence_dir(const std::vector<std::filesystem::path>& allowed_files) const {
    std::error_code ec;
    if (!std::filesystem::is_directory(evidence_root_, ec)) return 0;
    int removed = 0;
    for (const auto& p : allowed_files) {
        if (!is_path_within(evidence_root_, p)) continue;
        std::error_code fec;
        if (std::filesystem::is_regular_file(p, fec) && std::filesystem::remove(p, fec)) {
            ++removed;
        }
    }

    std::vector<std::filesystem::path> dirs;
    for (auto it = std::filesystem::recursive_directory_iterator(
             evidence_root_, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        std::error_code dec;
        if (it->is_directory(dec) && is_path_within(evidence_root_, it->path())) {
            dirs.push_back(it->path());
        }
    }
    std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
        return a.wstring().size() > b.wstring().size();
    });
    for (const auto& d : dirs) {
        std::error_code dec;
        std::filesystem::remove(d, dec);
    }
    return removed;
}

std::optional<std::filesystem::path> StationDatabase::safe_model_run_dir(const std::string& version) const {
    static const std::regex kVerRe("^[A-Za-z0-9._-]{1,80}$");
    if (version.empty() || version == "." || version == ".." ||
        version.find("..") != std::string::npos || !std::regex_match(version, kVerRe)) {
        return std::nullopt;
    }
    const auto runs_root = data_root_ / "runs" / "label_classifier";
    const auto candidate = runs_root / path_from_utf8(version);
    if (!is_path_within(runs_root, candidate)) {
        return std::nullopt;
    }
    std::error_code ec;
    if (std::filesystem::weakly_canonical(candidate.parent_path(), ec) !=
        std::filesystem::weakly_canonical(runs_root, ec)) {
        return std::nullopt;
    }
    return candidate;
}

json StationDatabase::get_persisted_station_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard stmt(db_, "SELECT state_json FROM station_state WHERE singleton_id=1;");
    if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        try {
            return json::parse(col_text(stmt.get(), 0));
        } catch (...) {
        }
    }
    return default_station_state();
}

void StationDatabase::set_persisted_station_state(const json& state_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string now = opt_json_string(state_data, "updated_at_utc").value_or(utc_now_iso());
    const std::string dumped = state_data.dump();
    StmtGuard stmt(db_, "UPDATE station_state SET updated_at_utc=?, state_json=? WHERE singleton_id=1;");
    bind_text(stmt.get(), 1, now);
    bind_text(stmt.get(), 2, dumped);
    sqlite3_step(stmt.get());
}

json StationDatabase::cycle_summary_all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cycle_summary_all_unlocked();
}

json StationDatabase::attach_latest_evidence_and_summary(json state) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return attach_latest_evidence_and_summary_unlocked(std::move(state));
}

json StationDatabase::get_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_config_unlocked();
}

json StationDatabase::update_config(const json& payload, const std::string& actor) {
    json config = validate_config_update(payload);
    const std::string calibrated = utc_now_iso();

    std::lock_guard<std::mutex> lock(mutex_);
    TransactionGuard tx(db_);
    json latest = latest_config_unlocked();
    int next_version = latest.value("config_version", 0) + 1;
    std::ostringstream ver_oss;
    ver_oss << "ROI_V" << std::setw(3) << std::setfill('0') << next_version;
    const std::string roi_ver = ver_oss.str();

    config["config_version"] = next_version;
    config["roi_config_version"] = roi_ver;
    config["calibrated_at_utc"] = calibrated;
    const std::string dumped = config.dump();

    {
        StmtGuard ins_hist(db_, "INSERT INTO configuration_history VALUES(?,?,?,?,?);");
        sqlite3_bind_int(ins_hist.get(), 1, next_version);
        bind_text(ins_hist.get(), 2, roi_ver);
        bind_text(ins_hist.get(), 3, calibrated);
        bind_text(ins_hist.get(), 4, actor);
        bind_text(ins_hist.get(), 5, dumped);
        if (sqlite3_step(ins_hist.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to insert configuration_history: ") + sqlite3_errmsg(db_));
        }
    }
    {
        StmtGuard upd_cur(db_, "INSERT OR REPLACE INTO config_current VALUES(1,?,?,?,?,?);");
        sqlite3_bind_int(upd_cur.get(), 1, next_version);
        bind_text(upd_cur.get(), 2, roi_ver);
        bind_text(upd_cur.get(), 3, calibrated);
        bind_text(upd_cur.get(), 4, actor);
        bind_text(upd_cur.get(), 5, dumped);
        if (sqlite3_step(upd_cur.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to update config_current: ") + sqlite3_errmsg(db_));
        }
    }
    tx.commit();
    return config;
}

json StationDatabase::list_config_history(int limit) const {
    const int clamped = std::clamp(limit, 1, 200);
    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard stmt(
        db_,
        "SELECT config_version, roi_config_version, created_at_utc, changed_by, config_json "
        "FROM configuration_history ORDER BY config_version DESC LIMIT ?;"
    );
    sqlite3_bind_int(stmt.get(), 1, clamped);
    json arr = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        json cfg_obj = json::object();
        try {
            cfg_obj = json::parse(col_text(stmt.get(), 4));
        } catch (...) {
        }
        arr.push_back(json{
            {"config_version", sqlite3_column_int(stmt.get(), 0)},
            {"roi_config_version", col_text(stmt.get(), 1)},
            {"created_at_utc", col_text(stmt.get(), 2)},
            {"changed_by", col_text(stmt.get(), 3)},
            {"config", std::move(cfg_obj)}
        });
    }
    return arr;
}

json StationDatabase::list_cycles(int limit) const {
    const int clamped = std::clamp(limit, 1, 500);
    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard stmt(
        db_,
        "SELECT cycle_id, created_at_utc, updated_at_utc, initial_label, final_outcome, "
        "result_reason, pipe_color, operator_anon_id, model_version, config_version, fsm_state, source "
        "FROM cycles ORDER BY updated_at_utc DESC LIMIT ?;"
    );
    sqlite3_bind_int(stmt.get(), 1, clamped);
    json arr = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        arr.push_back(json{
            {"cycle_id", col_text(stmt.get(), 0)},
            {"created_at_utc", col_text(stmt.get(), 1)},
            {"updated_at_utc", col_text(stmt.get(), 2)},
            {"initial_label", col_text_or_null_json(stmt.get(), 3)},
            {"final_outcome", col_text(stmt.get(), 4)},
            {"result_reason", col_text_or_null_json(stmt.get(), 5)},
            {"pipe_color", col_text_or_null_json(stmt.get(), 6)},
            {"operator_anon_id", col_text_or_null_json(stmt.get(), 7)},
            {"model_version", col_text_or_null_json(stmt.get(), 8)},
            {"config_version", col_text_or_null_json(stmt.get(), 9)},
            {"fsm_state", col_text(stmt.get(), 10)},
            {"source", col_text(stmt.get(), 11)}
        });
    }
    return arr;
}

json StationDatabase::list_cycle_events(const std::string& raw_cycle_id, int limit) const {
    const std::string cycle_id = validate_cycle_id(raw_cycle_id);
    const int clamped = std::clamp(limit, 1, 1000);
    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard stmt(
        db_,
        "SELECT event_id, cycle_id, created_at_utc, capture_monotonic_s, frame_id, "
        "event_type, state_before, state_after, outcome, reason, payload_json "
        "FROM events WHERE cycle_id=? ORDER BY created_at_utc DESC LIMIT ?;"
    );
    bind_text(stmt.get(), 1, cycle_id);
    sqlite3_bind_int(stmt.get(), 2, clamped);
    json arr = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        std::string raw_payload = col_text(stmt.get(), 10);
        json parsed_payload = json::object();
        try {
            parsed_payload = json::parse(raw_payload);
        } catch (...) {
        }
        arr.push_back(json{
            {"event_id", col_text(stmt.get(), 0)},
            {"cycle_id", col_text_or_null_json(stmt.get(), 1)},
            {"created_at_utc", col_text(stmt.get(), 2)},
            {"capture_monotonic_s", col_double_or_null_json(stmt.get(), 3)},
            {"frame_id", col_int64_or_null_json(stmt.get(), 4)},
            {"event_type", col_text(stmt.get(), 5)},
            {"state_before", col_text_or_null_json(stmt.get(), 6)},
            {"state_after", col_text_or_null_json(stmt.get(), 7)},
            {"outcome", col_text_or_null_json(stmt.get(), 8)},
            {"reason", col_text_or_null_json(stmt.get(), 9)},
            {"payload_json", raw_payload},
            {"payload", std::move(parsed_payload)}
        });
    }
    return arr;
}

json StationDatabase::delete_cycle(const std::string& raw_cycle_id) {
    const std::string cycle_id = validate_cycle_id(raw_cycle_id);
    std::vector<std::string> ev_relpaths;
    int deleted_evidence = 0;
    int deleted_events = 0;
    int deleted_cycles = 0;

    std::lock_guard<std::mutex> lock(mutex_);
    {
        TransactionGuard tx(db_);
        {
            StmtGuard chk(db_, "SELECT 1 FROM cycles WHERE cycle_id=?;");
            bind_text(chk.get(), 1, cycle_id);
            if (sqlite3_step(chk.get()) != SQLITE_ROW) {
                throw DatabaseHttpError(404, "CYCLE_NOT_FOUND");
            }
        }
        {
            StmtGuard sel(db_, "SELECT file_relpath FROM evidence WHERE cycle_id=?;");
            bind_text(sel.get(), 1, cycle_id);
            while (sqlite3_step(sel.get()) == SQLITE_ROW) {
                ev_relpaths.push_back(col_text(sel.get(), 0));
            }
        }
        {
            StmtGuard del_evd(db_, "DELETE FROM evidence WHERE cycle_id=?;");
            bind_text(del_evd.get(), 1, cycle_id);
            sqlite3_step(del_evd.get());
            deleted_evidence = sqlite3_changes(db_);
        }
        {
            StmtGuard del_evs(db_, "DELETE FROM events WHERE cycle_id=?;");
            bind_text(del_evs.get(), 1, cycle_id);
            sqlite3_step(del_evs.get());
            deleted_events = sqlite3_changes(db_);
        }
        {
            StmtGuard del_cyc(db_, "DELETE FROM cycles WHERE cycle_id=?;");
            bind_text(del_cyc.get(), 1, cycle_id);
            sqlite3_step(del_cyc.get());
            deleted_cycles = sqlite3_changes(db_);
        }
        tx.commit();
    }

    if (deleted_cycles_.size() >= kMaxTrackedDeletedCycles) {
        deleted_cycles_.clear();
    }
    deleted_cycles_.insert(cycle_id);

    int deleted_files = 0;
    for (const auto& rel : ev_relpaths) {
        if (safe_unlink_evidence_file(rel)) ++deleted_files;
    }
    auto cycle_dir = evidence_root_ / path_from_utf8(cycle_id);
    if (is_path_within(evidence_root_, cycle_dir)) {
        std::error_code ec;
        std::filesystem::remove(cycle_dir, ec);
    }

    return json{
        {"deleted", true},
        {"cycle_id", cycle_id},
        {"deleted_cycles", deleted_cycles},
        {"deleted_events", deleted_events},
        {"deleted_evidence", deleted_evidence},
        {"deleted_files", deleted_files},
        {"freed_files", deleted_files}
    };
}

json StationDatabase::purge_cycles(const std::string& scope) {
    if (scope != "simulated_only" && scope != "all") {
        throw DatabaseHttpError(422, "INVALID_PURGE_SCOPE");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    int deleted_cycles = 0;
    int deleted_events = 0;
    int deleted_evidence = 0;
    int deleted_files = 0;
    bool vacuumed = false;

    if (scope == "simulated_only") {
        std::vector<std::string> ev_relpaths;
        {
            TransactionGuard tx(db_);
            {
                StmtGuard sel(
                    db_,
                    "SELECT file_relpath FROM evidence "
                    "WHERE cycle_id IN (SELECT cycle_id FROM cycles WHERE source='simulated');"
                );
                while (sqlite3_step(sel.get()) == SQLITE_ROW) {
                    ev_relpaths.push_back(col_text(sel.get(), 0));
                }
            }
            {
                StmtGuard del_evd(
                    db_,
                    "DELETE FROM evidence WHERE cycle_id IN (SELECT cycle_id FROM cycles WHERE source='simulated');"
                );
                sqlite3_step(del_evd.get());
                deleted_evidence = sqlite3_changes(db_);
            }
            {
                StmtGuard del_evs(
                    db_,
                    "DELETE FROM events WHERE cycle_id IN (SELECT cycle_id FROM cycles WHERE source='simulated');"
                );
                sqlite3_step(del_evs.get());
                deleted_events = sqlite3_changes(db_);
            }
            {
                StmtGuard del_cyc(db_, "DELETE FROM cycles WHERE source='simulated';");
                sqlite3_step(del_cyc.get());
                deleted_cycles = sqlite3_changes(db_);
            }
            tx.commit();
        }
        for (const auto& rel : ev_relpaths) {
            if (safe_unlink_evidence_file(rel)) ++deleted_files;
        }
    } else {
        const auto pre_purge_files = snapshot_evidence_files();
        const std::string purged_utc = utc_now_iso();
        const double purged_mono = monotonic_seconds();
        std::vector<std::string> ev_relpaths;
        {
            TransactionGuard tx(db_);
            {
                StmtGuard sel(db_, "SELECT file_relpath FROM evidence;");
                while (sqlite3_step(sel.get()) == SQLITE_ROW) {
                    ev_relpaths.push_back(col_text(sel.get(), 0));
                }
            }
            {
                StmtGuard del_evd(db_, "DELETE FROM evidence;");
                sqlite3_step(del_evd.get());
                deleted_evidence = sqlite3_changes(db_);
            }
            {
                StmtGuard del_evs(db_, "DELETE FROM events;");
                sqlite3_step(del_evs.get());
                deleted_events = sqlite3_changes(db_);
            }
            {
                StmtGuard del_cyc(db_, "DELETE FROM cycles;");
                sqlite3_step(del_cyc.get());
                deleted_cycles = sqlite3_changes(db_);
            }
            {
                StmtGuard upd_p(
                    db_,
                    "UPDATE station_purge_state "
                    "SET purge_generation = purge_generation + 1, purged_at_utc = ?, purged_monotonic_s = ? "
                    "WHERE singleton_id = 1;"
                );
                bind_text(upd_p.get(), 1, purged_utc);
                sqlite3_bind_double(upd_p.get(), 2, purged_mono);
                sqlite3_step(upd_p.get());
            }
            tx.commit();
        }
        purge_cutoff_monotonic_ = purged_mono;
        deleted_cycles_.clear();

        for (const auto& rel : ev_relpaths) {
            if (safe_unlink_evidence_file(rel)) ++deleted_files;
        }
        deleted_files += safe_clean_evidence_dir(pre_purge_files);

        exec_sql_unlocked("VACUUM;");
        vacuumed = true;
    }

    return json{
        {"purged", true},
        {"scope", scope},
        {"deleted_cycles", deleted_cycles},
        {"deleted_events", deleted_events},
        {"deleted_evidence", deleted_evidence},
        {"deleted_files", deleted_files},
        {"freed_files", deleted_files},
        {"vacuumed", vacuumed}
    };
}

bool StationDatabase::record_live_transition(
    const json& state_data,
    const std::optional<std::string>& state_before,
    const std::vector<uint8_t>& evidence_jpeg
) {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_stale_live_transition_unlocked(state_data)) {
            return true;
        }

        const std::string now = utc_now_iso();
        json persisted_state = state_data.is_object() ? state_data : json::object();
        persisted_state["state_source"] = "live";

        std::optional<std::string> cycle_id = opt_json_string(state_data, "cycle_id");
        if (cycle_id.has_value() && cycle_id->empty()) cycle_id = std::nullopt;
        if (cycle_id.has_value()) {
            validate_cycle_id(*cycle_id);
        }

        std::optional<std::string> transition_event_id = opt_json_string(state_data, "transition_event_id");
        if (transition_event_id.has_value() && transition_event_id->empty()) {
            transition_event_id = std::nullopt;
        }

        const std::string outcome = opt_json_string(state_data, "outcome").value_or("WAITING");
        const std::string fsm_state = opt_json_string(state_data, "fsm_state").value_or("WAIT_PART");
        const auto result_reason = opt_json_string(state_data, "result_reason");
        const auto label_status = opt_json_string(state_data, "label_status");
        const auto pipe_color = opt_json_string(state_data, "pipe_color");
        const auto operator_id = opt_json_string(state_data, "operator_anon_id");
        const auto model_ver = opt_json_string(state_data, "model_version");
        const auto config_ver = opt_json_string(state_data, "config_version");
        const auto cap_mono = opt_json_double(state_data, "capture_monotonic_s");
        const auto frame_id = opt_json_int64(state_data, "frame_id");

        TransactionGuard tx(db_);
        {
            const std::string state_str = persisted_state.dump();
            StmtGuard upd_st(db_, "UPDATE station_state SET updated_at_utc=?, state_json=? WHERE singleton_id=1;");
            bind_text(upd_st.get(), 1, now);
            bind_text(upd_st.get(), 2, state_str);
            sqlite3_step(upd_st.get());
        }

        std::optional<ExclusiveEvidenceFileGuard> ev_file_guard;
        if (cycle_id.has_value()) {
            bool exists = false;
            {
                StmtGuard chk(db_, "SELECT 1 FROM cycles WHERE cycle_id=?;");
                bind_text(chk.get(), 1, *cycle_id);
                exists = (sqlite3_step(chk.get()) == SQLITE_ROW);
            }
            if (!exists) {
                StmtGuard ins_c(
                    db_,
                    "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,"
                    "result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state,source) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,'live');"
                );
                bind_text(ins_c.get(), 1, *cycle_id);
                bind_text(ins_c.get(), 2, now);
                bind_text(ins_c.get(), 3, now);
                bind_text_or_null(ins_c.get(), 4, label_status);
                bind_text(ins_c.get(), 5, outcome);
                bind_text_or_null(ins_c.get(), 6, result_reason);
                bind_text_or_null(ins_c.get(), 7, pipe_color);
                bind_text_or_null(ins_c.get(), 8, operator_id);
                bind_text_or_null(ins_c.get(), 9, model_ver);
                bind_text_or_null(ins_c.get(), 10, config_ver);
                bind_text(ins_c.get(), 11, fsm_state);
                sqlite3_step(ins_c.get());
            } else {
                StmtGuard upd_c(
                    db_,
                    "UPDATE cycles SET updated_at_utc=?,final_outcome=?,result_reason=?,pipe_color=?,"
                    "operator_anon_id=?,model_version=?,config_version=?,fsm_state=? WHERE cycle_id=?;"
                );
                bind_text(upd_c.get(), 1, now);
                bind_text(upd_c.get(), 2, outcome);
                bind_text_or_null(upd_c.get(), 3, result_reason);
                bind_text_or_null(upd_c.get(), 4, pipe_color);
                bind_text_or_null(upd_c.get(), 5, operator_id);
                bind_text_or_null(upd_c.get(), 6, model_ver);
                bind_text_or_null(upd_c.get(), 7, config_ver);
                bind_text(upd_c.get(), 8, fsm_state);
                bind_text(upd_c.get(), 9, *cycle_id);
                sqlite3_step(upd_c.get());
            }

            const std::string ev_type = result_reason.value_or("STATE_CHANGE");
            const std::string ev_id = insert_event_unlocked(
                cycle_id,
                ev_type,
                persisted_state,
                state_before,
                fsm_state,
                outcome,
                result_reason,
                cap_mono,
                frame_id,
                transition_event_id,
                now
            );

            if (outcome == "ALARM" && !evidence_jpeg.empty()) {
                bool dup_ev = false;
                {
                    StmtGuard chk_ev(db_, "SELECT 1 FROM evidence WHERE cycle_id=? AND event_id=? LIMIT 1;");
                    bind_text(chk_ev.get(), 1, *cycle_id);
                    bind_text(chk_ev.get(), 2, ev_id);
                    dup_ev = (sqlite3_step(chk_ev.get()) == SQLITE_ROW);
                }
                if (!dup_ev) {
                    const std::string evidence_id = random_token_hex(16);
                    const std::string sha = sha256_hex(evidence_jpeg);
                    const std::string relpath = "evidence/" + *cycle_id + "/" + evidence_id + ".jpg";
                    const auto abs_path = data_root_ / path_from_utf8(relpath);

                    ev_file_guard.emplace(evidence_root_, abs_path, evidence_jpeg.data(), evidence_jpeg.size());

                    json meta{
                        {"pipe_color", pipe_color.value_or("unknown")},
                        {"operation_event", result_reason.value_or("ALARM_AUTO_CAPTURE")},
                        {"operator_anon_id", operator_id.value_or("OP_STATION_01")},
                        {"label_status", label_status.has_value() ? json(*label_status) : json(nullptr)},
                        {"captured_at_utc", now}
                    };
                    const std::string meta_str = meta.dump();

                    StmtGuard ins_evd(db_, "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?);");
                    bind_text(ins_evd.get(), 1, evidence_id);
                    bind_text(ins_evd.get(), 2, *cycle_id);
                    bind_text(ins_evd.get(), 3, ev_id);
                    bind_text(ins_evd.get(), 4, now);
                    bind_text(ins_evd.get(), 5, "anomaly");
                    bind_text(ins_evd.get(), 6, relpath);
                    bind_text(ins_evd.get(), 7, "image/jpeg");
                    bind_text(ins_evd.get(), 8, sha);
                    sqlite3_bind_int64(ins_evd.get(), 9, frame_id.value_or(1));
                    bind_double_or_null(ins_evd.get(), 10, cap_mono);
                    bind_text(ins_evd.get(), 11, meta_str);
                    if (sqlite3_step(ins_evd.get()) != SQLITE_DONE) {
                        throw std::runtime_error(std::string("Failed to insert alarm evidence: ") + sqlite3_errmsg(db_));
                    }
                }
            }
        }

        tx.commit();
        if (ev_file_guard.has_value()) ev_file_guard->commit();
        if (++write_ops_since_checkpoint_ >= 100) {
            write_ops_since_checkpoint_ = 0;
            try {
                exec_sql_unlocked("PRAGMA wal_checkpoint(TRUNCATE);");
            } catch (...) {
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

void StationDatabase::run_wal_checkpoint_truncate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_) {
        exec_sql_unlocked("PRAGMA wal_checkpoint(TRUNCATE);");
        write_ops_since_checkpoint_ = 0;
    }
}

json StationDatabase::enforce_evidence_quota(std::uint64_t max_bytes, double min_free_ratio) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::error_code ec;
    std::uint64_t total_bytes = 0;
    if (std::filesystem::exists(evidence_root_, ec)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(evidence_root_, ec)) {
            if (entry.is_regular_file(ec)) {
                const auto fsz = entry.file_size(ec);
                if (!ec) total_bytes += static_cast<std::uint64_t>(fsz);
            }
        }
    }

    auto current_free_ratio = [&]() -> double {
        std::error_code sec;
        const auto sp = std::filesystem::space(evidence_root_, sec);
        if (sec || sp.capacity == 0) return 1.0;
        return static_cast<double>(sp.available) / static_cast<double>(sp.capacity);
    };

    int deleted_count = 0;
    std::uint64_t freed_bytes = 0;
    double free_ratio = current_free_ratio();

    while (db_ && (total_bytes > max_bytes || free_ratio < min_free_ratio)) {
        std::string ev_id;
        std::string relpath;
        {
            StmtGuard sel(
                db_,
                "SELECT evidence_id, file_relpath FROM evidence "
                "ORDER BY (CASE WHEN kind='anomaly' THEN 1 ELSE 0 END) ASC, created_at_utc ASC LIMIT 1;"
            );
            if (sqlite3_step(sel.get()) != SQLITE_ROW) {
                break;
            }
            ev_id = col_text(sel.get(), 0);
            relpath = col_text(sel.get(), 1);
        }

        const auto full_path = data_root_ / path_from_utf8(relpath);
        std::uint64_t file_sz = 0;
        std::error_code fec;
        if (is_path_within(evidence_root_, full_path) && std::filesystem::is_regular_file(full_path, fec)) {
            const auto sz = std::filesystem::file_size(full_path, fec);
            if (!fec) file_sz = static_cast<std::uint64_t>(sz);
        }

        {
            TransactionGuard tx(db_);
            StmtGuard del(db_, "DELETE FROM evidence WHERE evidence_id=?;");
            bind_text(del.get(), 1, ev_id);
            sqlite3_step(del.get());
            tx.commit();
        }

        if (safe_unlink_evidence_file(relpath) || file_sz == 0) {
            total_bytes = (total_bytes > file_sz) ? (total_bytes - file_sz) : 0;
            freed_bytes += file_sz;
        }
        ++deleted_count;
        free_ratio = current_free_ratio();
    }

    if (deleted_count > 0 && db_) {
        try {
            exec_sql_unlocked("PRAGMA wal_checkpoint(TRUNCATE);");
            write_ops_since_checkpoint_ = 0;
        } catch (...) {
        }
    }

    return json{
        {"deleted_count", deleted_count},
        {"freed_bytes", freed_bytes},
        {"remaining_bytes", total_bytes}
    };
}

json StationDatabase::pragma_diagnostics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int sync_mode = -1;
    int64_t journal_limit = -1;
    std::string journal_mode;
    if (db_) {
        {
            StmtGuard st(db_, "PRAGMA synchronous;");
            if (sqlite3_step(st.get()) == SQLITE_ROW) sync_mode = sqlite3_column_int(st.get(), 0);
        }
        {
            StmtGuard st(db_, "PRAGMA journal_size_limit;");
            if (sqlite3_step(st.get()) == SQLITE_ROW) journal_limit = sqlite3_column_int64(st.get(), 0);
        }
        {
            StmtGuard st(db_, "PRAGMA journal_mode;");
            if (sqlite3_step(st.get()) == SQLITE_ROW) journal_mode = col_text(st.get(), 0);
        }
    }
    return json{
        {"synchronous", sync_mode},
        {"journal_size_limit", journal_limit},
        {"journal_mode", journal_mode}
    };
}

json StationDatabase::save_manual_capture(
    const std::vector<uint8_t>& jpeg_bytes,
    const json& live_state,
    const std::string& fallback_cycle_id
) {
    const std::string raw_cid = opt_json_string(live_state, "cycle_id").value_or(fallback_cycle_id);
    const std::string cycle_id = validate_cycle_id(raw_cid.empty() ? fallback_cycle_id : raw_cid);
    const std::string now = utc_now_iso();
    const std::string evidence_id = random_token_hex(16);
    const std::string sha = sha256_hex(jpeg_bytes);
    const std::string relpath = "evidence/" + cycle_id + "/" + evidence_id + ".jpg";
    const auto abs_path = data_root_ / path_from_utf8(relpath);

    const auto cam_w = opt_json_int64(live_state, "camera_width");
    const auto cam_h = opt_json_int64(live_state, "camera_height");
    const std::string res_str = (cam_w.has_value() && cam_h.has_value())
                                    ? (std::to_string(*cam_w) + "x" + std::to_string(*cam_h))
                                    : "1280x720";

    json metadata{
        {"pipe_color", opt_json_string(live_state, "pipe_color").value_or("yellow")},
        {"operation_event", "LIVE_CAMERA_MANUAL_CAPTURE"},
        {"operator_anon_id", opt_json_string(live_state, "operator_anon_id").value_or("OP_STATION_01")},
        {"camera_metadata", json{
            {"camera_name", live_state.value("camera_name", json(nullptr))},
            {"camera_device_id", live_state.value("camera_device_id", json(nullptr))},
            {"resolution", res_str},
            {"fps", live_state.value("fps", json(nullptr))},
            {"laplacian_variance", live_state.value("laplacian_variance", json(nullptr))},
            {"brightness", live_state.value("brightness", json(nullptr))}
        }},
        {"label_status", live_state.value("label_status", json(nullptr))},
        {"captured_at_utc", now}
    };

    const auto frame_id = opt_json_int64(live_state, "frame_id");
    const auto cap_mono = opt_json_double(live_state, "capture_monotonic_s");
    const auto fsm_state = opt_json_string(live_state, "fsm_state");
    const auto outcome = opt_json_string(live_state, "outcome");

    std::lock_guard<std::mutex> lock(mutex_);
    ExclusiveEvidenceFileGuard file_guard(evidence_root_, abs_path, jpeg_bytes.data(), jpeg_bytes.size());
    TransactionGuard tx(db_);

    {
        StmtGuard chk(db_, "SELECT 1 FROM cycles WHERE cycle_id=?;");
        bind_text(chk.get(), 1, cycle_id);
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            StmtGuard ins_c(
                db_,
                "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,"
                "result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state,source) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'live');"
            );
            bind_text(ins_c.get(), 1, cycle_id);
            bind_text(ins_c.get(), 2, now);
            bind_text(ins_c.get(), 3, now);
            bind_text_or_null(ins_c.get(), 4, opt_json_string(live_state, "label_status"));
            bind_text(ins_c.get(), 5, outcome.value_or("WAITING"));
            bind_text_or_null(ins_c.get(), 6, opt_json_string(live_state, "result_reason"));
            bind_text_or_null(ins_c.get(), 7, opt_json_string(live_state, "pipe_color"));
            bind_text_or_null(ins_c.get(), 8, opt_json_string(live_state, "operator_anon_id"));
            bind_text_or_null(ins_c.get(), 9, opt_json_string(live_state, "model_version"));
            bind_text_or_null(ins_c.get(), 10, opt_json_string(live_state, "config_version"));
            bind_text(ins_c.get(), 11, fsm_state.value_or("INSPECTING"));
            sqlite3_step(ins_c.get());
        }
    }

    const std::string ev_id = insert_event_unlocked(
        cycle_id,
        "LIVE_CAPTURE",
        metadata,
        fsm_state,
        fsm_state,
        outcome,
        std::string("手动抓拍实时相机画面"),
        cap_mono,
        frame_id,
        std::nullopt,
        now
    );

    {
        const std::string meta_str = metadata.dump();
        StmtGuard ins_evd(db_, "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?);");
        bind_text(ins_evd.get(), 1, evidence_id);
        bind_text(ins_evd.get(), 2, cycle_id);
        bind_text(ins_evd.get(), 3, ev_id);
        bind_text(ins_evd.get(), 4, now);
        bind_text(ins_evd.get(), 5, "initial_label");
        bind_text(ins_evd.get(), 6, relpath);
        bind_text(ins_evd.get(), 7, "image/jpeg");
        bind_text(ins_evd.get(), 8, sha);
        sqlite3_bind_int64(ins_evd.get(), 9, frame_id.value_or(1));
        bind_double_or_null(ins_evd.get(), 10, cap_mono);
        bind_text(ins_evd.get(), 11, meta_str);
        if (sqlite3_step(ins_evd.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to insert manual capture evidence: ") + sqlite3_errmsg(db_));
        }
    }

    tx.commit();
    file_guard.commit();

    return json{
        {"evidence_id", evidence_id},
        {"cycle_id", cycle_id},
        {"sha256", sha},
        {"frame_id", frame_id.has_value() ? json(*frame_id) : json(nullptr)},
        {"capture_monotonic_s", cap_mono.has_value() ? json(*cap_mono) : json(nullptr)},
        {"url", "/api/evidence/" + evidence_id + "/file"},
        {"captured_at_utc", now}
    };
}

json StationDatabase::record_simulated_run(
    const SimulationRunResult& run,
    const std::string& requested_scenario,
    const json& validated_state
) {
    json data = validated_state.is_object() ? validated_state : json::object();
    std::string raw_cid = opt_json_string(data, "cycle_id").value_or("");
    if (raw_cid.empty() && run.transitions.is_array()) {
        for (const auto& tr : run.transitions) {
            if (auto cid = opt_json_string(tr, "cycle_id"); cid.has_value() && !cid->empty()) {
                raw_cid = *cid;
                break;
            }
        }
    }
    if (raw_cid.empty()) {
        raw_cid = "CYC_SIM_" + to_upper_ascii(random_token_hex(3));
    }
    const std::string cycle_id = validate_cycle_id(raw_cid);

    double elapsed_s = 1.8;
    if (run.transitions.is_array() && run.transitions.size() >= 2) {
        auto t0 = opt_json_double(run.transitions.front(), "now_monotonic_s");
        auto t1 = opt_json_double(run.transitions.back(), "now_monotonic_s");
        if (t0.has_value() && t1.has_value() && *t1 > *t0) {
            elapsed_s = std::clamp(*t1 - *t0, 0.8, 30.0);
        }
    }
    const std::string created_utc = utc_iso_offset_ms(-static_cast<int64_t>(elapsed_s * 1000.0));
    const std::string updated_utc = utc_now_iso();
    data["updated_at_utc"] = updated_utc;
    data["state_source"] = "simulated";

    std::vector<uint8_t> body = run.evidence_image_bytes;
    if (body.empty() && !run.reference_image_path.empty()) {
        read_file_bytes(path_from_utf8(run.reference_image_path), body);
    }
    if (body.empty()) {
        cv::Mat synth(240, 320, CV_8UC3, cv::Scalar(42, 90, 140));
        cv::imencode(".jpg", synth, body, {cv::IMWRITE_JPEG_QUALITY, 90});
    }

    const std::string sha = sha256_hex(body);
    const std::string kind = run.evidence_kind.empty() ? "initial_label" : run.evidence_kind;
    const std::string pipe_color = run.pipe_color.empty() ? "yellow" : run.pipe_color;
    const std::string initial_label =
        (requested_scenario == "alarm" || requested_scenario == "initial_alarm") ? "absent" : "complete";

    std::lock_guard<std::mutex> lock(mutex_);
    TransactionGuard tx(db_);

    {
        StmtGuard chk(db_, "SELECT 1 FROM cycles WHERE cycle_id=?;");
        bind_text(chk.get(), 1, cycle_id);
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            StmtGuard ins_c(
                db_,
                "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,"
                "result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state,source) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'simulated');"
            );
            bind_text(ins_c.get(), 1, cycle_id);
            bind_text(ins_c.get(), 2, created_utc);
            bind_text(ins_c.get(), 3, updated_utc);
            bind_text(ins_c.get(), 4, initial_label);
            bind_text(ins_c.get(), 5, opt_json_string(data, "outcome").value_or("WAITING"));
            bind_text_or_null(ins_c.get(), 6, opt_json_string(data, "result_reason"));
            bind_text(ins_c.get(), 7, pipe_color);
            bind_text_or_null(ins_c.get(), 8, opt_json_string(data, "operator_anon_id"));
            bind_text_or_null(ins_c.get(), 9, opt_json_string(data, "model_version"));
            bind_text_or_null(ins_c.get(), 10, opt_json_string(data, "config_version"));
            bind_text(ins_c.get(), 11, opt_json_string(data, "fsm_state").value_or("WAIT_PART"));
            sqlite3_step(ins_c.get());
        } else {
            StmtGuard upd_c(
                db_,
                "UPDATE cycles SET updated_at_utc=?,final_outcome=?,result_reason=?,pipe_color=?,"
                "operator_anon_id=?,model_version=?,config_version=?,fsm_state=? WHERE cycle_id=?;"
            );
            bind_text(upd_c.get(), 1, updated_utc);
            bind_text(upd_c.get(), 2, opt_json_string(data, "outcome").value_or("WAITING"));
            bind_text_or_null(upd_c.get(), 3, opt_json_string(data, "result_reason"));
            bind_text(upd_c.get(), 4, pipe_color);
            bind_text_or_null(upd_c.get(), 5, opt_json_string(data, "operator_anon_id"));
            bind_text_or_null(upd_c.get(), 6, opt_json_string(data, "model_version"));
            bind_text_or_null(upd_c.get(), 7, opt_json_string(data, "config_version"));
            bind_text(upd_c.get(), 8, opt_json_string(data, "fsm_state").value_or("WAIT_PART"));
            bind_text(upd_c.get(), 9, cycle_id);
            sqlite3_step(upd_c.get());
        }
    }

    // Resolve STATE_TRANSITION entries and paired frame_id
    std::vector<std::pair<json, std::optional<int64_t>>> state_transitions;
    if (run.transitions.is_array()) {
        for (std::size_t idx = 0; idx < run.transitions.size(); ++idx) {
            const auto& tr = run.transitions[idx];
            if (!tr.is_object() || tr.value("record_type", "") != "STATE_TRANSITION") continue;
            auto fid = opt_json_int64(tr, "frame_id");
            if (!fid.has_value() && tr.contains("event") && tr["event"].is_object()) {
                fid = opt_json_int64(tr["event"], "frame_id");
            }
            if (!fid.has_value()) {
                std::optional<std::string> tr_kind;
                if (tr.contains("event") && tr["event"].is_object()) {
                    tr_kind = opt_json_string(tr["event"], "kind");
                }
                auto tr_now = opt_json_double(tr, "now_monotonic_s");
                for (std::size_t j = idx + 1; j < run.transitions.size(); ++j) {
                    const auto& later = run.transitions[j];
                    if (!later.is_object() || later.value("record_type", "") != "EVENT") continue;
                    const json later_ev = (later.contains("event") && later["event"].is_object())
                                              ? later["event"]
                                              : json::object();
                    if (tr_kind.has_value() && opt_json_string(later_ev, "kind") != tr_kind) continue;
                    if (tr_now.has_value() && opt_json_double(later, "now_monotonic_s") != tr_now) continue;
                    fid = opt_json_int64(later_ev, "frame_id");
                    break;
                }
            }
            state_transitions.emplace_back(tr, fid);
        }
    }

    std::optional<std::string> last_event_id;
    const std::size_t n_tr = std::max<std::size_t>(1, state_transitions.size());
    for (std::size_t idx = 0; idx < state_transitions.size(); ++idx) {
        const auto& [tr, tr_fid] = state_transitions[idx];
        const double back_s = std::max(0.01, elapsed_s * (1.0 - static_cast<double>(idx + 1) / static_cast<double>(n_tr)));
        const std::string ev_time = utc_iso_offset_ms(-static_cast<int64_t>(back_s * 1000.0));

        std::string ev_type = opt_json_string(tr, "reason").value_or("");
        if (ev_type.empty() && tr.contains("event") && tr["event"].is_object()) {
            ev_type = opt_json_string(tr["event"], "kind").value_or("STATE_TRANSITION");
        }
        if (ev_type.empty()) ev_type = "STATE_TRANSITION";

        json tr_payload = tr;
        tr_payload["frame_id"] = tr_fid.has_value() ? json(*tr_fid) : json(nullptr);

        last_event_id = insert_event_unlocked(
            cycle_id,
            ev_type,
            tr_payload,
            opt_json_string(tr, "state_before"),
            opt_json_string(tr, "state_after"),
            opt_json_string(tr, "outcome"),
            opt_json_string(tr, "reason"),
            opt_json_double(tr, "capture_monotonic_s"),
            tr_fid,
            std::nullopt,
            ev_time
        );
    }

    std::optional<ExclusiveEvidenceFileGuard> ev_file_guard;
    bool existing_ev = false;
    {
        StmtGuard chk_ev(db_, "SELECT evidence_id FROM evidence WHERE cycle_id=? AND sha256=? LIMIT 1;");
        bind_text(chk_ev.get(), 1, cycle_id);
        bind_text(chk_ev.get(), 2, sha);
        existing_ev = (sqlite3_step(chk_ev.get()) == SQLITE_ROW);
    }

    if (!existing_ev) {
        const std::string evidence_id = random_token_hex(16);
        const std::string relpath = "evidence/" + cycle_id + "/" + evidence_id + ".jpg";
        const auto abs_path = data_root_ / path_from_utf8(relpath);
        ev_file_guard.emplace(evidence_root_, abs_path, body.data(), body.size());

        const auto cw = opt_json_int64(data, "camera_width");
        const auto ch = opt_json_int64(data, "camera_height");
        const std::string res_str = (cw.has_value() && ch.has_value())
                                        ? (std::to_string(*cw) + "x" + std::to_string(*ch))
                                        : "1280x720";

        json metadata{
            {"pipe_color", pipe_color},
            {"operation_event", "SCENARIO_" + to_upper_ascii(requested_scenario)},
            {"operator_anon_id", data.value("operator_anon_id", json(nullptr))},
            {"camera_metadata", json{
                {"camera_device_id", data.value("camera_device_id", json(nullptr))},
                {"resolution", res_str},
                {"fps", data.value("fps", json(nullptr))}
            }},
            {"label_status", data.value("label_status", json(nullptr))},
            {"captured_at_utc", updated_utc}
        };

        if (!last_event_id.has_value()) {
            last_event_id = insert_event_unlocked(
                cycle_id,
                "EVIDENCE_" + kind,
                metadata,
                std::string("WAIT_PART"),
                opt_json_string(data, "fsm_state"),
                opt_json_string(data, "outcome"),
                opt_json_string(data, "result_reason"),
                opt_json_double(data, "capture_monotonic_s"),
                opt_json_int64(data, "frame_id"),
                std::nullopt,
                updated_utc
            );
        }

        const std::string meta_str = metadata.dump();
        StmtGuard ins_evd(db_, "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?);");
        bind_text(ins_evd.get(), 1, evidence_id);
        bind_text(ins_evd.get(), 2, cycle_id);
        bind_text(ins_evd.get(), 3, *last_event_id);
        bind_text(ins_evd.get(), 4, updated_utc);
        bind_text(ins_evd.get(), 5, kind);
        bind_text(ins_evd.get(), 6, relpath);
        bind_text(ins_evd.get(), 7, "image/jpeg");
        bind_text(ins_evd.get(), 8, sha);
        sqlite3_bind_int64(ins_evd.get(), 9, opt_json_int64(data, "frame_id").value_or(1));
        bind_double_or_null(ins_evd.get(), 10, opt_json_double(data, "capture_monotonic_s"));
        bind_text(ins_evd.get(), 11, meta_str);
        if (sqlite3_step(ins_evd.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to insert scenario evidence: ") + sqlite3_errmsg(db_));
        }
    }

    {
        const std::string dumped = data.dump();
        StmtGuard upd_st(db_, "UPDATE station_state SET updated_at_utc=?, state_json=? WHERE singleton_id=1;");
        bind_text(upd_st.get(), 1, updated_utc);
        bind_text(upd_st.get(), 2, dumped);
        sqlite3_step(upd_st.get());
    }

    tx.commit();
    if (ev_file_guard.has_value()) ev_file_guard->commit();
    return attach_latest_evidence_and_summary_unlocked(std::move(data));
}

json StationDatabase::record_runtime_snapshot(const json& validated_snapshot) {
    json data = validated_snapshot.is_object() ? validated_snapshot : json::object();
    const std::string now = utc_now_iso();
    data["updated_at_utc"] = now;
    data["state_source"] = "station";

    std::optional<std::string> cycle_id = opt_json_string(data, "cycle_id");
    if (cycle_id.has_value() && cycle_id->empty()) cycle_id = std::nullopt;
    if (cycle_id.has_value()) {
        validate_cycle_id(*cycle_id);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    TransactionGuard tx(db_);

    json previous = default_station_state();
    {
        StmtGuard old_st(db_, "SELECT state_json FROM station_state WHERE singleton_id=1;");
        if (sqlite3_step(old_st.get()) == SQLITE_ROW) {
            try {
                previous = json::parse(col_text(old_st.get(), 0));
            } catch (...) {
            }
        }
    }
    const auto before_fsm = opt_json_string(previous, "fsm_state");

    {
        const std::string dumped = data.dump();
        StmtGuard upd_st(db_, "UPDATE station_state SET updated_at_utc=?, state_json=? WHERE singleton_id=1;");
        bind_text(upd_st.get(), 1, now);
        bind_text(upd_st.get(), 2, dumped);
        sqlite3_step(upd_st.get());
    }

    if (cycle_id.has_value()) {
        bool exists = false;
        {
            StmtGuard chk(db_, "SELECT 1 FROM cycles WHERE cycle_id=?;");
            bind_text(chk.get(), 1, *cycle_id);
            exists = (sqlite3_step(chk.get()) == SQLITE_ROW);
        }
        const std::string outcome = opt_json_string(data, "outcome").value_or("WAITING");
        const std::string fsm_state = opt_json_string(data, "fsm_state").value_or("WAIT_PART");
        if (!exists) {
            StmtGuard ins_c(
                db_,
                "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,"
                "result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state,source) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,'station');"
            );
            bind_text(ins_c.get(), 1, *cycle_id);
            bind_text(ins_c.get(), 2, now);
            bind_text(ins_c.get(), 3, now);
            bind_text_or_null(ins_c.get(), 4, opt_json_string(data, "label_status"));
            bind_text(ins_c.get(), 5, outcome);
            bind_text_or_null(ins_c.get(), 6, opt_json_string(data, "result_reason"));
            bind_text_or_null(ins_c.get(), 7, opt_json_string(data, "pipe_color"));
            bind_text_or_null(ins_c.get(), 8, opt_json_string(data, "operator_anon_id"));
            bind_text_or_null(ins_c.get(), 9, opt_json_string(data, "model_version"));
            bind_text_or_null(ins_c.get(), 10, opt_json_string(data, "config_version"));
            bind_text(ins_c.get(), 11, fsm_state);
            sqlite3_step(ins_c.get());
        } else {
            StmtGuard upd_c(
                db_,
                "UPDATE cycles SET updated_at_utc=?,final_outcome=?,result_reason=?,pipe_color=?,"
                "operator_anon_id=?,model_version=?,config_version=?,fsm_state=? WHERE cycle_id=?;"
            );
            bind_text(upd_c.get(), 1, now);
            bind_text(upd_c.get(), 2, outcome);
            bind_text_or_null(upd_c.get(), 3, opt_json_string(data, "result_reason"));
            bind_text_or_null(upd_c.get(), 4, opt_json_string(data, "pipe_color"));
            bind_text_or_null(upd_c.get(), 5, opt_json_string(data, "operator_anon_id"));
            bind_text_or_null(upd_c.get(), 6, opt_json_string(data, "model_version"));
            bind_text_or_null(upd_c.get(), 7, opt_json_string(data, "config_version"));
            bind_text(upd_c.get(), 8, fsm_state);
            bind_text(upd_c.get(), 9, *cycle_id);
            sqlite3_step(upd_c.get());
        }

        bool changed = false;
        for (const char* k : {"fsm_state", "outcome", "label_status", "view_valid", "hand_contact_candidate", "result_reason"}) {
            if (previous.value(k, json(nullptr)) != data.value(k, json(nullptr))) {
                changed = true;
                break;
            }
        }
        const auto explicit_ev_type = opt_json_string(data, "event_type");
        if (changed || (explicit_ev_type.has_value() && !explicit_ev_type->empty())) {
            insert_event_unlocked(
                cycle_id,
                explicit_ev_type.value_or("STATE_CHANGE"),
                data,
                before_fsm,
                fsm_state,
                outcome,
                opt_json_string(data, "result_reason"),
                opt_json_double(data, "capture_monotonic_s"),
                opt_json_int64(data, "frame_id"),
                std::nullopt,
                now
            );
        }
    }

    tx.commit();
    return data;
}

json StationDatabase::upload_evidence_bytes(
    const std::string& raw_cycle_id,
    const std::string& kind,
    const std::vector<uint8_t>& body,
    const std::string& mime_type,
    int64_t frame_id,
    std::optional<double> capture_monotonic_s,
    const std::optional<std::string>& pipe_color,
    const std::optional<std::string>& operation_event,
    const std::optional<std::string>& operator_anon_id,
    const json& camera_metadata,
    const std::optional<std::string>& label_status,
    const std::string& captured_utc_iso
) {
    const std::string cycle_id = validate_cycle_id(raw_cycle_id);
    const std::string suffix = (mime_type == "image/png") ? ".png" : ".jpg";
    const std::string evidence_id = random_token_hex(16);
    const std::string sha = sha256_hex(body);
    const std::string relpath = "evidence/" + cycle_id + "/" + evidence_id + suffix;
    const auto abs_path = data_root_ / path_from_utf8(relpath);

    json metadata{
        {"pipe_color", pipe_color.has_value() ? json(*pipe_color) : json(nullptr)},
        {"operation_event", operation_event.has_value() ? json(*operation_event) : json(nullptr)},
        {"operator_anon_id", operator_anon_id.has_value() ? json(*operator_anon_id) : json(nullptr)},
        {"camera_metadata", camera_metadata},
        {"label_status", label_status.has_value() ? json(*label_status) : json(nullptr)},
        {"captured_at_utc", captured_utc_iso}
    };

    std::lock_guard<std::mutex> lock(mutex_);
    ExclusiveEvidenceFileGuard file_guard(evidence_root_, abs_path, body.data(), body.size());
    TransactionGuard tx(db_);

    {
        StmtGuard chk(db_, "SELECT 1 FROM cycles WHERE cycle_id=?;");
        bind_text(chk.get(), 1, cycle_id);
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            StmtGuard ins_c(
                db_,
                "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,final_outcome,fsm_state,source) "
                "VALUES(?,?,?,'WAITING','WAIT_PART','station');"
            );
            bind_text(ins_c.get(), 1, cycle_id);
            bind_text(ins_c.get(), 2, captured_utc_iso);
            bind_text(ins_c.get(), 3, captured_utc_iso);
            sqlite3_step(ins_c.get());
        }
    }

    const std::string ev_id = insert_event_unlocked(
        cycle_id,
        "EVIDENCE_" + kind,
        metadata,
        std::nullopt,
        std::nullopt,
        std::string("WAITING"),
        operation_event,
        capture_monotonic_s,
        frame_id,
        std::nullopt,
        captured_utc_iso
    );

    {
        const std::string meta_str = metadata.dump();
        StmtGuard ins_evd(db_, "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?);");
        bind_text(ins_evd.get(), 1, evidence_id);
        bind_text(ins_evd.get(), 2, cycle_id);
        bind_text(ins_evd.get(), 3, ev_id);
        bind_text(ins_evd.get(), 4, captured_utc_iso);
        bind_text(ins_evd.get(), 5, kind);
        bind_text(ins_evd.get(), 6, relpath);
        bind_text(ins_evd.get(), 7, mime_type);
        bind_text(ins_evd.get(), 8, sha);
        sqlite3_bind_int64(ins_evd.get(), 9, frame_id);
        bind_double_or_null(ins_evd.get(), 10, capture_monotonic_s);
        bind_text(ins_evd.get(), 11, meta_str);
        if (sqlite3_step(ins_evd.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to insert uploaded evidence: ") + sqlite3_errmsg(db_));
        }
    }

    if (kind == "initial_label" && label_status.has_value() && !label_status->empty()) {
        StmtGuard upd(db_, "UPDATE cycles SET initial_label=COALESCE(initial_label,?) WHERE cycle_id=?;");
        bind_text(upd.get(), 1, *label_status);
        bind_text(upd.get(), 2, cycle_id);
        sqlite3_step(upd.get());
    }
    if (pipe_color.has_value() || operator_anon_id.has_value()) {
        StmtGuard upd(
            db_,
            "UPDATE cycles SET updated_at_utc=?, pipe_color=COALESCE(?,pipe_color), "
            "operator_anon_id=COALESCE(?,operator_anon_id) WHERE cycle_id=?;"
        );
        bind_text(upd.get(), 1, captured_utc_iso);
        bind_text_or_null(upd.get(), 2, pipe_color);
        bind_text_or_null(upd.get(), 3, operator_anon_id);
        bind_text(upd.get(), 4, cycle_id);
        sqlite3_step(upd.get());
    }

    tx.commit();
    file_guard.commit();

    return json{
        {"evidence_id", evidence_id},
        {"cycle_id", cycle_id},
        {"kind", kind},
        {"sha256", sha},
        {"frame_id", frame_id},
        {"capture_monotonic_s", capture_monotonic_s.has_value() ? json(*capture_monotonic_s) : json(nullptr)}
    };
}

json StationDatabase::archive_dataset_sample_evidence(
    const std::vector<uint8_t>& frame_jpeg,
    const json& sample_info,
    const std::string& label,
    const std::string& pipe_color,
    const json& live_state,
    const std::string& fallback_cycle_id
) {
    const std::string raw_cid = opt_json_string(live_state, "cycle_id").value_or(fallback_cycle_id);
    const std::string cycle_id = validate_cycle_id(raw_cid.empty() ? fallback_cycle_id : raw_cid);
    const std::string now = utc_now_iso();
    const std::string evidence_id = random_token_hex(16);
    const std::string sha = sha256_hex(frame_jpeg);
    const std::string relpath = "evidence/" + cycle_id + "/" + evidence_id + ".jpg";
    const auto abs_path = data_root_ / path_from_utf8(relpath);

    std::string ev_kind = "initial_label";
    if (label == "crossed_out") ev_kind = "A";
    else if (label == "covered") ev_kind = "B";
    else if (label == "absent") ev_kind = "C";
    else if (label == "unknown_challenge") ev_kind = "anomaly";

    const std::string lbl_status = (label == "unknown_challenge") ? "unknown" : label;
    const auto cw = opt_json_int64(live_state, "camera_width");
    const auto ch = opt_json_int64(live_state, "camera_height");
    const std::string res_str = (cw.has_value() && ch.has_value())
                                    ? (std::to_string(*cw) + "x" + std::to_string(*ch))
                                    : "1280x720";

    json meta{
        {"pipe_color", pipe_color},
        {"operation_event", "DATASET_SAMPLE_" + to_upper_ascii(label)},
        {"operator_anon_id", "OP_STATION_01"},
        {"sample_id", sample_info.value("sample_id", json(nullptr))},
        {"physical_tube_id", sample_info.value("physical_tube_id", json(nullptr))},
        {"batch_id", sample_info.value("batch_id", json(nullptr))},
        {"camera_metadata", json{
            {"camera_name", live_state.value("camera_name", json(nullptr))},
            {"camera_device_id", live_state.value("camera_device_id", json(nullptr))},
            {"resolution", res_str},
            {"fps", live_state.value("fps", json(nullptr))}
        }},
        {"label_status", lbl_status},
        {"captured_at_utc", now}
    };

    const auto frame_id = opt_json_int64(live_state, "frame_id");
    const auto cap_mono = opt_json_double(live_state, "capture_monotonic_s");
    const auto fsm_state = opt_json_string(live_state, "fsm_state");
    const auto outcome = opt_json_string(live_state, "outcome");

    std::lock_guard<std::mutex> lock(mutex_);
    ExclusiveEvidenceFileGuard file_guard(evidence_root_, abs_path, frame_jpeg.data(), frame_jpeg.size());
    TransactionGuard tx(db_);

    {
        StmtGuard chk(db_, "SELECT 1 FROM cycles WHERE cycle_id=?;");
        bind_text(chk.get(), 1, cycle_id);
        if (sqlite3_step(chk.get()) != SQLITE_ROW) {
            StmtGuard ins_c(
                db_,
                "INSERT INTO cycles(cycle_id,created_at_utc,updated_at_utc,initial_label,final_outcome,"
                "result_reason,pipe_color,operator_anon_id,model_version,config_version,fsm_state,source) "
                "VALUES(?,?,?,?,?,?,?,'OP_STATION_01',?,?,?,'live');"
            );
            bind_text(ins_c.get(), 1, cycle_id);
            bind_text(ins_c.get(), 2, now);
            bind_text(ins_c.get(), 3, now);
            bind_text(ins_c.get(), 4, lbl_status);
            bind_text(ins_c.get(), 5, outcome.value_or("WAITING"));
            bind_text_or_null(ins_c.get(), 6, opt_json_string(live_state, "result_reason"));
            bind_text(ins_c.get(), 7, pipe_color);
            bind_text_or_null(ins_c.get(), 8, opt_json_string(live_state, "model_version"));
            bind_text_or_null(ins_c.get(), 9, opt_json_string(live_state, "config_version"));
            bind_text(ins_c.get(), 10, fsm_state.value_or("INSPECTING"));
            sqlite3_step(ins_c.get());
        }
    }

    const std::string ev_id = insert_event_unlocked(
        cycle_id,
        "SAMPLE_" + to_upper_ascii(label),
        meta,
        fsm_state,
        fsm_state,
        outcome,
        "采集训练样本 (" + label + ")",
        cap_mono,
        frame_id,
        std::nullopt,
        now
    );

    {
        const std::string meta_str = meta.dump();
        StmtGuard ins_evd(db_, "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?);");
        bind_text(ins_evd.get(), 1, evidence_id);
        bind_text(ins_evd.get(), 2, cycle_id);
        bind_text(ins_evd.get(), 3, ev_id);
        bind_text(ins_evd.get(), 4, now);
        bind_text(ins_evd.get(), 5, ev_kind);
        bind_text(ins_evd.get(), 6, relpath);
        bind_text(ins_evd.get(), 7, "image/jpeg");
        bind_text(ins_evd.get(), 8, sha);
        sqlite3_bind_int64(ins_evd.get(), 9, frame_id.value_or(1));
        bind_double_or_null(ins_evd.get(), 10, cap_mono);
        bind_text(ins_evd.get(), 11, meta_str);
        if (sqlite3_step(ins_evd.get()) != SQLITE_DONE) {
            throw std::runtime_error(std::string("Failed to archive sample evidence: ") + sqlite3_errmsg(db_));
        }
    }

    tx.commit();
    file_guard.commit();

    return json{
        {"evidence_archived", true},
        {"evidence_id", evidence_id},
        {"evidence_url", "/api/evidence/" + evidence_id + "/file"},
        {"evidence_error", nullptr}
    };
}

json StationDatabase::list_evidence(int limit) const {
    const int clamped = std::clamp(limit, 1, 500);
    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard stmt(
        db_,
        "SELECT evidence_id, cycle_id, created_at_utc, kind, mime_type, sha256, "
        "frame_id, capture_monotonic_s, metadata_json "
        "FROM evidence ORDER BY created_at_utc DESC LIMIT ?;"
    );
    sqlite3_bind_int(stmt.get(), 1, clamped);
    json arr = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        const std::string eid = col_text(stmt.get(), 0);
        json meta = json::object();
        try {
            meta = json::parse(col_text(stmt.get(), 8));
        } catch (...) {
        }
        arr.push_back(json{
            {"evidence_id", eid},
            {"cycle_id", col_text(stmt.get(), 1)},
            {"created_at_utc", col_text(stmt.get(), 2)},
            {"kind", col_text(stmt.get(), 3)},
            {"mime_type", col_text(stmt.get(), 4)},
            {"sha256", col_text(stmt.get(), 5)},
            {"frame_id", col_int64_or_null_json(stmt.get(), 6)},
            {"capture_monotonic_s", col_double_or_null_json(stmt.get(), 7)},
            {"metadata", std::move(meta)},
            {"url", "/api/evidence/" + eid + "/file"}
        });
    }
    return arr;
}

EvidenceFileResult StationDatabase::get_evidence_file(const std::string& evidence_id) const {
    static const std::regex kEvIdRe("^[0-9a-f]{32}$");
    if (!std::regex_match(evidence_id, kEvIdRe)) {
        throw DatabaseHttpError(404, "EVIDENCE_NOT_FOUND");
    }

    std::string relpath;
    std::string mime_type;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        StmtGuard stmt(db_, "SELECT file_relpath, mime_type FROM evidence WHERE evidence_id=?;");
        bind_text(stmt.get(), 1, evidence_id);
        if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
            throw DatabaseHttpError(404, "EVIDENCE_NOT_FOUND");
        }
        relpath = col_text(stmt.get(), 0);
        mime_type = col_text(stmt.get(), 1);
    }

    const auto full_path = data_root_ / path_from_utf8(relpath);
    std::error_code ec;
    if (!is_path_within(evidence_root_, full_path) || !std::filesystem::is_regular_file(full_path, ec)) {
        throw DatabaseHttpError(404, "EVIDENCE_NOT_FOUND");
    }
    return EvidenceFileResult{full_path, mime_type.empty() ? "image/jpeg" : mime_type};
}

json StationDatabase::delete_evidence(const std::string& evidence_id) {
    static const std::regex kEvIdRe("^[0-9a-f]{32}$");
    if (!std::regex_match(evidence_id, kEvIdRe)) {
        throw DatabaseHttpError(404, "EVIDENCE_NOT_FOUND");
    }

    std::string cycle_id;
    std::string relpath;
    std::lock_guard<std::mutex> lock(mutex_);
    {
        TransactionGuard tx(db_);
        {
            StmtGuard sel(db_, "SELECT cycle_id, file_relpath FROM evidence WHERE evidence_id=?;");
            bind_text(sel.get(), 1, evidence_id);
            if (sqlite3_step(sel.get()) != SQLITE_ROW) {
                throw DatabaseHttpError(404, "EVIDENCE_NOT_FOUND");
            }
            cycle_id = col_text(sel.get(), 0);
            relpath = col_text(sel.get(), 1);
        }
        {
            StmtGuard del(db_, "DELETE FROM evidence WHERE evidence_id=?;");
            bind_text(del.get(), 1, evidence_id);
            sqlite3_step(del.get());
        }
        tx.commit();
    }

    int freed = safe_unlink_evidence_file(relpath) ? 1 : 0;
    return json{
        {"deleted", true},
        {"evidence_id", evidence_id},
        {"cycle_id", cycle_id},
        {"deleted_files", freed},
        {"freed_files", freed}
    };
}

json StationDatabase::register_model(
    const std::string& model_name,
    const std::string& model_version,
    const std::string& sha256_hex_str,
    const std::string& license_status,
    const std::string& notes,
    const std::string& registered_by
) {
    static const std::regex kVerRe("^[A-Za-z0-9._-]{1,64}$");
    static const std::regex kShaRe("^[0-9a-fA-F]{64}$");

    if (model_name.empty() || model_name.size() > 128) {
        throw DatabaseHttpError(422, "INVALID_MODEL_NAME");
    }
    if (model_version.empty() || model_version == "." || model_version == ".." ||
        model_version.find("..") != std::string::npos || !std::regex_match(model_version, kVerRe)) {
        throw DatabaseHttpError(422, "INVALID_MODEL_VERSION");
    }
    if (!std::regex_match(sha256_hex_str, kShaRe)) {
        throw DatabaseHttpError(422, "INVALID_MODEL_SHA256");
    }
    if (license_status != "reviewed" && license_status != "needs_review" && license_status != "unknown") {
        throw DatabaseHttpError(422, "INVALID_LICENSE_STATUS");
    }
    if (notes.size() > 2000) {
        throw DatabaseHttpError(422, "NOTES_TOO_LONG");
    }

    std::string norm_sha = sha256_hex_str;
    std::transform(norm_sha.begin(), norm_sha.end(), norm_sha.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    const std::string model_id = random_token_hex(16);
    const std::string now = utc_now_iso();

    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard ins(db_, "INSERT INTO model_registry VALUES(?,?,?,?,?,?,?,?);");
    bind_text(ins.get(), 1, model_id);
    bind_text(ins.get(), 2, model_name);
    bind_text(ins.get(), 3, model_version);
    bind_text(ins.get(), 4, norm_sha);
    bind_text(ins.get(), 5, license_status);
    bind_text(ins.get(), 6, notes);
    bind_text(ins.get(), 7, now);
    bind_text(ins.get(), 8, registered_by);
    if (sqlite3_step(ins.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("Failed to insert model_registry: ") + sqlite3_errmsg(db_));
    }

    return json{
        {"model_id", model_id},
        {"model_name", model_name},
        {"model_version", model_version},
        {"sha256", norm_sha},
        {"license_status", license_status},
        {"notes", notes}
    };
}

json StationDatabase::list_models(int limit) const {
    const int clamped = std::clamp(limit, 1, 200);
    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard stmt(
        db_,
        "SELECT model_id, model_name, model_version, sha256, license_status, notes, created_at_utc, registered_by "
        "FROM model_registry ORDER BY created_at_utc DESC LIMIT ?;"
    );
    sqlite3_bind_int(stmt.get(), 1, clamped);
    json arr = json::array();
    while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
        arr.push_back(json{
            {"model_id", col_text(stmt.get(), 0)},
            {"model_name", col_text(stmt.get(), 1)},
            {"model_version", col_text(stmt.get(), 2)},
            {"sha256", col_text(stmt.get(), 3)},
            {"license_status", col_text(stmt.get(), 4)},
            {"notes", col_text(stmt.get(), 5)},
            {"created_at_utc", col_text(stmt.get(), 6)},
            {"registered_by", col_text(stmt.get(), 7)}
        });
    }
    return arr;
}

std::optional<json> StationDatabase::get_model_by_id(const std::string& model_id) const {
    static const std::regex kIdRe("^[A-Za-z0-9_-]{1,80}$");
    if (!std::regex_match(model_id, kIdRe)) return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);
    StmtGuard stmt(
        db_,
        "SELECT model_id, model_name, model_version, sha256, license_status, notes, created_at_utc, registered_by "
        "FROM model_registry WHERE model_id=?;"
    );
    bind_text(stmt.get(), 1, model_id);
    if (sqlite3_step(stmt.get()) != SQLITE_ROW) return std::nullopt;
    return json{
        {"model_id", col_text(stmt.get(), 0)},
        {"model_name", col_text(stmt.get(), 1)},
        {"model_version", col_text(stmt.get(), 2)},
        {"sha256", col_text(stmt.get(), 3)},
        {"license_status", col_text(stmt.get(), 4)},
        {"notes", col_text(stmt.get(), 5)},
        {"created_at_utc", col_text(stmt.get(), 6)},
        {"registered_by", col_text(stmt.get(), 7)}
    };
}

json StationDatabase::delete_model(const std::string& model_id, const std::string& active_model_version) {
    static const std::regex kIdRe("^[A-Za-z0-9_-]{1,80}$");
    if (!std::regex_match(model_id, kIdRe)) {
        throw DatabaseHttpError(404, "MODEL_NOT_FOUND");
    }

    std::string version;
    int64_t remaining_same_version = 0;

    std::lock_guard<std::mutex> lock(mutex_);
    {
        TransactionGuard tx(db_);
        {
            StmtGuard sel(db_, "SELECT model_version FROM model_registry WHERE model_id=?;");
            bind_text(sel.get(), 1, model_id);
            if (sqlite3_step(sel.get()) != SQLITE_ROW) {
                throw DatabaseHttpError(404, "MODEL_NOT_FOUND");
            }
            version = col_text(sel.get(), 0);
        }

        if (!active_model_version.empty() && version == active_model_version) {
            throw DatabaseHttpError(409, "不能删除当前正在实时运行的模型，请先切换其他模型");
        }

        {
            StmtGuard del(db_, "DELETE FROM model_registry WHERE model_id=?;");
            bind_text(del.get(), 1, model_id);
            sqlite3_step(del.get());
        }
        {
            StmtGuard cnt(db_, "SELECT COUNT(*) FROM model_registry WHERE model_version=?;");
            bind_text(cnt.get(), 1, version);
            if (sqlite3_step(cnt.get()) == SQLITE_ROW) {
                remaining_same_version = sqlite3_column_int64(cnt.get(), 0);
            }
        }
        tx.commit();
    }

    bool deleted_artifact_dir = false;
    if (remaining_same_version == 0) {
        if (auto run_dir = safe_model_run_dir(version); run_dir.has_value()) {
            std::error_code ec;
            if (std::filesystem::is_directory(*run_dir, ec)) {
                std::filesystem::remove_all(*run_dir, ec);
                deleted_artifact_dir = !ec;
            }
        }
    }

    return json{
        {"deleted", true},
        {"model_id", model_id},
        {"model_version", version},
        {"deleted_artifact_dir", deleted_artifact_dir}
    };
}

json StationDatabase::storage_overview(const json& training_stats, const std::string& active_model_version) const {
    std::error_code ec;
    const uint64_t db_bytes = std::filesystem::is_regular_file(db_path_, ec)
                                  ? std::filesystem::file_size(db_path_, ec)
                                  : 0;
    const auto [ev_files, ev_bytes] = directory_file_usage(evidence_root_);

    int64_t cycles_cnt = 0, events_cnt = 0, evidence_cnt = 0, config_cnt = 0, model_cnt = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto count_table = [&](const char* sql) -> int64_t {
            StmtGuard s(db_, sql);
            return sqlite3_step(s.get()) == SQLITE_ROW ? sqlite3_column_int64(s.get(), 0) : 0;
        };
        cycles_cnt = count_table("SELECT COUNT(*) FROM cycles;");
        events_cnt = count_table("SELECT COUNT(*) FROM events;");
        evidence_cnt = count_table("SELECT COUNT(*) FROM evidence;");
        config_cnt = count_table("SELECT COUNT(*) FROM configuration_history;");
        model_cnt = count_table("SELECT COUNT(*) FROM model_registry;");
    }

    const double db_size_kb = std::round((static_cast<double>(db_bytes) / 1024.0) * 100.0) / 100.0;
    const double ev_size_mb = std::round((static_cast<double>(ev_bytes) / (1024.0 * 1024.0)) * 1000.0) / 1000.0;

    json class_counts = json::object();
    if (training_stats.contains("class_counts")) {
        class_counts = training_stats["class_counts"];
    } else if (training_stats.contains("counts_by_class")) {
        class_counts = training_stats["counts_by_class"];
    }

    json active_ver_json = nullptr;
    if (!active_model_version.empty() && active_model_version != kHeuristicModelVersion) {
        active_ver_json = active_model_version;
    }

    return json{
        {"business_db", json{
            {"db_path", path_to_utf8(db_path_)},
            {"db_size_kb", db_size_kb},
            {"evidence_dir", path_to_utf8(evidence_root_)},
            {"evidence_files", ev_files},
            {"evidence_size_mb", ev_size_mb},
            {"tables", json{
                {"cycles", cycles_cnt},
                {"events", events_cnt},
                {"evidence", evidence_cnt},
                {"configuration_history", config_cnt},
                {"model_registry", model_cnt}
            }}
        }},
        {"training_dataset", json{
            {"dataset_dir", training_stats.value("dataset_dir", path_to_utf8(data_root_ / "dataset"))},
            {"manifest_path", training_stats.value("manifest_path", path_to_utf8(data_root_ / "dataset" / "manifest.csv"))},
            {"total_samples", training_stats.value("total_samples", 0)},
            {"class_counts", class_counts},
            {"independent_tube_groups", training_stats.value("independent_tube_groups", 0)},
            {"coco_annotations", training_stats.value("coco_annotations", 0)},
            {"dataset_size_mb", training_stats.value("dataset_size_mb", 0.0)}
        }},
        {"model_artifacts", json{
            {"runs_dir", training_stats.value("runs_dir", path_to_utf8(data_root_ / "runs" / "label_classifier"))},
            {"run_count", training_stats.value("run_count", 0)},
            {"models_size_mb", training_stats.value("models_size_mb", 0.0)},
            {"active_model_version", active_ver_json}
        }}
    };
}

}  // namespace igbt
