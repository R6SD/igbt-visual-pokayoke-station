#include "igbt/http_server.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <regex>
#include <set>
#include <sstream>

namespace igbt {

namespace {

constexpr std::size_t kMaxEvidenceUploadBytes = 10 * 1024 * 1024;
constexpr std::size_t kMaxOnnxUploadBytes = 64 * 1024 * 1024;
constexpr int kMaxEvidenceImageWidth = 8192;
constexpr int kMaxEvidenceImageHeight = 8192;
constexpr int64_t kMaxEvidenceImagePixels = 33'554'432;
constexpr double kStreamNoFrameTimeoutS = 3.0;
constexpr double kStreamOfflineTimeoutS = 0.25;

std::string read_process_env_utf8(const std::string& key) {
    if (key.empty()) return {};
    std::wstring wkey = path_from_utf8(key).wstring();
    if (const wchar_t* wval = _wgetenv(wkey.c_str()); wval != nullptr && wval[0] != L'\0') {
        return path_to_utf8(std::filesystem::path(wval));
    }
    DWORD needed = GetEnvironmentVariableW(wkey.c_str(), nullptr, 0);
    if (needed > 1) {
        std::wstring buf(needed, L'\0');
        DWORD copied = GetEnvironmentVariableW(wkey.c_str(), buf.data(), needed);
        if (copied > 0) {
            buf.resize(copied);
            return path_to_utf8(std::filesystem::path(buf));
        }
    }
    return {};
}

void send_json(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_header("Cache-Control", "no-store");
    res.set_content(body.dump(), "application/json; charset=utf-8");
}

void send_error(httplib::Response& res, int status, const json& detail) {
    send_json(res, status, json{{"detail", detail}});
}

void send_error(httplib::Response& res, int status, const std::string& detail) {
    send_json(res, status, json{{"detail", detail}});
}

void send_error(httplib::Response& res, int status, const char* detail) {
    send_json(res, status, json{{"detail", detail ? detail : ""}});
}

std::optional<std::string> get_opt_header(const httplib::Request& req, const char* key) {
    if (!req.has_header(key)) return std::nullopt;
    return req.get_header_value(key);
}

int get_int_query(const httplib::Request& req, const char* key, int default_val) {
    if (!req.has_param(key)) return default_val;
    try {
        std::string s = req.get_param_value(key);
        std::size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return default_val;
        return v;
    } catch (...) {
        return default_val;
    }
}

json parse_json_body_or_throw(const httplib::Request& req) {
    if (req.body.empty()) {
        throw DatabaseHttpError(422, "EMPTY_JSON_BODY");
    }
    try {
        return json::parse(req.body);
    } catch (...) {
        throw DatabaseHttpError(422, "INVALID_JSON_BODY");
    }
}

std::string url_decode(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '%' && i + 2 < in.size() &&
            std::isxdigit(static_cast<unsigned char>(in[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(in[i + 2]))) {
            int hi = std::isdigit(static_cast<unsigned char>(in[i + 1]))
                         ? (in[i + 1] - '0')
                         : (std::tolower(static_cast<unsigned char>(in[i + 1])) - 'a' + 10);
            int lo = std::isdigit(static_cast<unsigned char>(in[i + 2]))
                         ? (in[i + 2] - '0')
                         : (std::tolower(static_cast<unsigned char>(in[i + 2])) - 'a' + 10);
            out.push_back(static_cast<char>((hi << 4) | lo));
            i += 2;
        } else {
            out.push_back(in[i]);
        }
    }
    return out;
}

std::optional<std::string> clean_optional_header(const std::optional<std::string>& raw, std::size_t max_len = 64) {
    if (!raw.has_value()) return std::nullopt;
    std::string decoded = url_decode(*raw);
    std::string cleaned;
    cleaned.reserve(decoded.size());
    for (char c : decoded) {
        if (c != '\r' && c != '\n' && c != '\t') cleaned.push_back(c);
    }
    const auto first = cleaned.find_first_not_of(' ');
    if (first == std::string::npos) return std::nullopt;
    const auto last = cleaned.find_last_not_of(' ');
    cleaned = cleaned.substr(first, last - first + 1);
    if (cleaned.empty()) return std::nullopt;
    if (cleaned.size() > max_len) cleaned.resize(max_len);
    return cleaned;
}

int parse_optional_input_size(const json& raw, int default_val = kDefaultInputSize) {
    if (raw.is_null() || raw.is_boolean()) return default_val;
    if (raw.is_number_integer()) {
        int v = raw.get<int>();
        return (v >= 32 && v <= 1024) ? v : default_val;
    }
    if (raw.is_string()) {
        try {
            int v = std::stoi(raw.get<std::string>());
            return (v >= 32 && v <= 1024) ? v : default_val;
        } catch (...) {
            return default_val;
        }
    }
    return default_val;
}

std::optional<Box> validate_explicit_crop_box(const json& box_json) {
    if (box_json.is_null()) return std::nullopt;
    if (!box_json.is_array() || box_json.size() != 4) {
        throw DatabaseHttpError(422, "INVALID_CROP_BOX: box must have 4 normalized coordinates [x1, y1, x2, y2]");
    }
    Box coords{};
    for (std::size_t i = 0; i < 4; ++i) {
        if (!box_json[i].is_number() || box_json[i].is_boolean()) {
            throw DatabaseHttpError(422, "INVALID_CROP_BOX: coordinates must be finite numbers");
        }
        coords[i] = box_json[i].get<double>();
        if (!std::isfinite(coords[i])) {
            throw DatabaseHttpError(422, "INVALID_CROP_BOX: coordinates must be finite");
        }
    }
    if (!(0.0 <= coords[0] && coords[0] < coords[2] && coords[2] <= 1.0 &&
          0.0 <= coords[1] && coords[1] < coords[3] && coords[3] <= 1.0)) {
        throw DatabaseHttpError(422, "INVALID_CROP_BOX: expected 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1");
    }
    return coords;
}

bool is_all_digits(const std::string& s) {
    if (s.empty()) return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

// Validate ISO-8601 timestamp with explicit timezone ('Z' or +/-HH:MM) and return normalized UTC string
std::string validate_iso_timestamp_with_tz(const std::string& raw) {
    if (raw.size() < 20 || raw.size() > 40) {
        throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
    }
    int y = 0, m = 0, d = 0, hh = 0, mm = 0, ss = 0;
    if (std::sscanf(raw.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &m, &d, &hh, &mm, &ss) != 6) {
        throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
    }
    if (m < 1 || m > 12 || d < 1 || d > 31 || hh < 0 || hh > 23 || mm < 0 || mm > 59 || ss < 0 || ss > 60) {
        throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
    }
    const bool ends_z = (raw.back() == 'Z' || raw.back() == 'z');
    const std::size_t tz_pos = raw.find_first_of("+-", 19);
    if (!ends_z && tz_pos == std::string::npos) {
        throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
    }

    std::tm tm_val{};
    tm_val.tm_year = y - 1900;
    tm_val.tm_mon = m - 1;
    tm_val.tm_mday = d;
    tm_val.tm_hour = hh;
    tm_val.tm_min = mm;
    tm_val.tm_sec = ss;
    std::time_t epoch_s = _mkgmtime(&tm_val);
    if (epoch_s < 0) {
        throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
    }

    int ms = 0;
    std::size_t dot = raw.find('.', 18);
    if (dot != std::string::npos) {
        std::size_t i = dot + 1;
        int digits = 0;
        while (i < raw.size() && std::isdigit(static_cast<unsigned char>(raw[i])) && digits < 3) {
            ms = ms * 10 + (raw[i] - '0');
            ++i;
            ++digits;
        }
        while (digits > 0 && digits < 3) {
            ms *= 10;
            ++digits;
        }
    }

    int64_t offset_s = 0;
    if (!ends_z && tz_pos != std::string::npos) {
        int tz_h = 0, tz_m = 0;
        if (std::sscanf(raw.c_str() + tz_pos + 1, "%2d:%2d", &tz_h, &tz_m) != 2 ||
            tz_h < 0 || tz_h > 23 || tz_m < 0 || tz_m > 59) {
            throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
        }
        offset_s = static_cast<int64_t>(tz_h * 3600 + tz_m * 60);
        if (raw[tz_pos] == '+') offset_s = -offset_s;
    }

    std::time_t utc_s = static_cast<std::time_t>(static_cast<int64_t>(epoch_s) + offset_s);
    std::tm gmt{};
    gmtime_s(&gmt, &utc_s);
    std::ostringstream oss;
    oss << std::put_time(&gmt, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setw(3) << std::setfill('0') << ms << "Z";
    return oss.str();
}

void validate_and_decode_evidence_image(const std::vector<uint8_t>& body, const std::string& mime) {
    const bool is_jpeg_magic = body.size() >= 4 &&
                               body[0] == 0xFF && body[1] == 0xD8 &&
                               body[body.size() - 2] == 0xFF && body[body.size() - 1] == 0xD9;
    const bool is_png_magic = body.size() >= 8 &&
                              body[0] == 0x89 && body[1] == 'P' && body[2] == 'N' && body[3] == 'G' &&
                              body[4] == '\r' && body[5] == '\n' && body[6] == 0x1A && body[7] == '\n';

    if (!((mime == "image/jpeg" && is_jpeg_magic) || (mime == "image/png" && is_png_magic))) {
        throw DatabaseHttpError(415, "ONLY_VALID_JPEG_OR_PNG_EVIDENCE_IS_ACCEPTED");
    }

    cv::Mat decoded = cv::imdecode(body, cv::IMREAD_COLOR);
    if (decoded.empty()) {
        throw DatabaseHttpError(422, "INVALID_OR_UNDECODABLE_EVIDENCE_IMAGE");
    }
    const int64_t w = decoded.cols;
    const int64_t h = decoded.rows;
    if (w < 1 || h < 1 || w > kMaxEvidenceImageWidth || h > kMaxEvidenceImageHeight ||
        (w * h) > kMaxEvidenceImagePixels) {
        throw DatabaseHttpError(422, "EVIDENCE_IMAGE_DIMENSIONS_OUT_OF_BOUNDS");
    }
}

json validate_runtime_snapshot_payload(const json& payload) {
    if (!payload.is_object()) {
        throw DatabaseHttpError(422, "INVALID_SNAPSHOT_PAYLOAD");
    }
    static const std::set<std::string> kAllowed = {
        "mode", "camera_status", "camera_name", "camera_device_id", "backend",
        "camera_width", "camera_height", "camera_settings_locked", "fps",
        "label_status", "label_confidence", "view_valid", "view_valid_reasons",
        "quality_ok", "laplacian_variance", "brightness", "overexposed_fraction",
        "underexposed_fraction", "left_grip_active", "right_grip_active",
        "label_contact_active", "hand_contact_candidate", "contact_evidence_type",
        "proves_physical_contact", "fsm_state", "outcome", "result_reason",
        "c_timer_ms", "cycle_id", "pipe_color", "operator_anon_id", "event_type",
        "frame_id", "capture_monotonic_s", "model_version", "config_version"
    };
    for (auto it = payload.begin(); it != payload.end(); ++it) {
        if (kAllowed.find(it.key()) == kAllowed.end()) {
            throw DatabaseHttpError(422, "EXTRA_FIELD_FORBIDDEN:" + it.key());
        }
    }

    if (!payload.contains("camera_status") || !payload["camera_status"].is_string()) {
        throw DatabaseHttpError(422, "MISSING_CAMERA_STATUS");
    }
    const std::string cam_status = payload["camera_status"].get<std::string>();
    static const std::set<std::string> kCamStatuses = {"ONLINE", "OFFLINE", "FAULT", "UNVERIFIED"};
    if (kCamStatuses.find(cam_status) == kCamStatuses.end()) {
        throw DatabaseHttpError(422, "INVALID_CAMERA_STATUS");
    }

    if (payload.contains("mode") && payload["mode"] != "shadow") {
        throw DatabaseHttpError(422, "ONLY_SHADOW_MODE_ALLOWED");
    }
    if (payload.contains("proves_physical_contact") && payload["proves_physical_contact"] != false) {
        throw DatabaseHttpError(422, "PROVES_PHYSICAL_CONTACT_MUST_BE_FALSE");
    }

    static const std::set<std::string> kLabels = {"complete", "crossed_out", "covered", "absent", "unknown"};
    std::string label_status = payload.value("label_status", "unknown");
    if (kLabels.find(label_status) == kLabels.end()) {
        throw DatabaseHttpError(422, "INVALID_LABEL_STATUS");
    }

    static const std::set<std::string> kFsmStates = {
        "BOOT", "WAIT_PART", "VERIFY_INITIAL", "READY", "INSPECTING", "PROCESSING",
        "VERIFY_C_ABSENT", "VIEW_INVALID", "PASS", "ALARM", "FAULT", "REVIEW_REQUIRED"
    };
    std::string fsm_state = payload.value("fsm_state", "WAIT_PART");
    if (kFsmStates.find(fsm_state) == kFsmStates.end()) {
        throw DatabaseHttpError(422, "INVALID_FSM_STATE");
    }

    static const std::set<std::string> kOutcomes = {"WAITING", "PASS", "ALARM", "FAULT", "VIEW_INVALID"};
    std::string outcome = payload.value("outcome", "WAITING");
    if (kOutcomes.find(outcome) == kOutcomes.end()) {
        throw DatabaseHttpError(422, "INVALID_OUTCOME");
    }

    int c_timer_ms = payload.value("c_timer_ms", 0);
    if (c_timer_ms < 0 || c_timer_ms > 1200) {
        throw DatabaseHttpError(422, "INVALID_C_TIMER_MS");
    }

    json out{
        {"mode", "shadow"},
        {"camera_status", cam_status},
        {"camera_name", payload.value("camera_name", json(nullptr))},
        {"camera_device_id", payload.value("camera_device_id", json(nullptr))},
        {"backend", payload.value("backend", json(nullptr))},
        {"camera_width", payload.value("camera_width", json(nullptr))},
        {"camera_height", payload.value("camera_height", json(nullptr))},
        {"camera_settings_locked", payload.value("camera_settings_locked", json(nullptr))},
        {"fps", payload.value("fps", json(nullptr))},
        {"label_status", label_status},
        {"label_confidence", payload.value("label_confidence", json(nullptr))},
        {"view_valid", payload.value("view_valid", json(nullptr))},
        {"view_valid_reasons", payload.value("view_valid_reasons", json::array())},
        {"quality_ok", payload.value("quality_ok", json(nullptr))},
        {"laplacian_variance", payload.value("laplacian_variance", json(nullptr))},
        {"brightness", payload.value("brightness", json(nullptr))},
        {"overexposed_fraction", payload.value("overexposed_fraction", json(nullptr))},
        {"underexposed_fraction", payload.value("underexposed_fraction", json(nullptr))},
        {"left_grip_active", payload.value("left_grip_active", json(nullptr))},
        {"right_grip_active", payload.value("right_grip_active", json(nullptr))},
        {"label_contact_active", payload.value("label_contact_active", json(nullptr))},
        {"hand_contact_candidate", payload.value("hand_contact_candidate", false)},
        {"contact_evidence_type", "2d_visual_proximity_proxy"},
        {"proves_physical_contact", false},
        {"fsm_state", fsm_state},
        {"outcome", outcome},
        {"result_reason", payload.value("result_reason", json(nullptr))},
        {"c_timer_ms", c_timer_ms},
        {"cycle_id", payload.value("cycle_id", json(nullptr))},
        {"pipe_color", payload.value("pipe_color", json(nullptr))},
        {"operator_anon_id", payload.value("operator_anon_id", json(nullptr))},
        {"event_type", payload.value("event_type", json(nullptr))},
        {"frame_id", payload.value("frame_id", json(nullptr))},
        {"capture_monotonic_s", payload.value("capture_monotonic_s", json(nullptr))},
        {"model_version", payload.value("model_version", json(nullptr))},
        {"config_version", payload.value("config_version", json(nullptr))}
    };
    return out;
}

std::string guess_mime_type(const std::filesystem::path& p) {
    const std::string ext = to_lower_ascii(p.extension().string());
    if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
    if (ext == ".js" || ext == ".mjs") return "application/javascript; charset=utf-8";
    if (ext == ".css") return "text/css; charset=utf-8";
    if (ext == ".json" || ext == ".map") return "application/json; charset=utf-8";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".webp") return "image/webp";
    if (ext == ".ico") return "image/x-icon";
    if (ext == ".woff2") return "font/woff2";
    if (ext == ".woff") return "font/woff";
    if (ext == ".ttf") return "font/ttf";
    return "application/octet-stream";
}

bool is_trusted_loopback_host(const std::string& host_header) {
    if (host_header.empty()) return true;
    std::string host = to_lower_ascii(host_header);
    if (auto colon = host.find(':'); colon != std::string::npos) {
        host = host.substr(0, colon);
    }
    return host == "127.0.0.1" || host == "localhost" || host == "testserver";
}

bool has_recent_crash_dump(const std::filesystem::path& data_dir) {
    std::error_code ec;
    const auto dumps_dir = data_dir / "dumps";
    if (!std::filesystem::is_directory(dumps_dir, ec)) return false;
    const auto now_ft = std::filesystem::file_time_type::clock::now();
    for (std::filesystem::directory_iterator it(dumps_dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        if (it->path().extension() != L".dmp") continue;
        const auto mtime = it->last_write_time(fec);
        if (!fec && (now_ft - mtime) <= std::chrono::minutes(10)) {
            return true;
        }
    }
    return false;
}

AdamModbusConfig resolve_adam_config(
    const StationServerOptions& options,
    const std::filesystem::path& data_dir,
    const EngineerSessionAuth& auth
) {
    if (options.adam_config.has_value()) {
        return *options.adam_config;
    }
    const auto cfg_path = data_dir / "hal_adam.json";
    std::error_code ec;
    if (std::filesystem::is_regular_file(cfg_path, ec)) {
        return load_adam_config_file(cfg_path);
    }
    const std::string env_host = auth.get_env_value("IGBT_ADAM_HOST");
    if (!env_host.empty()) {
        AdamModbusConfig cfg;
        cfg.enabled = true;
        cfg.host = env_host;
        const std::string env_port = auth.get_env_value("IGBT_ADAM_PORT");
        if (!env_port.empty()) {
            try {
                const int p = std::stoi(env_port);
                if (p >= 1 && p <= 65535) cfg.port = static_cast<uint16_t>(p);
            } catch (...) {
            }
        }
        return cfg;
    }
    return {};
}

}  // namespace

// ============================================================================
// EngineerSessionAuth
// ============================================================================

std::string EngineerSessionAuth::get_env_value(const std::string& key) const {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = env_overrides_.find(key);
        if (it != env_overrides_.end()) {
            return it->second.value_or("");
        }
    }
    return read_process_env_utf8(key);
}

void EngineerSessionAuth::set_env_override(const std::string& key, const std::optional<std::string>& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    env_overrides_[key] = value;
}

void EngineerSessionAuth::clear_env_overrides() {
    std::lock_guard<std::mutex> lock(mutex_);
    env_overrides_.clear();
}

void EngineerSessionAuth::prune_expired_locked(double now_mono) {
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (it->second <= now_mono) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

std::string EngineerSessionAuth::mint_session(int ttl_seconds) {
    const std::string token = random_token_hex(24);  // 48 lowercase hex chars
    const std::string digest = sha256_hex(token);
    const double now_mono = monotonic_seconds();

    std::lock_guard<std::mutex> lock(mutex_);
    prune_expired_locked(now_mono);
    if (sessions_.size() >= kMaxActiveSessions) {
        auto oldest = std::min_element(
            sessions_.begin(),
            sessions_.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        if (oldest != sessions_.end()) {
            sessions_.erase(oldest);
        }
    }
    sessions_[digest] = now_mono + static_cast<double>(std::max(1, ttl_seconds));
    return token;
}

bool EngineerSessionAuth::verify_session_token(const std::string& token) {
    if (token.empty()) return false;
    const std::string digest = sha256_hex(token);
    const double now_mono = monotonic_seconds();
    std::lock_guard<std::mutex> lock(mutex_);
    prune_expired_locked(now_mono);
    auto it = sessions_.find(digest);
    return it != sessions_.end() && it->second > now_mono;
}

void EngineerSessionAuth::clear_sessions() {
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.clear();
}

void EngineerSessionAuth::expire_all_sessions() {
    const double expired_ts = monotonic_seconds() - 1.0;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : sessions_) {
        kv.second = expired_ts;
    }
}

void EngineerSessionAuth::set_session_ttl_seconds(int ttl_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    ttl_seconds_ = std::max(1, ttl_seconds);
}

void EngineerSessionAuth::set_lockout_policy(int max_attempts, double lockout_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_failed_attempts_ = std::max(1, max_attempts);
    lockout_seconds_ = std::max(1.0, lockout_seconds);
}

void EngineerSessionAuth::reset_rate_limit() {
    std::lock_guard<std::mutex> lock(mutex_);
    failed_attempts_ = 0;
    lockout_until_mono_ = 0.0;
}

json EngineerSessionAuth::status_json() const {
    const bool pin_configured = !get_env_value("IGBT_ENGINEER_PIN").empty();
    const bool admin_token_configured = !get_env_value("IGBT_ADMIN_TOKEN").empty();
    return json{
        {"role_gate_enabled", true},
        {"pin_configured", pin_configured},
        {"admin_token_configured", admin_token_configured},
        {"auth_configured", pin_configured || admin_token_configured},
        {"default_pin_active", false}
    };
}

json EngineerSessionAuth::verify_passcode(const std::string& passcode) {
    if (passcode.size() > 256) {
        throw DatabaseHttpError(422, "PASSCODE_TOO_LONG");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (monotonic_seconds() < lockout_until_mono_) {
            throw DatabaseHttpError(429, "AUTH_RATE_LIMITED");
        }
    }
    const std::string configured_admin = get_env_value("IGBT_ADMIN_TOKEN");
    const std::string configured_pin = get_env_value("IGBT_ENGINEER_PIN");
    if (configured_admin.empty() && configured_pin.empty()) {
        throw DatabaseHttpError(401, "工程师认证未配置，请先设置 IGBT_ENGINEER_PIN 或 IGBT_ADMIN_TOKEN");
    }

    const bool matches_admin = !configured_admin.empty() && constant_time_equals(passcode, configured_admin);
    const bool matches_pin = !configured_pin.empty() && constant_time_equals(passcode, configured_pin);
    if (!(matches_admin || matches_pin)) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++failed_attempts_;
        if (max_failed_attempts_ > 0 && failed_attempts_ >= max_failed_attempts_) {
            lockout_until_mono_ = monotonic_seconds() + lockout_seconds_;
            throw DatabaseHttpError(429, "AUTH_RATE_LIMITED");
        }
        throw DatabaseHttpError(401, "工程师口令错误，请重试");
    }

    int ttl = kDefaultTtlSeconds;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_attempts_ = 0;
        lockout_until_mono_ = 0.0;
        ttl = ttl_seconds_;
    }
    const std::string session_token = mint_session(ttl);
    return json{
        {"authenticated", true},
        {"role", "engineer"},
        {"admin_token", session_token},
        {"expires_in_seconds", ttl}
    };
}

std::string EngineerSessionAuth::require_admin_token(const std::optional<std::string>& supplied) {
    if (supplied.has_value() && verify_session_token(*supplied)) {
        return "engineer-session";
    }
    const std::string expected_admin = get_env_value("IGBT_ADMIN_TOKEN");
    if (!expected_admin.empty() && supplied.has_value() && constant_time_equals(*supplied, expected_admin)) {
        return "local-admin";
    }
    const std::string expected_pin = get_env_value("IGBT_ENGINEER_PIN");
    if (expected_admin.empty() && expected_pin.empty()) {
        throw DatabaseHttpError(401, "IGBT_ADMIN_TOKEN_NOT_CONFIGURED");
    }
    throw DatabaseHttpError(401, "INVALID_TOKEN");
}

std::string EngineerSessionAuth::require_station_token(const std::optional<std::string>& supplied) const {
    const std::string expected_station = get_env_value("IGBT_STATION_TOKEN");
    if (expected_station.empty()) {
        throw DatabaseHttpError(401, "IGBT_STATION_TOKEN_NOT_CONFIGURED");
    }
    if (!supplied.has_value() || !constant_time_equals(*supplied, expected_station)) {
        throw DatabaseHttpError(401, "INVALID_TOKEN");
    }
    return "station-runtime";
}

std::string EngineerSessionAuth::require_station_or_admin_token(
    const std::optional<std::string>& x_station_token,
    const std::optional<std::string>& x_admin_token
) {
    const std::string expected_station = get_env_value("IGBT_STATION_TOKEN");
    if (x_station_token.has_value() && !expected_station.empty() &&
        constant_time_equals(*x_station_token, expected_station)) {
        return "station-runtime";
    }
    if (x_admin_token.has_value()) {
        if (verify_session_token(*x_admin_token)) {
            return "engineer-session";
        }
        const std::string expected_admin = get_env_value("IGBT_ADMIN_TOKEN");
        if (!expected_admin.empty() && constant_time_equals(*x_admin_token, expected_admin)) {
            return "local-admin";
        }
    }
    if (expected_station.empty() &&
        get_env_value("IGBT_ADMIN_TOKEN").empty() &&
        get_env_value("IGBT_ENGINEER_PIN").empty()) {
        throw DatabaseHttpError(401, "STATION_OR_ADMIN_TOKEN_NOT_CONFIGURED");
    }
    throw DatabaseHttpError(401, "INVALID_TOKEN");
}

// ============================================================================
// StationHttpServer Path Auto-Detection
// ============================================================================

std::filesystem::path StationHttpServer::detect_default_package_root() {
    std::vector<std::filesystem::path> roots;
    std::error_code ec;
    roots.push_back(std::filesystem::current_path(ec));

    wchar_t exe_buf[MAX_PATH * 4]{};
    DWORD n = GetModuleFileNameW(nullptr, exe_buf, static_cast<DWORD>(std::size(exe_buf)));
    if (n > 0) {
        std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();
        roots.push_back(exe_dir);
        if (exe_dir.has_parent_path()) roots.push_back(exe_dir.parent_path());
        if (exe_dir.parent_path().has_parent_path()) roots.push_back(exe_dir.parent_path().parent_path());
    }

    for (const auto& base : roots) {
        if (base.empty()) continue;
        auto pkg = base / "IGBT_PokaYoke_Development_Package";
        if (std::filesystem::is_directory(pkg, ec)) {
            return std::filesystem::weakly_canonical(pkg, ec);
        }
        if (std::filesystem::is_directory(base / "web" / "dist", ec) ||
            std::filesystem::is_directory(base / "src" / "igbt_pokayoke", ec)) {
            return std::filesystem::weakly_canonical(base, ec);
        }
    }
    return std::filesystem::current_path(ec) / "IGBT_PokaYoke_Development_Package";
}

std::filesystem::path StationHttpServer::detect_default_data_dir() {
    std::string env_dir = read_process_env_utf8("IGBT_DATA_DIR");
    if (!env_dir.empty()) {
        std::error_code ec;
        return std::filesystem::absolute(path_from_utf8(env_dir), ec);
    }
    return detect_default_package_root() / "data";
}

std::filesystem::path StationHttpServer::detect_default_web_dist_dir() {
    std::string env_dir = read_process_env_utf8("IGBT_WEB_DIST_DIR");
    if (!env_dir.empty()) {
        std::error_code ec;
        return std::filesystem::absolute(path_from_utf8(env_dir), ec);
    }
    return detect_default_package_root() / "web" / "dist";
}

std::filesystem::path StationHttpServer::detect_default_assets_dir() {
    return detect_default_package_root() / "assets";
}

// ============================================================================
// StationHttpServer
// ============================================================================

StationHttpServer::StationHttpServer(
    StationDatabase& db,
    LiveCameraEngine& camera,
    TrainingManager& training,
    StationServerOptions options
)
    : db_(db),
      camera_(camera),
      training_(training),
      options_(std::move(options)),
      red_rabbit_(
          db_,
          "STATION_01",
          RedRabbitPolicy{},
          default_wall_clock_ms,
          has_recent_crash_dump(options_.data_dir.empty() ? db_.data_root() : options_.data_dir)
      ) {
    if (options_.data_dir.empty()) {
        options_.data_dir = db_.data_root();
    }
    if (options_.web_dist_dir.empty()) {
        options_.web_dist_dir = detect_default_web_dist_dir();
    }
    if (options_.assets_dir.empty()) {
        options_.assets_dir = detect_default_assets_dir();
    }

    const AdamModbusConfig adam_cfg = resolve_adam_config(options_, options_.data_dir, auth_);
    if (adam_cfg.enabled) {
        auto driver = std::make_unique<AdamModbusDioDriver>(
            adam_cfg,
            [this]() { return camera_.is_running(); },
            [this]() { return camera_.last_vision_tick_ms(); },
            [this](bool fault_locked, const std::string& reason) {
                try {
                    if (fault_locked && !reason.empty()) {
                        db_.rr_append_lock_audit(
                            "WATCHDOG_FAULT_LOCK",
                            "SYSTEM",
                            "SYSTEM",
                            reason,
                            "ADAM Modbus TCP 看门狗或安全输入触发硬件故障落锁",
                            std::nullopt,
                            default_wall_clock_ms()
                        );
                    }
                } catch (...) {
                }
            }
        );
        driver->start();
        interlock_ = std::move(driver);
    } else {
        interlock_ = std::make_unique<ShadowNoopInterlock>();
    }

    red_rabbit_.set_interlock_callback([this](InterlockSignal sig) {
        if (interlock_) interlock_->set_interlock(sig);
    });

    // Wire LiveCameraEngine configuration loader & cycle persistence callback
    camera_.set_evidence_root(db_.evidence_root());
    camera_.set_config_loader([this]() { return db_.get_config(); });
    camera_.update_config(db_.get_config());
    {
        const json saved_optics = db_.get_camera_optics();
        camera_.set_camera_optics(
            camera_optics_from_json(saved_optics),
            saved_optics.value("calibrated", false)
        );
    }
    camera_.set_optics_persist_callback([this](const CameraOpticsConfig& cfg) {
        try {
            db_.save_camera_optics(camera_optics_to_json(cfg), "auto-anti-glare");
        } catch (...) {
        }
    });
    camera_.set_cycle_callback(
        [this](const json& snap, const std::optional<std::string>& prev_fsm, const std::vector<uint8_t>& ev_jpg) {
            const bool saved = db_.record_live_transition(snap, prev_fsm, ev_jpg);
            const std::string outcome = snap.value("outcome", "");
            if (outcome == "PASS" || outcome == "ALARM") {
                std::optional<double> ood_energy;
                if (snap.contains("ood_energy") && snap["ood_energy"].is_number()) {
                    const double v = snap["ood_energy"].get<double>();
                    if (std::isfinite(v)) ood_energy = v;
                }
                std::optional<double> ood_maha;
                if (snap.contains("ood_mahalanobis") && snap["ood_mahalanobis"].is_number()) {
                    const double v = snap["ood_mahalanobis"].get<double>();
                    if (std::isfinite(v)) ood_maha = v;
                }
                (void)red_rabbit_.try_consume_live_verdict(
                    outcome,
                    snap.value("latest_evidence_url", ""),
                    ood_energy,
                    ood_maha
                );
            }
            if (interlock_) {
                const DioHealthStatus dio = interlock_->health_status();
                InterlockSignal sig = InterlockSignal::LOCKED;
                if (dio.fault_locked) {
                    sig = InterlockSignal::FAULT_HALT;
                } else if (outcome == "ALARM" || red_rabbit_.state() == RedRabbitState::CHALLENGE_HARD_LOCK) {
                    sig = InterlockSignal::ALARM_LATCHED;
                } else if (!red_rabbit_.is_blocking() && outcome == "PASS" && snap.value("view_valid", false)) {
                    sig = InterlockSignal::PASS_ALLOW;
                }
                interlock_->set_interlock(sig);
            }
            return saved;
        }
    );

    // Wire skip-frame auto-capture callback directly into TrainingManager
    camera_.set_auto_sample_callback(
        [this](const cv::Mat& frame_bgr,
               const cv::Mat& crop_bgr,
               const std::string& label,
               const std::string& pipe_color,
               const std::string& physical_tube_id,
               const std::string& batch_id,
               std::optional<Box> frame_box) {
            try {
                training_.capture_sample(
                    frame_bgr,
                    crop_bgr,
                    label,
                    pipe_color,
                    physical_tube_id,
                    batch_id,
                    frame_box
                );
            } catch (...) {
            }
        }
    );

    if (options_.initial_work_mode == "strict" || options_.initial_work_mode == "debug") {
        camera_.set_work_mode(options_.initial_work_mode);
    }

    svr_.new_task_queue = [] { return new httplib::ThreadPool(24); };
    svr_.set_payload_max_length(kMaxOnnxUploadBytes);
    svr_.set_read_timeout(15, 0);
    svr_.set_write_timeout(15, 0);

    register_routes();
    register_static_and_spa_routes();
}

StationHttpServer::~StationHttpServer() {
    stop();
}

json StationHttpServer::apply_validity_gate(json state) {
    if (!state.is_object()) state = json::object();
    state["mode"] = "shadow";
    state["interlock_output_enabled"] = false;
    if (state.value("outcome", "") == "PASS") {
        const bool view_ok = state.contains("view_valid") && state["view_valid"].is_boolean() && state["view_valid"].get<bool>();
        const bool qual_ok = state.contains("quality_ok") && state["quality_ok"].is_boolean() && state["quality_ok"].get<bool>();
        const bool cam_ok = state.value("camera_status", "") == "ONLINE";
        const bool lbl_ok = state.value("label_status", "unknown") != "unknown";
        const bool fsm_ok = state.value("fsm_state", "") == "PASS";
        if (!view_ok || !qual_ok || !cam_ok || !lbl_ok || !fsm_ok) {
            state["outcome"] = view_ok ? "ALARM" : "VIEW_INVALID";
            state["result_reason"] = "SHADOW_PASS_REJECTED_MISSING_VALID_EVIDENCE";
        }
    }
    return state;
}

json StationHttpServer::build_offline_live_state() const {
    json state = camera_.get_state_json();
    if (!state.is_object()) state = StationDatabase::default_station_state();
    state["mode"] = "shadow";
    state["interlock_output_enabled"] = false;
    state["online"] = false;
    state["camera_online"] = false;
    std::string cur_status = state.value("camera_status", "NOT_CONNECTED");
    state["camera_status"] = (cur_status == "ONLINE") ? "OFFLINE" : (cur_status.empty() ? "NOT_CONNECTED" : cur_status);
    state["fps"] = 0.0;
    state["view_valid"] = false;
    state["quality_ok"] = false;

    std::vector<std::string> reasons;
    if (state.contains("view_valid_reasons") && state["view_valid_reasons"].is_array()) {
        for (const auto& item : state["view_valid_reasons"]) {
            if (item.is_string()) reasons.push_back(item.get<std::string>());
        }
    }
    if (std::find(reasons.begin(), reasons.end(), "CAMERA_NOT_CONNECTED") == reasons.end() &&
        std::find(reasons.begin(), reasons.end(), "CAMERA_OFFLINE") == reasons.end()) {
        reasons.insert(reasons.begin(), "CAMERA_NOT_CONNECTED");
    }
    state["view_valid_reasons"] = reasons;
    state["label_status"] = "unknown";
    state["label_confidence"] = nullptr;
    state["c_timer_ms"] = 0;

    const bool alarm_latched = state.value("alarm_latched", false) || state.value("outcome", "") == "ALARM";
    if (alarm_latched) {
        state["outcome"] = "ALARM";
        state["fsm_state"] = "ALARM";
    } else {
        state["outcome"] = "VIEW_INVALID";
        state["fsm_state"] = "VIEW_INVALID";
        state["result_reason"] = "CAMERA_NOT_CONNECTED";
    }

    state.erase("live_stream_url");
    state.erase("live_crop_url");
    state["mirror_horizontal"] = camera_.get_mirror_horizontal();
    state["work_mode"] = camera_.get_work_mode();
    state["auto_capture"] = camera_.get_auto_capture().to_json();
    state["camera_optics"] = camera_.get_camera_optics_json();
    state["engine"] = "cpp20";
    if (!state.contains("updated_at_utc") || state["updated_at_utc"].is_null()) {
        state["updated_at_utc"] = utc_now_iso();
    }
    return apply_validity_gate(std::move(state));
}

json StationHttpServer::build_current_state_response() {
    const json rr = red_rabbit_.status_json();
    const bool rr_blocking = rr.value("blocking", true);
    const std::string rr_block_reason =
        rr.value("hard_locked", false) ? "RED_RABBIT_HARD_LOCK" : "RED_RABBIT_CHALLENGE_REQUIRED";
    const DioHealthStatus dio = interlock_ ? interlock_->health_status() : DioHealthStatus{};
    const json dio_json = dio_health_to_json(dio);

    auto attach_red_rabbit = [&](json st, bool override_pass) -> json {
        st["red_rabbit"] = rr;
        st["red_rabbit_blocking"] = rr_blocking;
        st["hal_dio"] = dio_json;
        st["hardware_fault_locked"] = dio.fault_locked;
        if (override_pass) {
            if (dio.fault_locked) {
                st["interlock_output_enabled"] = false;
                st["outcome"] = "FAULT";
                st["fsm_state"] = "FAULT";
                st["result_reason"] = dio.fault_reason.empty() ? "MODBUS_WATCHDOG_TIMEOUT" : dio.fault_reason;
            } else if (rr_blocking) {
                st["interlock_output_enabled"] = false;
                if (st.value("outcome", "") == "PASS") {
                    st["outcome"] = "ALARM";
                    st["result_reason"] = rr_block_reason;
                }
            } else if (dio.configured && dio.hardware_ready) {
                st["mode"] = "hardware_interlock";
                st["interlock_output_enabled"] = true;
            }
        }
        return st;
    };

    double sim_until = 0.0;
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        sim_until = sim_override_until_mono_;
    }
    const bool in_sim_override = monotonic_seconds() < sim_until;
    const bool cam_online = camera_.camera_online();

    if (cam_online && !in_sim_override) {
        json live = apply_validity_gate(camera_.get_state_json());
        live["online"] = true;
        live["mirror_horizontal"] = camera_.get_mirror_horizontal();
        live["work_mode"] = camera_.get_work_mode();
        live["auto_capture"] = camera_.get_auto_capture().to_json();
        live["engine"] = "cpp20";
        live["live_stream_url"] = "/api/camera/stream";
        live["live_crop_url"] = "/api/camera/stream_crop";
        return db_.attach_latest_evidence_and_summary(attach_red_rabbit(std::move(live), true));
    }

    json persisted = db_.get_persisted_station_state();
    const bool live_enabled = camera_.is_running() || options_.enable_live_camera ||
                              auth_.get_env_value("IGBT_ENABLE_LIVE_CAMERA") == "1";
    const std::string cyc_id = persisted.contains("cycle_id") && persisted["cycle_id"].is_string()
                                   ? persisted["cycle_id"].get<std::string>()
                                   : "";
    const bool persisted_is_live = persisted.value("state_source", "") == "live" ||
                                   persisted.value("live_stream_url", "") == "/api/camera/stream" ||
                                   cyc_id.rfind("LIVE_", 0) == 0;

    if (!cam_online && !in_sim_override && (live_enabled || persisted_is_live)) {
        return db_.attach_latest_evidence_and_summary(attach_red_rabbit(build_offline_live_state(), true));
    }

    json state = apply_validity_gate(std::move(persisted));
    state["mirror_horizontal"] = camera_.get_mirror_horizontal();
    state["work_mode"] = camera_.get_work_mode();
    state["auto_capture"] = camera_.get_auto_capture().to_json();
    state["camera_optics"] = camera_.get_camera_optics_json();
    state["engine"] = "cpp20";
    state["online"] = cam_online || state.value("camera_status", "") == "ONLINE";
    if (cam_online) {
        state["live_stream_url"] = "/api/camera/stream";
        state["live_crop_url"] = "/api/camera/stream_crop";
    } else {
        state.erase("live_stream_url");
        state.erase("live_crop_url");
    }
    return db_.attach_latest_evidence_and_summary(attach_red_rabbit(std::move(state), !in_sim_override));
}

json StationHttpServer::load_verified_model_into_camera(
    const std::filesystem::path& model_path,
    const std::string& artifact_format,
    const std::string& model_version,
    const std::string& expected_sha256,
    int input_size,
    int input_height,
    const std::filesystem::path& ood_calibration_path
) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model_path, ec)) {
        throw DatabaseHttpError(404, "MODEL_ARTIFACT_UNAVAILABLE");
    }

    std::vector<uint8_t> raw_bytes;
    if (!read_file_bytes(model_path, raw_bytes) || raw_bytes.empty()) {
        throw DatabaseHttpError(404, "MODEL_ARTIFACT_UNAVAILABLE");
    }

    const std::string actual_sha = sha256_hex(raw_bytes);
    const std::string expected_norm = to_lower_ascii(expected_sha256);
    if (expected_norm.empty() || !constant_time_equals(actual_sha, expected_norm)) {
        throw DatabaseHttpError(409, "MODEL_SHA256_MISMATCH");
    }

    const std::string ext = to_lower_ascii(model_path.extension().string());
    const bool is_mlp = (artifact_format == "opencv_mlp" || ext == ".xml");
    try {
        if (is_mlp) {
            return camera_.load_mlp_model(model_path, model_version, expected_norm, ood_calibration_path);
        }
        return camera_.load_onnx_model(
            model_path, model_version, expected_norm, input_size, input_height, ood_calibration_path
        );
    } catch (const DatabaseHttpError&) {
        throw;
    } catch (const std::exception& exc) {
        const std::string msg = exc.what();
        if (msg.find("SHA256") != std::string::npos || msg.find("sha256") != std::string::npos) {
            throw DatabaseHttpError(409, "MODEL_SHA256_MISMATCH");
        }
        throw DatabaseHttpError(422, "MODEL_ARTIFACT_LOAD_FAILED:" + msg);
    }
}

bool StationHttpServer::autoload_latest_verified_model() {
    try {
        const std::string stride_env = auth_.get_env_value("IGBT_HAND_GLOVE_FRAME_STRIDE");
        if (!stride_env.empty()) {
            try {
                camera_.set_hand_glove_frame_stride(std::stoi(stride_env));
            } catch (...) {
            }
        }
        std::filesystem::path hg_path;
        const std::string hg_env = auth_.get_env_value("IGBT_HAND_GLOVE_ONNX");
        if (!hg_env.empty()) {
            hg_path = path_from_utf8(hg_env);
        } else if (!options_.data_dir.empty()) {
            hg_path = options_.data_dir / "models" / "nanodet_hand_glove.onnx";
        }
        std::error_code ec;
        if (!hg_path.empty() && std::filesystem::is_regular_file(hg_path, ec)) {
            std::string sidecar_sha;
            auto sha_path = hg_path;
            sha_path += ".sha256";
            std::vector<uint8_t> sha_bytes;
            if (std::filesystem::is_regular_file(sha_path, ec) && read_file_bytes(sha_path, sha_bytes)) {
                std::string raw(sha_bytes.begin(), sha_bytes.end());
                std::istringstream iss(raw);
                iss >> sidecar_sha;
                sidecar_sha = to_lower_ascii(sidecar_sha);
            }
            try {
                camera_.load_hand_glove_onnx_model(hg_path, "nanodet-plus-m-auto", sidecar_sha);
            } catch (...) {
            }
        }
    } catch (...) {
    }

    try {
        const auto runs = training_.find_verified_runs();
        for (const auto& run : runs) {
            try {
                load_verified_model_into_camera(
                    run.model_path,
                    run.artifact_format,
                    run.model_version,
                    run.sha256,
                    run.input_width,
                    run.input_height,
                    run.ood_calibration_path
                );
                return true;
            } catch (...) {
                continue;
            }
        }
    } catch (...) {
    }
    return false;
}

bool StationHttpServer::on_training_completed(const json& run_summary) {
    try {
        if (run_summary.contains("onnx_parity_ok") && run_summary["onnx_parity_ok"] == false) {
            return false;
        }
        if (run_summary.contains("artifact_verified") && run_summary["artifact_verified"] == false) {
            return false;
        }

        std::string path_str = run_summary.value("model_path", "");
        if (path_str.empty()) path_str = run_summary.value("onnx_path", "");
        std::string expected_sha = to_lower_ascii(run_summary.value("sha256", ""));
        if (expected_sha.empty()) expected_sha = to_lower_ascii(run_summary.value("onnx_sha256", ""));
        const std::string model_version = run_summary.value("model_version", "");
        if (path_str.empty() || expected_sha.empty() || model_version.empty()) {
            return false;
        }

        const std::string model_name = run_summary.value("model_name", "opencv-mlp-cpp-igbt");
        const std::string artifact_format = run_summary.value("artifact_format", "opencv_mlp");
        const int input_size = run_summary.contains("input_size")
                                   ? parse_optional_input_size(run_summary["input_size"], kDefaultInputSize)
                                   : kDefaultInputSize;

        const auto model_path = path_from_utf8(path_str);
        std::vector<uint8_t> raw_bytes;
        if (!read_file_bytes(model_path, raw_bytes) || raw_bytes.empty()) {
            return false;
        }
        const std::string actual_sha = sha256_hex(raw_bytes);
        if (!constant_time_equals(actual_sha, expected_sha)) {
            return false;
        }

        // Hot-load into LiveCameraEngine and register in SQLite model_registry
        load_verified_model_into_camera(model_path, artifact_format, model_version, expected_sha, input_size);
        try {
            red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", "trainer-pipeline");
        } catch (...) {
        }

        const double test_acc = run_summary.value("test_accuracy", 0.0) * 100.0;
        const int train_n = run_summary.value("train_samples", 0);
        const int val_n = run_summary.value("val_samples", 0);
        const int test_n = run_summary.value("test_samples", 0);
        std::ostringstream notes_oss;
        notes_oss << std::fixed << std::setprecision(1)
                  << "测试集准确率: " << test_acc << "% · 格式: " << artifact_format
                  << " · 样本(训练/验证/测试): " << train_n << "/" << val_n << "/" << test_n;

        db_.register_model(
            model_name,
            model_version,
            expected_sha,
            "reviewed",
            notes_oss.str(),
            "trainer-pipeline"
        );
        return true;
    } catch (...) {
        return false;
    }
}

void StationHttpServer::register_routes() {
    // TrustedHost loopback protection
    svr_.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        if (req.has_header("Host") && !is_trusted_loopback_host(req.get_header_value("Host"))) {
            send_error(res, 400, "Invalid host header");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    auto wrap = [](httplib::Server::Handler fn) -> httplib::Server::Handler {
        return [fn = std::move(fn)](const httplib::Request& req, httplib::Response& res) {
            try {
                fn(req, res);
            } catch (const DatabaseHttpError& err) {
                if (err.status_code() == 429) {
                    res.set_header("Retry-After", "60");
                }
                send_error(res, err.status_code(), err.detail());
            } catch (const std::out_of_range& err) {
                send_error(res, 404, err.what());
            } catch (const std::invalid_argument& err) {
                send_error(res, 422, err.what());
            } catch (const std::exception& err) {
                send_error(res, 500, std::string("INTERNAL_ERROR:") + err.what());
            }
        };
    };

    // 1. Health & Auth
    svr_.Get("/api/health", wrap([this](const httplib::Request&, httplib::Response& res) {
        const DioHealthStatus dio = interlock_ ? interlock_->health_status() : DioHealthStatus{};
        const bool hw_interlock_active = dio.configured && dio.hardware_ready && !dio.fault_locked;
        send_json(res, 200, json{
            {"status", "ok"},
            {"host_policy", "loopback_only"},
            {"mode", hw_interlock_active ? "hardware_interlock" : "shadow"},
            {"interlock_output_enabled", hw_interlock_active},
            {"interlock_backend", interlock_ ? interlock_->backend_name() : "shadow_noop"},
            {"interlock_hardware_ready", dio.hardware_ready},
            {"interlock_fault_locked", dio.fault_locked},
            {"mes_connected", mes_client_.is_connected()},
            {"engine", "cpp20"}
        });
    }));

    svr_.Get("/api/diagnostics/runtime", wrap([this](const httplib::Request&, httplib::Response& res) {
        const json cam_state = camera_.get_state_json();
        const DioHealthStatus dio = interlock_ ? interlock_->health_status() : DioHealthStatus{};
        send_json(res, 200, json{
            {"async_evidence", camera_.async_evidence_stats()},
            {"db_pragmas", db_.pragma_diagnostics()},
            {"hardware_3a", cam_state.value("hardware_3a", json::object())},
            {"camera_optics", camera_.get_camera_optics_json()},
            {"red_rabbit", red_rabbit_.status_json()},
            {"hal_dio", dio_health_to_json(dio)},
            {"tube_alignment", camera_.tube_alignment_status_json()},
            {"hand_glove_detector", camera_.hand_glove_detector_status_json()},
            {"label_ood_classifier", camera_.label_ood_status_json()}
        });
    }));

    svr_.Get("/api/vision/pipeline_status", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, json{
            {"tube_alignment", camera_.tube_alignment_status_json()},
            {"hand_glove_detector", camera_.hand_glove_detector_status_json()},
            {"label_ood_classifier", camera_.label_ood_status_json()}
        });
    }));

    svr_.Get("/api/auth/status", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, auth_.status_json());
    }));

    svr_.Post("/api/auth/verify", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || body.size() != 1 || !body.contains("passcode") || !body["passcode"].is_string()) {
            throw DatabaseHttpError(422, "INVALID_AUTH_VERIFY_PAYLOAD");
        }
        send_json(res, 200, auth_.verify_passcode(body["passcode"].get<std::string>()));
    }));

    // 2. Real-time State & MJPEG Streams
    svr_.Get("/api/state", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, build_current_state_response());
    }));

    auto make_mjpeg_handler = [this](bool is_crop) -> httplib::Server::Handler {
        return [this, is_crop](const httplib::Request&, httplib::Response& res) {
            int cur = active_mjpeg_streams_.load();
            while (true) {
                if (cur >= kMaxConcurrentMjpegStreams) {
                    res.status = 503;
                    res.set_header("Retry-After", "2");
                    res.set_content(R"({"detail":"MJPEG_STREAM_LIMIT_REACHED"})", "application/json");
                    return;
                }
                if (active_mjpeg_streams_.compare_exchange_weak(cur, cur + 1)) {
                    break;
                }
            }

            struct StreamSlotGuard {
                std::atomic<int>& counter;
                explicit StreamSlotGuard(std::atomic<int>& c) : counter(c) {}
                ~StreamSlotGuard() { counter.fetch_sub(1); }
            };

            res.status = 200;
            res.set_header("Cache-Control", "no-store");

            struct StreamState {
                std::shared_ptr<StreamSlotGuard> slot_guard;
                uint64_t last_seq{0};
                double no_frame_since{-1.0};
                double last_new_frame_at{-1.0};
            };
            auto st = std::make_shared<StreamState>();
            st->slot_guard = std::make_shared<StreamSlotGuard>(active_mjpeg_streams_);

            res.set_chunked_content_provider(
                "multipart/x-mixed-replace; boundary=frame",
                [this, is_crop, st](std::size_t, httplib::DataSink& sink) -> bool {
                    if (stopping_.load() || !sink.is_writable()) {
                        return false;
                    }
                    const bool engine_active = camera_.camera_online() || camera_.is_running();
                    const double timeout_s = engine_active ? kStreamNoFrameTimeoutS : kStreamOfflineTimeoutS;
                    const int slice_ms = engine_active ? 200 : 120;

                    WaitJpegResult frame = camera_.wait_for_jpeg(st->last_seq, slice_ms, is_crop);
                    if (stopping_.load() || !sink.is_writable()) {
                        return false;
                    }

                    const double now = monotonic_seconds();
                    if (frame.has_frame() && frame.seq != st->last_seq) {
                        st->last_seq = frame.seq;
                        st->last_new_frame_at = now;
                        st->no_frame_since = -1.0;

                        std::string header =
                            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                            std::to_string(frame.jpeg.size()) + "\r\n\r\n";
                        std::string packet;
                        packet.reserve(header.size() + frame.jpeg.size() + 2);
                        packet.append(header);
                        packet.append(reinterpret_cast<const char*>(frame.jpeg.data()), frame.jpeg.size());
                        packet.append("\r\n");

                        if (!sink.write(packet.data(), packet.size())) {
                            return false;
                        }
                        if (!engine_active) {
                            sink.done();
                            return true;
                        }
                        return true;
                    }

                    if (!engine_active) {
                        if (st->no_frame_since < 0.0) st->no_frame_since = now;
                        if ((now - st->no_frame_since) >= timeout_s || st->last_new_frame_at >= 0.0) {
                            sink.done();
                            return true;
                        }
                    } else {
                        if (st->last_new_frame_at >= 0.0) {
                            if ((now - st->last_new_frame_at) >= timeout_s) {
                                sink.done();
                                return true;
                            }
                        } else {
                            if (st->no_frame_since < 0.0) st->no_frame_since = now;
                            if ((now - st->no_frame_since) >= timeout_s) {
                                sink.done();
                                return true;
                            }
                        }
                    }
                    return true;
                }
            );
        };
    };

    svr_.Get("/api/camera/stream", make_mjpeg_handler(false));
    svr_.Get("/api/camera/stream_crop", make_mjpeg_handler(true));

    svr_.Get("/api/camera/frame.jpg", wrap([this](const httplib::Request&, httplib::Response& res) {
        auto jpg = camera_.get_raw_frame_jpeg(92);
        if (jpg.empty()) {
            throw DatabaseHttpError(503, "LIVE_CAMERA_FRAME_UNAVAILABLE");
        }
        res.status = 200;
        res.set_header("Cache-Control", "no-store");
        res.set_content(reinterpret_cast<const char*>(jpg.data()), jpg.size(), "image/jpeg");
    }));

    svr_.Post("/api/camera/capture", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        auto snap = camera_.get_atomic_capture(std::nullopt, true);
        if (!snap.has_frame() || snap.frame_jpeg.empty()) {
            throw DatabaseHttpError(503, "当前未获取到实时摄像头画面");
        }
        json out = db_.save_manual_capture(
            snap.frame_jpeg,
            snap.state,
            LiveCameraEngine::generate_live_cycle_id()
        );
        send_json(res, 200, out);
    }));

    // 3. Camera Runtime Controls
    svr_.Post("/api/camera/auxiliary_rois", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || body.size() != 1 || !body.contains("show") || !body["show"].is_boolean()) {
            throw DatabaseHttpError(422, "INVALID_AUXILIARY_ROIS_PAYLOAD");
        }
        bool show = camera_.set_show_auxiliary_rois(body["show"].get<bool>());
        send_json(res, 200, json{{"show_auxiliary_rois", show}});
    }));

    svr_.Post("/api/camera/mirror", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || body.size() != 1 || !body.contains("mirror") || !body["mirror"].is_boolean()) {
            throw DatabaseHttpError(422, "INVALID_MIRROR_PAYLOAD");
        }
        bool mirror = camera_.set_mirror_horizontal(body["mirror"].get<bool>());
        send_json(res, 200, json{{"mirror_horizontal", mirror}});
    }));

    svr_.Post("/api/camera/work_mode", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || body.size() != 1 || !body.contains("work_mode") || !body["work_mode"].is_string()) {
            throw DatabaseHttpError(422, "INVALID_WORK_MODE_PAYLOAD");
        }
        const std::string mode = body["work_mode"].get<std::string>();
        if (mode != "debug" && mode != "strict") {
            throw DatabaseHttpError(422, "INVALID_WORK_MODE:" + mode);
        }
        send_json(res, 200, json{{"work_mode", camera_.set_work_mode(mode)}});
    }));

    svr_.Get("/api/camera/optics", wrap([this](const httplib::Request&, httplib::Response& res) {
        json out = camera_.get_camera_optics_json();
        const json persisted = db_.get_camera_optics();
        out["updated_at_utc"] = persisted.value("updated_at_utc", json(nullptr));
        out["changed_by"] = persisted.value("changed_by", json(nullptr));
        send_json(res, 200, out);
    }));

    svr_.Post("/api/camera/optics", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        const json saved = db_.save_camera_optics(body, actor);
        json out = camera_.set_camera_optics(camera_optics_from_json(saved), true);
        red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
        out["updated_at_utc"] = saved.value("updated_at_utc", json(nullptr));
        out["changed_by"] = saved.value("changed_by", json(nullptr));
        send_json(res, 200, out);
    }));

    svr_.Post("/api/camera/optics/auto_calibrate", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        json out = camera_.trigger_anti_glare_calibration();
        red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
        send_json(res, 200, json{{"triggered", true}, {"optics", std::move(out)}});
    }));

    // 4. Simulation & Station Snapshot
    svr_.Post("/api/runtime/simulate/:scenario", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const std::string scenario = req.path_params.at("scenario");
        static const std::set<std::string> kScenarios = {
            "a_pass", "b_pass", "c_pass", "alarm", "initial_alarm",
            "bare_hand_alarm", "glare_invalid", "reset"
        };
        if (kScenarios.find(scenario) == kScenarios.end()) {
            throw DatabaseHttpError(422, "UNSUPPORTED_SCENARIO:" + scenario);
        }

        if (camera_.camera_online() && scenario == "reset") {
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                sim_override_until_mono_ = 0.0;
            }
            json st = apply_validity_gate(camera_.trigger_new_cycle());
            st["online"] = true;
            send_json(res, 200, db_.attach_latest_evidence_and_summary(std::move(st)));
            return;
        }

        if (camera_.camera_online()) {
            camera_.pause_for_simulation(10.0);
            std::lock_guard<std::mutex> lk(state_mutex_);
            sim_override_until_mono_ = monotonic_seconds() + 10.0;
        }

        SimulationRunResult run = camera_.run_simulation_scenario(scenario, options_.assets_dir);
        json gated = apply_validity_gate(run.snapshot);
        send_json(res, 200, db_.record_simulated_run(run, scenario, gated));
    }));

    svr_.Post("/api/runtime/snapshot", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_station_token(get_opt_header(req, "X-Station-Token"));
        json validated = validate_runtime_snapshot_payload(parse_json_body_or_throw(req));
        json gated = apply_validity_gate(std::move(validated));
        const bool view_ok = gated.contains("view_valid") && gated["view_valid"].is_boolean() && gated["view_valid"].get<bool>();
        if (!view_ok && gated.value("outcome", "") == "WAITING") {
            gated["outcome"] = "VIEW_INVALID";
            if (!gated.contains("result_reason") || gated["result_reason"].is_null()) {
                gated["result_reason"] = "VIEW_INVALID";
            }
        }
        send_json(res, 200, db_.record_runtime_snapshot(gated));
    }));

    // 5. Configuration & Storage Overview
    svr_.Get("/api/config", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, db_.get_config());
    }));

    svr_.Put("/api/config", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        json updated = db_.update_config(parse_json_body_or_throw(req), actor);
        camera_.update_config(updated);
        send_json(res, 200, updated);
    }));

    svr_.Get("/api/config/history", wrap([this](const httplib::Request& req, httplib::Response& res) {
        send_json(res, 200, db_.list_config_history(get_int_query(req, "limit", 50)));
    }));

    svr_.Get("/api/storage/overview", wrap([this](const httplib::Request&, httplib::Response& res) {
        const json ds_stats = training_.storage_stats();
        const json cam_state = camera_.get_state_json();
        std::string active_ver = cam_state.value("model_version", "");
        send_json(res, 200, db_.storage_overview(ds_stats, active_ver));
    }));

    // 6. Model Registry, Activation & Raw ONNX Import
    svr_.Post("/api/models", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object()) throw DatabaseHttpError(422, "INVALID_MODEL_PAYLOAD");
        static const std::set<std::string> kAllowed = {
            "model_name", "model_version", "sha256", "license_status", "notes"
        };
        for (auto it = body.begin(); it != body.end(); ++it) {
            if (kAllowed.find(it.key()) == kAllowed.end()) {
                throw DatabaseHttpError(422, "EXTRA_FIELD_FORBIDDEN:" + it.key());
            }
        }
        if (!body.contains("model_name") || !body["model_name"].is_string() ||
            !body.contains("model_version") || !body["model_version"].is_string() ||
            !body.contains("sha256") || !body["sha256"].is_string()) {
            throw DatabaseHttpError(422, "MISSING_REQUIRED_MODEL_FIELDS");
        }
        std::string license_status = body.value("license_status", "unknown");
        std::string notes = body.value("notes", "");
        send_json(res, 200, db_.register_model(
            body["model_name"].get<std::string>(),
            body["model_version"].get<std::string>(),
            body["sha256"].get<std::string>(),
            license_status,
            notes,
            actor
        ));
    }));

    svr_.Get("/api/models", wrap([this](const httplib::Request& req, httplib::Response& res) {
        send_json(res, 200, db_.list_models(get_int_query(req, "limit", 50)));
    }));

    svr_.Post("/api/models/import_onnx", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        if (req.body.empty()) {
            throw DatabaseHttpError(400, "EMPTY_ONNX_MODEL_BODY");
        }
        if (req.body.size() > kMaxOnnxUploadBytes) {
            throw DatabaseHttpError(413, "ONNX_MODEL_EXCEEDS_MAX_SIZE");
        }

        std::string version_name;
        if (auto hv = clean_optional_header(get_opt_header(req, "X-Model-Version")); hv.has_value()) {
            version_name = *hv;
        } else if (req.has_param("version")) {
            version_name = req.get_param_value("version");
        } else if (req.has_param("model_version")) {
            version_name = req.get_param_value("model_version");
        }

        int input_size = kDefaultInputSize;
        if (auto hs = clean_optional_header(get_opt_header(req, "X-Input-Size"), 16); hs.has_value()) {
            input_size = parse_optional_input_size(json(*hs), kDefaultInputSize);
        } else if (req.has_param("input_size")) {
            input_size = parse_optional_input_size(json(req.get_param_value("input_size")), kDefaultInputSize);
        }

        int input_height = 0;
        if (auto hh = clean_optional_header(get_opt_header(req, "X-Input-Height"), 16); hh.has_value()) {
            input_height = parse_optional_input_size(json(*hh), 0);
        } else if (req.has_param("input_height")) {
            input_height = parse_optional_input_size(json(req.get_param_value("input_height")), 0);
        }

        json imported = training_.import_onnx_bytes(req.body, version_name, input_size, input_height);
        const std::string model_version = imported.value("model_version", "");
        const std::string model_name = imported.value("model_name", "onnx-imported-cpp-igbt");
        std::string sha = to_lower_ascii(imported.value("sha256", ""));
        if (sha.empty()) sha = to_lower_ascii(imported.value("onnx_sha256", ""));
        std::string path_str = imported.value("model_path", "");
        if (path_str.empty()) path_str = imported.value("onnx_path", "");
        const int actual_w = imported.value("input_width", imported.value("input_size", input_size));
        const int actual_h = imported.value("input_height", input_height > 0 ? input_height : actual_w);

        json load_info = load_verified_model_into_camera(
            path_from_utf8(path_str),
            "onnx",
            model_version,
            sha,
            actual_w,
            actual_h
        );
        red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
        json reg = db_.register_model(
            model_name,
            model_version,
            sha,
            "reviewed",
            "外部导入 ONNX 模型 (OpenCV DNN C++20 校验通过)",
            actor
        );

        json out = imported;
        out["imported"] = true;
        out["activated"] = true;
        out["model_id"] = reg["model_id"];
        for (auto it = load_info.begin(); it != load_info.end(); ++it) {
            out[it.key()] = it.value();
        }
        send_json(res, 200, out);
    }));

    svr_.Post("/api/models/hand_glove/load", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || !body.contains("onnx_path") || !body["onnx_path"].is_string()) {
            throw DatabaseHttpError(422, "INVALID_HAND_GLOVE_ONNX_PAYLOAD");
        }
        const std::string raw_path = body["onnx_path"].get<std::string>();
        if (raw_path.empty()) {
            throw DatabaseHttpError(422, "INVALID_HAND_GLOVE_ONNX_PAYLOAD");
        }
        const std::string model_version =
            (body.contains("model_version") && body["model_version"].is_string())
                ? body["model_version"].get<std::string>()
                : "nanodet-plus-m";
        const std::string expected_sha =
            (body.contains("expected_sha256") && body["expected_sha256"].is_string())
                ? to_lower_ascii(body["expected_sha256"].get<std::string>())
                : "";
        int frame_stride = 1;
        if (body.contains("frame_stride")) {
            if (!body["frame_stride"].is_number_integer()) {
                throw DatabaseHttpError(422, "INVALID_FRAME_STRIDE");
            }
            frame_stride = std::clamp(body["frame_stride"].get<int>(), 1, 8);
        }

        std::filesystem::path target_path = path_from_utf8(raw_path);
        if (target_path.is_relative() && !options_.data_dir.empty()) {
            target_path = options_.data_dir / "models" / target_path;
        }
        try {
            const json st = camera_.load_hand_glove_onnx_model(target_path, model_version, expected_sha, frame_stride);
            red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
            send_json(res, 200, json{
                {"loaded", true},
                {"hand_glove_detector", st}
            });
        } catch (const DatabaseHttpError&) {
            throw;
        } catch (const std::exception& exc) {
            const std::string msg = exc.what();
            if (msg.find("NOT_FOUND") != std::string::npos || msg.find("READ_FAILED") != std::string::npos) {
                throw DatabaseHttpError(404, msg);
            }
            if (msg.find("SHA256") != std::string::npos) {
                throw DatabaseHttpError(409, msg);
            }
            throw DatabaseHttpError(422, msg);
        }
    }));

    svr_.Get("/api/models/ood/status", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, camera_.label_ood_status_json());
    }));

    svr_.Post("/api/models/ood/calibration", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object()) {
            throw DatabaseHttpError(422, "INVALID_OOD_CALIBRATION_PAYLOAD");
        }

        const json cur_status = camera_.label_ood_status_json();
        if (cur_status.value("model_kind", "heuristic") == "heuristic") {
            throw DatabaseHttpError(409, "NO_ACTIVE_NEURAL_MODEL");
        }

        json next_status;
        std::filesystem::path saved_cal_path;
        if (body.contains("calibration_path") && body["calibration_path"].is_string()) {
            const std::string raw_path = body["calibration_path"].get<std::string>();
            if (raw_path.empty()) {
                throw DatabaseHttpError(422, "INVALID_OOD_CALIBRATION_PATH");
            }
            std::filesystem::path target_path = path_from_utf8(raw_path);
            if (target_path.is_relative() && !options_.data_dir.empty()) {
                target_path = options_.data_dir / "models" / target_path;
            }
            try {
                next_status = camera_.load_ood_calibration_file(target_path);
            } catch (const std::invalid_argument& ex) {
                throw DatabaseHttpError(409, ex.what());
            } catch (const std::exception& ex) {
                const std::string msg = ex.what();
                if (msg.find("NOT_FOUND") != std::string::npos) {
                    throw DatabaseHttpError(404, msg);
                }
                if (msg.find("NO_ACTIVE_NEURAL_MODEL") != std::string::npos) {
                    throw DatabaseHttpError(409, msg);
                }
                throw DatabaseHttpError(422, msg);
            }
        } else {
            std::string parse_err;
            auto parsed = vision::OodCalibration::from_json(body, parse_err);
            if (!parsed.has_value()) {
                throw DatabaseHttpError(422, parse_err.empty() ? "INVALID_OOD_CALIBRATION_PAYLOAD" : parse_err);
            }
            try {
                next_status = camera_.load_ood_calibration(*parsed);
            } catch (const std::invalid_argument& ex) {
                throw DatabaseHttpError(409, ex.what());
            } catch (const std::exception& ex) {
                const std::string msg = ex.what();
                if (msg.find("NO_ACTIVE_NEURAL_MODEL") != std::string::npos) {
                    throw DatabaseHttpError(409, msg);
                }
                throw DatabaseHttpError(422, msg);
            }
            const std::string active_ver = cur_status.value("model_version", "");
            if (!active_ver.empty()) {
                try {
                    saved_cal_path = training_.save_ood_calibration_for_version(active_ver, *parsed);
                } catch (...) {
                }
            }
        }

        red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
        send_json(res, 200, json{
            {"loaded", true},
            {"calibration_path", path_to_utf8(saved_cal_path)},
            {"label_ood_classifier", next_status}
        });
    }));

    svr_.Post("/api/models/ood/calibrate", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json cur_status = camera_.label_ood_status_json();
        if (cur_status.value("model_kind", "heuristic") == "heuristic") {
            throw DatabaseHttpError(409, "NO_ACTIVE_NEURAL_MODEL");
        }

        json body = json::object();
        if (!req.body.empty()) {
            try {
                body = json::parse(req.body);
            } catch (...) {
                throw DatabaseHttpError(422, "INVALID_JSON_BODY");
            }
            if (!body.is_object()) {
                throw DatabaseHttpError(422, "INVALID_OOD_CALIBRATE_PAYLOAD");
            }
        }

        double quantile = 0.98;
        if (body.contains("quantile")) {
            if (!body["quantile"].is_number() || body["quantile"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_OOD_QUANTILE");
            }
            quantile = body["quantile"].get<double>();
            if (!std::isfinite(quantile) || quantile < 0.5 || quantile >= 1.0) {
                throw DatabaseHttpError(422, "INVALID_OOD_QUANTILE");
            }
        }

        double prob_threshold = 0.85;
        if (body.contains("prob_threshold")) {
            if (!body["prob_threshold"].is_number() || body["prob_threshold"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_OOD_PROB_THRESHOLD");
            }
            prob_threshold = body["prob_threshold"].get<double>();
            if (!std::isfinite(prob_threshold) || prob_threshold < 0.0 || prob_threshold > 1.0) {
                throw DatabaseHttpError(422, "INVALID_OOD_PROB_THRESHOLD");
            }
        }

        double reg_lambda = 1e-3;
        if (body.contains("reg_lambda")) {
            if (!body["reg_lambda"].is_number() || body["reg_lambda"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_OOD_REG_LAMBDA");
            }
            reg_lambda = body["reg_lambda"].get<double>();
            if (!std::isfinite(reg_lambda) || reg_lambda <= 0.0) {
                throw DatabaseHttpError(422, "INVALID_OOD_REG_LAMBDA");
            }
        }

        bool fit_temperature = true;
        if (body.contains("fit_temperature")) {
            if (!body["fit_temperature"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_FIT_TEMPERATURE");
            }
            fit_temperature = body["fit_temperature"].get<bool>();
        }

        int max_per_class = 150;
        if (body.contains("max_samples_per_class")) {
            if (!body["max_samples_per_class"].is_number_integer()) {
                throw DatabaseHttpError(422, "INVALID_MAX_SAMPLES_PER_CLASS");
            }
            max_per_class = std::clamp(body["max_samples_per_class"].get<int>(), 1, 2000);
        }

        bool persist = true;
        if (body.contains("persist")) {
            if (!body["persist"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_PERSIST_FLAG");
            }
            persist = body["persist"].get<bool>();
        }

        const auto crops = training_.collect_ood_calibration_crops(max_per_class);
        vision::OodCalibration cal;
        try {
            cal = camera_.calibrate_active_model_ood(
                crops, quantile, prob_threshold, reg_lambda, fit_temperature
            );
        } catch (const std::invalid_argument& ex) {
            throw DatabaseHttpError(422, ex.what());
        } catch (const std::exception& ex) {
            const std::string msg = ex.what();
            if (msg.find("NO_ACTIVE_NEURAL_MODEL") != std::string::npos) {
                throw DatabaseHttpError(409, msg);
            }
            throw DatabaseHttpError(422, msg);
        }

        std::filesystem::path saved_path;
        if (persist && !cal.model_version.empty()) {
            try {
                saved_path = training_.save_ood_calibration_for_version(cal.model_version, cal);
            } catch (...) {
            }
        }

        red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
        send_json(res, 200, json{
            {"calibrated", true},
            {"calibration_path", path_to_utf8(saved_path)},
            {"calibration", cal.to_json(false)},
            {"label_ood_classifier", camera_.label_ood_status_json()}
        });
    }));

    svr_.Post("/api/models/ood/config", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object()) {
            throw DatabaseHttpError(422, "INVALID_OOD_CONFIG_PAYLOAD");
        }

        const json cur = camera_.label_ood_status_json();
        const json cur_cfg = cur.value("gate_config", json::object());
        vision::OodGateConfig cfg{};
        cfg.gate_enabled = cur_cfg.value("gate_enabled", true);
        cfg.alarm_streak_threshold = cur_cfg.value("alarm_streak_threshold", 5);
        cfg.require_calibration_for_onnx = cur_cfg.value("require_calibration_for_onnx", false);
        if (cur_cfg.contains("prob_threshold_override") && cur_cfg["prob_threshold_override"].is_number()) {
            cfg.prob_threshold_override = cur_cfg["prob_threshold_override"].get<double>();
        }
        if (cur_cfg.contains("energy_threshold_override") && cur_cfg["energy_threshold_override"].is_number()) {
            cfg.energy_threshold_override = cur_cfg["energy_threshold_override"].get<double>();
        }
        if (cur_cfg.contains("mahalanobis_threshold_override") && cur_cfg["mahalanobis_threshold_override"].is_number()) {
            cfg.mahalanobis_threshold_override = cur_cfg["mahalanobis_threshold_override"].get<double>();
        }

        if (body.contains("gate_enabled")) {
            if (!body["gate_enabled"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_GATE_ENABLED");
            }
            cfg.gate_enabled = body["gate_enabled"].get<bool>();
        }
        if (body.contains("alarm_streak_threshold")) {
            if (!body["alarm_streak_threshold"].is_number_integer()) {
                throw DatabaseHttpError(422, "INVALID_ALARM_STREAK_THRESHOLD");
            }
            const int v = body["alarm_streak_threshold"].get<int>();
            if (v < 1 || v > 30) {
                throw DatabaseHttpError(422, "INVALID_ALARM_STREAK_THRESHOLD");
            }
            cfg.alarm_streak_threshold = v;
        }
        if (body.contains("require_calibration_for_onnx")) {
            if (!body["require_calibration_for_onnx"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_REQUIRE_CALIBRATION_FLAG");
            }
            cfg.require_calibration_for_onnx = body["require_calibration_for_onnx"].get<bool>();
        }
        if (body.contains("prob_threshold_override")) {
            if (body["prob_threshold_override"].is_null()) {
                cfg.prob_threshold_override.reset();
            } else if (body["prob_threshold_override"].is_number() && !body["prob_threshold_override"].is_boolean()) {
                const double v = body["prob_threshold_override"].get<double>();
                if (!std::isfinite(v) || v < 0.0 || v > 1.0) {
                    throw DatabaseHttpError(422, "INVALID_PROB_THRESHOLD_OVERRIDE");
                }
                cfg.prob_threshold_override = v;
            } else {
                throw DatabaseHttpError(422, "INVALID_PROB_THRESHOLD_OVERRIDE");
            }
        }
        if (body.contains("energy_threshold_override")) {
            if (body["energy_threshold_override"].is_null()) {
                cfg.energy_threshold_override.reset();
            } else if (body["energy_threshold_override"].is_number() && !body["energy_threshold_override"].is_boolean()) {
                const double v = body["energy_threshold_override"].get<double>();
                if (!std::isfinite(v)) {
                    throw DatabaseHttpError(422, "INVALID_ENERGY_THRESHOLD_OVERRIDE");
                }
                cfg.energy_threshold_override = v;
            } else {
                throw DatabaseHttpError(422, "INVALID_ENERGY_THRESHOLD_OVERRIDE");
            }
        }
        if (body.contains("mahalanobis_threshold_override")) {
            if (body["mahalanobis_threshold_override"].is_null()) {
                cfg.mahalanobis_threshold_override.reset();
            } else if (body["mahalanobis_threshold_override"].is_number() && !body["mahalanobis_threshold_override"].is_boolean()) {
                const double v = body["mahalanobis_threshold_override"].get<double>();
                if (!std::isfinite(v) || v < 0.0) {
                    throw DatabaseHttpError(422, "INVALID_MAHALANOBIS_THRESHOLD_OVERRIDE");
                }
                cfg.mahalanobis_threshold_override = v;
            } else {
                throw DatabaseHttpError(422, "INVALID_MAHALANOBIS_THRESHOLD_OVERRIDE");
            }
        }

        const json st = camera_.set_ood_gate_config(cfg);
        red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
        send_json(res, 200, json{
            {"updated", true},
            {"label_ood_classifier", st}
        });
    }));

    svr_.Post("/api/models/:model_id/activate", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const std::string model_id = req.path_params.at("model_id");
        auto row_opt = db_.get_model_by_id(model_id);
        if (!row_opt.has_value()) {
            throw DatabaseHttpError(404, "MODEL_NOT_FOUND");
        }

        const std::string version = (*row_opt)["model_version"].get<std::string>();
        const std::string expected_sha = to_lower_ascii((*row_opt)["sha256"].get<std::string>());

        static const std::regex kVerRe("^[A-Za-z0-9._-]{1,80}$");
        if (version.empty() || version == "." || version == ".." ||
            version.find("..") != std::string::npos || !std::regex_match(version, kVerRe)) {
            throw DatabaseHttpError(404, "MODEL_ARTIFACT_UNAVAILABLE");
        }

        const auto runs_root = training_.runs_dir();
        const auto run_dir = runs_root / path_from_utf8(version);
        if (!is_path_within(runs_root, run_dir)) {
            throw DatabaseHttpError(404, "MODEL_ARTIFACT_UNAVAILABLE");
        }

        int input_size = kDefaultInputSize;
        int input_height = 0;
        std::string artifact_format;
        const auto metrics_path = run_dir / "metrics.json";
        std::vector<uint8_t> meta_bytes;
        if (read_file_bytes(metrics_path, meta_bytes) && !meta_bytes.empty()) {
            try {
                json meta = json::parse(meta_bytes.begin(), meta_bytes.end());
                if (meta.is_object()) {
                    if (meta.contains("input_width")) {
                        input_size = parse_optional_input_size(meta["input_width"], kDefaultInputSize);
                    } else if (meta.contains("input_size")) {
                        input_size = parse_optional_input_size(meta["input_size"], kDefaultInputSize);
                    }
                    if (meta.contains("input_height")) {
                        input_height = parse_optional_input_size(meta["input_height"], 0);
                    }
                    artifact_format = meta.value("artifact_format", "");
                }
            } catch (...) {
            }
        }

        std::error_code ec;
        const auto xml_candidate = run_dir / "label_classifier.xml";
        const auto onnx_candidate = run_dir / "label_classifier.onnx";
        std::filesystem::path chosen_path;
        std::string chosen_format;

        if (artifact_format == "opencv_mlp" && std::filesystem::is_regular_file(xml_candidate, ec)) {
            chosen_path = xml_candidate;
            chosen_format = "opencv_mlp";
        } else if (artifact_format == "onnx" && std::filesystem::is_regular_file(onnx_candidate, ec)) {
            chosen_path = onnx_candidate;
            chosen_format = "onnx";
        } else if (std::filesystem::is_regular_file(xml_candidate, ec)) {
            chosen_path = xml_candidate;
            chosen_format = "opencv_mlp";
        } else if (std::filesystem::is_regular_file(onnx_candidate, ec)) {
            chosen_path = onnx_candidate;
            chosen_format = "onnx";
        } else {
            throw DatabaseHttpError(404, "MODEL_ARTIFACT_UNAVAILABLE");
        }

        std::filesystem::path cal_candidate = run_dir / L"ood_calibration.json";
        if (!std::filesystem::is_regular_file(cal_candidate, ec)) {
            cal_candidate.clear();
        }

        json info = load_verified_model_into_camera(
            chosen_path, chosen_format, version, expected_sha, input_size, input_height, cal_candidate
        );
        red_rabbit_.trigger_challenge("MODEL_OR_OPTICS_CHANGE", "", actor);
        json resp{
            {"activated", true},
            {"mode", chosen_format == "opencv_mlp" ? "opencv_mlp" : "onnx_runtime"}
        };
        if (info.is_object()) {
            for (auto it = info.begin(); it != info.end(); ++it) {
                resp[it.key()] = it.value();
            }
        }
        send_json(res, 200, resp);
    }));

    svr_.Delete("/api/models/:model_id", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const std::string model_id = req.path_params.at("model_id");
        const std::string active_ver = camera_.get_state_json().value("model_version", "");
        send_json(res, 200, db_.delete_model(model_id, active_ver));
    }));

    // 7. Cycles, Events & Evidence CRUD
    svr_.Get("/api/cycles", wrap([this](const httplib::Request& req, httplib::Response& res) {
        send_json(res, 200, db_.list_cycles(get_int_query(req, "limit", 100)));
    }));

    svr_.Post("/api/cycles/purge", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || body.size() != 2 ||
            !body.contains("scope") || !body["scope"].is_string() ||
            !body.contains("confirm") || !body["confirm"].is_boolean() ||
            body["confirm"].get<bool>() != true) {
            throw DatabaseHttpError(422, "INVALID_PURGE_REQUEST");
        }
        const std::string scope = body["scope"].get<std::string>();
        if (scope != "simulated_only" && scope != "all") {
            throw DatabaseHttpError(422, "INVALID_PURGE_SCOPE");
        }
        if (scope == "all") {
            camera_.discard_pending_transitions();
        }
        send_json(res, 200, db_.purge_cycles(scope));
    }));

    svr_.Delete("/api/cycles/:cycle_id", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const std::string cycle_id = StationDatabase::validate_cycle_id(req.path_params.at("cycle_id"));
        camera_.discard_pending_transitions({cycle_id});
        send_json(res, 200, db_.delete_cycle(cycle_id));
    }));

    svr_.Get("/api/cycles/:cycle_id/events", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string cycle_id = req.path_params.at("cycle_id");
        send_json(res, 200, db_.list_cycle_events(cycle_id, get_int_query(req, "limit", 200)));
    }));

    svr_.Post("/api/cycles/:cycle_id/evidence/:kind", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_station_or_admin_token(
            get_opt_header(req, "X-Station-Token"),
            get_opt_header(req, "X-Admin-Token")
        );
        const std::string cycle_id = StationDatabase::validate_cycle_id(req.path_params.at("cycle_id"));
        const std::string kind = req.path_params.at("kind");
        static const std::set<std::string> kKinds = {"initial_label", "A", "B", "C", "anomaly"};
        if (kKinds.find(kind) == kKinds.end()) {
            throw DatabaseHttpError(422, "INVALID_EVIDENCE_KIND");
        }

        auto raw_frame_id = get_opt_header(req, "X-Frame-Id");
        auto raw_cam_meta = get_opt_header(req, "X-Camera-Metadata");
        auto raw_captured_utc = get_opt_header(req, "X-Captured-At-Utc");
        if (!raw_frame_id.has_value() || !raw_cam_meta.has_value() || !raw_captured_utc.has_value()) {
            throw DatabaseHttpError(422, "MISSING_EVIDENCE_HEADERS");
        }

        int64_t frame_id = -1;
        try {
            std::size_t pos = 0;
            frame_id = std::stoll(*raw_frame_id, &pos);
            if (pos != raw_frame_id->size() || frame_id < 0) {
                throw DatabaseHttpError(422, "INVALID_X_FRAME_ID");
            }
        } catch (const DatabaseHttpError&) {
            throw;
        } catch (...) {
            throw DatabaseHttpError(422, "INVALID_X_FRAME_ID");
        }

        std::optional<double> cap_mono;
        if (auto raw_mono = get_opt_header(req, "X-Capture-Monotonic-S"); raw_mono.has_value() && !raw_mono->empty()) {
            try {
                std::size_t pos = 0;
                double v = std::stod(*raw_mono, &pos);
                if (pos != raw_mono->size() || !std::isfinite(v)) {
                    throw DatabaseHttpError(422, "INVALID_X_CAPTURE_MONOTONIC_S");
                }
                cap_mono = v;
            } catch (const DatabaseHttpError&) {
                throw;
            } catch (...) {
                throw DatabaseHttpError(422, "INVALID_X_CAPTURE_MONOTONIC_S");
            }
        }

        auto x_pipe_color = get_opt_header(req, "X-Pipe-Color");
        if (x_pipe_color.has_value()) {
            static const std::set<std::string> kColors = {"yellow", "blue", "other", "unknown"};
            if (kColors.find(*x_pipe_color) == kColors.end()) {
                throw DatabaseHttpError(422, "INVALID_PIPE_COLOR");
            }
        }

        auto x_label_status = get_opt_header(req, "X-Label-Status");
        if (x_label_status.has_value()) {
            static const std::set<std::string> kStates = {"complete", "crossed_out", "covered", "absent", "unknown"};
            if (kStates.find(*x_label_status) == kStates.end()) {
                throw DatabaseHttpError(422, "INVALID_LABEL_STATUS");
            }
        }

        const std::string captured_utc_iso = validate_iso_timestamp_with_tz(*raw_captured_utc);
        json cam_meta;
        try {
            cam_meta = json::parse(*raw_cam_meta);
        } catch (...) {
            throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
        }
        if (!cam_meta.is_object() || cam_meta.empty()) {
            throw DatabaseHttpError(422, "INVALID_CAPTURE_TIMESTAMP_OR_CAMERA_METADATA");
        }

        auto cl_hdr = get_opt_header(req, "Content-Length");
        if (!cl_hdr.has_value() || !is_all_digits(*cl_hdr)) {
            throw DatabaseHttpError(413, "EVIDENCE_SIZE_LIMIT_OR_LENGTH_REQUIRED");
        }
        std::size_t expected_len = 0;
        try {
            expected_len = std::stoull(*cl_hdr);
        } catch (...) {
            throw DatabaseHttpError(413, "EVIDENCE_SIZE_LIMIT_OR_LENGTH_REQUIRED");
        }
        if (expected_len > kMaxEvidenceUploadBytes) {
            throw DatabaseHttpError(413, "EVIDENCE_SIZE_LIMIT_OR_LENGTH_REQUIRED");
        }
        if (req.body.empty() || req.body.size() != expected_len || req.body.size() > kMaxEvidenceUploadBytes) {
            throw DatabaseHttpError(400, "INVALID_EVIDENCE_BODY");
        }

        std::string mime = to_lower_ascii(req.get_header_value("Content-Type"));
        if (auto semi = mime.find(';'); semi != std::string::npos) {
            mime = mime.substr(0, semi);
        }
        const auto first = mime.find_first_not_of(' ');
        const auto last = mime.find_last_not_of(' ');
        mime = (first == std::string::npos) ? "" : mime.substr(first, last - first + 1);

        std::vector<uint8_t> body_bytes(req.body.begin(), req.body.end());
        validate_and_decode_evidence_image(body_bytes, mime);

        send_json(res, 200, db_.upload_evidence_bytes(
            cycle_id,
            kind,
            body_bytes,
            mime,
            frame_id,
            cap_mono,
            x_pipe_color,
            get_opt_header(req, "X-Operation-Event"),
            get_opt_header(req, "X-Operator-Anon-Id"),
            cam_meta,
            x_label_status,
            captured_utc_iso
        ));
    }));

    svr_.Get("/api/evidence", wrap([this](const httplib::Request& req, httplib::Response& res) {
        send_json(res, 200, db_.list_evidence(get_int_query(req, "limit", 100)));
    }));

    svr_.Get("/api/evidence/:evidence_id/file", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const auto ev_file = db_.get_evidence_file(req.path_params.at("evidence_id"));
        std::vector<uint8_t> bytes;
        if (!read_file_bytes(ev_file.file_path, bytes)) {
            throw DatabaseHttpError(404, "EVIDENCE_NOT_FOUND");
        }
        res.status = 200;
        res.set_header("Cache-Control", "no-store");
        res.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), ev_file.mime_type);
    }));

    svr_.Delete("/api/evidence/:evidence_id", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        send_json(res, 200, db_.delete_evidence(req.path_params.at("evidence_id")));
    }));

    // 8. Dataset CRUD, Skip-Frame Auto-Capture & Pure C++ Training
    svr_.Get("/api/dataset/summary", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, training_.summary());
    }));

    svr_.Get("/api/dataset/samples", wrap([this](const httplib::Request& req, httplib::Response& res) {
        std::string label = req.has_param("label") ? req.get_param_value("label") : "";
        std::string q = req.has_param("q") ? req.get_param_value("q") : "";
        int limit = get_int_query(req, "limit", 200);
        send_json(res, 200, training_.list_samples(label, q, limit));
    }));

    svr_.Get("/api/dataset/auto_capture", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, camera_.get_auto_capture().to_json());
    }));

    svr_.Post("/api/dataset/auto_capture", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object()) {
            throw DatabaseHttpError(422, "INVALID_AUTO_CAPTURE_PAYLOAD");
        }

        AutoCaptureConfig cfg = camera_.get_auto_capture();
        if (body.contains("enabled")) {
            if (!body["enabled"].is_boolean()) throw DatabaseHttpError(422, "INVALID_ENABLED_FLAG");
            cfg.enabled = body["enabled"].get<bool>();
        }
        if (body.contains("every_n_frames") && !body["every_n_frames"].is_null()) {
            if (!body["every_n_frames"].is_number_integer() || body["every_n_frames"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_EVERY_N_FRAMES");
            }
            int n = body["every_n_frames"].get<int>();
            if (n < 1 || n > 1000) throw DatabaseHttpError(422, "EVERY_N_FRAMES_OUT_OF_RANGE");
            cfg.every_n_frames = n;
        }
        if (body.contains("label") && !body["label"].is_null()) {
            if (!body["label"].is_string()) throw DatabaseHttpError(422, "INVALID_LABEL");
            std::string lbl = body["label"].get<std::string>();
            if (std::find(kAllDatasetLabels.begin(), kAllDatasetLabels.end(), lbl) == kAllDatasetLabels.end()) {
                throw DatabaseHttpError(422, "INVALID_SAMPLE_LABEL:" + lbl);
            }
            cfg.label = lbl;
        }
        if (body.contains("pipe_color") && !body["pipe_color"].is_null()) {
            if (!body["pipe_color"].is_string()) throw DatabaseHttpError(422, "INVALID_PIPE_COLOR");
            std::string col = body["pipe_color"].get<std::string>();
            if (col != "yellow" && col != "blue") {
                throw DatabaseHttpError(422, "INVALID_PIPE_COLOR:" + col);
            }
            cfg.pipe_color = col;
        }
        if (body.contains("physical_tube_id") && !body["physical_tube_id"].is_null()) {
            if (!body["physical_tube_id"].is_string()) throw DatabaseHttpError(422, "INVALID_PHYSICAL_TUBE_ID");
            std::string tid = body["physical_tube_id"].get<std::string>();
            cfg.physical_tube_id = tid.empty() ? "UNGROUPED" : tid;
        }
        if (body.contains("max_samples") && !body["max_samples"].is_null()) {
            if (!body["max_samples"].is_number_integer() || body["max_samples"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_MAX_SAMPLES");
            }
            int m = body["max_samples"].get<int>();
            if (m < 1 || m > 10000) throw DatabaseHttpError(422, "MAX_SAMPLES_OUT_OF_RANGE");
            cfg.max_samples = m;
        }
        if (body.contains("min_diff_mad") && !body["min_diff_mad"].is_null()) {
            if (!body["min_diff_mad"].is_number() || body["min_diff_mad"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_MIN_DIFF_MAD");
            }
            double mad = body["min_diff_mad"].get<double>();
            if (!std::isfinite(mad) || mad < 0.0 || mad > 255.0) {
                throw DatabaseHttpError(422, "MIN_DIFF_MAD_OUT_OF_RANGE");
            }
            cfg.min_diff_mad = mad;
        }
        if (cfg.enabled && body.value("reset_count", true)) {
            cfg.captured_count = 0;
        }

        send_json(res, 200, camera_.set_auto_capture(cfg).to_json());
    }));

    svr_.Post("/api/dataset/upload", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));

        auto raw_label = get_opt_header(req, "X-Sample-Label");
        if (!raw_label.has_value()) {
            throw DatabaseHttpError(422, "MISSING_X_SAMPLE_LABEL");
        }
        const std::string clean_label = clean_optional_header(raw_label, 64).value_or("");
        if (std::find(kAllDatasetLabels.begin(), kAllDatasetLabels.end(), clean_label) == kAllDatasetLabels.end()) {
            throw DatabaseHttpError(422, "INVALID_SAMPLE_LABEL:" + clean_label);
        }

        std::string clean_color = to_lower_ascii(
            clean_optional_header(get_opt_header(req, "X-Pipe-Color"), 32).value_or("yellow")
        );
        if (clean_color.empty()) clean_color = "yellow";
        if (clean_color != "yellow" && clean_color != "blue") {
            throw DatabaseHttpError(422, "INVALID_PIPE_COLOR:" + clean_color);
        }

        if (auto cl = get_opt_header(req, "Content-Length"); cl.has_value()) {
            if (!is_all_digits(*cl) || std::stoull(*cl) > kMaxEvidenceUploadBytes) {
                throw DatabaseHttpError(413, "SAMPLE_IMAGE_TOO_LARGE_OR_INVALID_LENGTH");
            }
        }
        if (req.body.empty()) {
            throw DatabaseHttpError(400, "EMPTY_SAMPLE_IMAGE_BODY");
        }
        if (req.body.size() > kMaxEvidenceUploadBytes) {
            throw DatabaseHttpError(413, "SAMPLE_IMAGE_EXCEEDS_MAX_SIZE");
        }

        std::string mime = to_lower_ascii(req.get_header_value("Content-Type"));
        if (auto semi = mime.find(';'); semi != std::string::npos) mime = mime.substr(0, semi);
        const auto f = mime.find_first_not_of(' ');
        const auto l = mime.find_last_not_of(' ');
        mime = (f == std::string::npos) ? "" : mime.substr(f, l - f + 1);
        if (!mime.empty() && mime != "image/jpeg" && mime != "image/jpg" &&
            mime != "image/png" && mime != "application/octet-stream") {
            throw DatabaseHttpError(415, "ONLY_JPEG_OR_PNG_SAMPLE_IS_ACCEPTED");
        }

        const bool is_jpeg = req.body.size() >= 2 &&
                             static_cast<uint8_t>(req.body[0]) == 0xFF &&
                             static_cast<uint8_t>(req.body[1]) == 0xD8;
        const bool is_png = req.body.size() >= 8 &&
                            req.body.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0;
        if (!(is_jpeg || is_png)) {
            throw DatabaseHttpError(415, "ONLY_VALID_JPEG_OR_PNG_BYTES_ACCEPTED");
        }

        const std::string tube_id = clean_optional_header(get_opt_header(req, "X-Tube-Id")).value_or("");
        const std::string batch_id = clean_optional_header(get_opt_header(req, "X-Batch-Id")).value_or("");
        send_json(res, 200, training_.upload_sample_bytes(req.body, clean_label, clean_color, tube_id, batch_id));
    }));

    svr_.Post("/api/dataset/capture", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || !body.contains("label") || !body["label"].is_string()) {
            throw DatabaseHttpError(422, "INVALID_DATASET_CAPTURE_PAYLOAD");
        }
        static const std::set<std::string> kAllowed = {
            "label", "pipe_color", "physical_tube_id", "batch_id", "box"
        };
        for (auto it = body.begin(); it != body.end(); ++it) {
            if (kAllowed.find(it.key()) == kAllowed.end()) {
                throw DatabaseHttpError(422, "EXTRA_FIELD_FORBIDDEN:" + it.key());
            }
        }

        const std::string label = body["label"].get<std::string>();
        if (std::find(kAllDatasetLabels.begin(), kAllDatasetLabels.end(), label) == kAllDatasetLabels.end()) {
            throw DatabaseHttpError(422, "INVALID_SAMPLE_LABEL:" + label);
        }
        std::string pipe_color = "yellow";
        if (body.contains("pipe_color") && !body["pipe_color"].is_null()) {
            if (!body["pipe_color"].is_string()) throw DatabaseHttpError(422, "INVALID_PIPE_COLOR");
            pipe_color = body["pipe_color"].get<std::string>();
            if (pipe_color != "yellow" && pipe_color != "blue") {
                throw DatabaseHttpError(422, "INVALID_PIPE_COLOR:" + pipe_color);
            }
        }
        std::string tube_id;
        if (body.contains("physical_tube_id") && !body["physical_tube_id"].is_null()) {
            if (!body["physical_tube_id"].is_string()) throw DatabaseHttpError(422, "INVALID_PHYSICAL_TUBE_ID");
            tube_id = body["physical_tube_id"].get<std::string>();
        }
        std::string batch_id;
        if (body.contains("batch_id") && !body["batch_id"].is_null()) {
            if (!body["batch_id"].is_string()) throw DatabaseHttpError(422, "INVALID_BATCH_ID");
            batch_id = body["batch_id"].get<std::string>();
        }
        std::optional<Box> explicit_box;
        if (body.contains("box")) {
            explicit_box = validate_explicit_crop_box(body["box"]);
        }

        auto snap = camera_.get_atomic_capture(explicit_box, true);
        if (!snap.has_frame() || snap.crop_bgr.empty()) {
            throw DatabaseHttpError(503, "当前未获取到实时摄像头画面");
        }

        json sample = training_.capture_sample(
            snap.frame_bgr,
            snap.crop_bgr,
            label,
            pipe_color,
            tube_id,
            batch_id,
            snap.crop_box
        );

        if (snap.frame_jpeg.empty()) {
            sample["evidence_archived"] = false;
            sample["evidence_error"] = "EVIDENCE_ENCODE_FAILED";
            send_json(res, 200, sample);
            return;
        }

        try {
            json arch = db_.archive_dataset_sample_evidence(
                snap.frame_jpeg,
                sample,
                label,
                pipe_color,
                snap.state,
                LiveCameraEngine::generate_live_cycle_id()
            );
            for (auto it = arch.begin(); it != arch.end(); ++it) {
                sample[it.key()] = it.value();
            }
        } catch (const std::exception& exc) {
            sample["evidence_archived"] = false;
            sample["evidence_error"] = std::string("EVIDENCE_ARCHIVE_FAILED:") + exc.what();
        }
        send_json(res, 200, sample);
    }));

    svr_.Post("/api/dataset/bootstrap", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        cv::Mat live_crop = camera_.get_tracked_crop_bgr();
        std::optional<cv::Mat> opt_crop;
        if (!live_crop.empty()) opt_crop = live_crop;
        send_json(res, 200, training_.bootstrap_standard_dataset(opt_crop));
    }));

    svr_.Post("/api/dataset/batch_delete", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object()) throw DatabaseHttpError(422, "INVALID_BATCH_DELETE_PAYLOAD");
        for (auto it = body.begin(); it != body.end(); ++it) {
            if (it.key() != "sample_ids" && it.key() != "label") {
                throw DatabaseHttpError(422, "EXTRA_FIELD_FORBIDDEN:" + it.key());
            }
        }

        std::vector<std::string> sample_ids;
        if (body.contains("sample_ids") && !body["sample_ids"].is_null()) {
            if (!body["sample_ids"].is_array()) throw DatabaseHttpError(422, "INVALID_SAMPLE_IDS");
            static const std::regex kSampleIdRe("^[A-Za-z0-9_-]{3,80}$");
            for (const auto& item : body["sample_ids"]) {
                if (!item.is_string()) throw DatabaseHttpError(422, "INVALID_SAMPLE_ID");
                std::string sid = item.get<std::string>();
                if (!std::regex_match(sid, kSampleIdRe)) {
                    throw DatabaseHttpError(422, "INVALID_SAMPLE_ID:" + sid);
                }
                sample_ids.push_back(std::move(sid));
            }
        }

        std::string label_filter;
        if (body.contains("label") && !body["label"].is_null()) {
            if (!body["label"].is_string()) throw DatabaseHttpError(422, "INVALID_LABEL");
            label_filter = body["label"].get<std::string>();
            if (std::find(kAllDatasetLabels.begin(), kAllDatasetLabels.end(), label_filter) == kAllDatasetLabels.end()) {
                throw DatabaseHttpError(422, "INVALID_SAMPLE_LABEL:" + label_filter);
            }
        }

        if (sample_ids.empty() && label_filter.empty()) {
            throw DatabaseHttpError(422, "MUST_SPECIFY_SAMPLE_IDS_OR_LABEL");
        }
        send_json(res, 200, training_.batch_delete_samples(sample_ids, label_filter));
    }));

    svr_.Post("/api/dataset/clear", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        send_json(res, 200, training_.clear_dataset());
    }));

    svr_.Get("/api/dataset/coco", wrap([this](const httplib::Request&, httplib::Response& res) {
        const auto coco_path = training_.coco_json_path();
        std::vector<uint8_t> bytes;
        if (!read_file_bytes(coco_path, bytes) || bytes.empty()) {
            send_json(res, 200, json{
                {"images", json::array()},
                {"annotations", json::array()},
                {"categories", json::array()}
            });
            return;
        }
        res.status = 200;
        res.set_header("Content-Disposition", "attachment; filename=\"igbt_annotations_coco.json\"");
        res.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "application/json; charset=utf-8");
    }));

    svr_.Get("/api/dataset/sample/:sample_id/image", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string sample_id = req.path_params.at("sample_id");
        static const std::regex kSampleIdRe("^[A-Za-z0-9_-]{3,80}$");
        if (!std::regex_match(sample_id, kSampleIdRe)) {
            throw DatabaseHttpError(404, "SAMPLE_NOT_FOUND");
        }
        auto img_path = training_.get_sample_image_path(sample_id);
        std::vector<uint8_t> bytes;
        if (!img_path.has_value() || !read_file_bytes(*img_path, bytes) || bytes.empty()) {
            throw DatabaseHttpError(404, "SAMPLE_NOT_FOUND");
        }
        res.status = 200;
        res.set_header("Cache-Control", "no-store");
        res.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "image/jpeg");
    }));

    svr_.Patch("/api/dataset/sample/:sample_id", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const std::string sample_id = req.path_params.at("sample_id");
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object()) throw DatabaseHttpError(422, "INVALID_SAMPLE_UPDATE_PAYLOAD");
        static const std::set<std::string> kAllowed = {
            "label", "physical_tube_id", "batch_id", "pipe_color"
        };
        for (auto it = body.begin(); it != body.end(); ++it) {
            if (kAllowed.find(it.key()) == kAllowed.end()) {
                throw DatabaseHttpError(422, "EXTRA_FIELD_FORBIDDEN:" + it.key());
            }
        }

        auto opt_str_field = [&](const char* k) -> std::optional<std::string> {
            if (!body.contains(k) || body[k].is_null()) return std::nullopt;
            if (!body[k].is_string()) throw DatabaseHttpError(422, std::string("INVALID_FIELD:") + k);
            return body[k].get<std::string>();
        };

        auto new_label = opt_str_field("label");
        if (new_label.has_value() &&
            std::find(kAllDatasetLabels.begin(), kAllDatasetLabels.end(), *new_label) == kAllDatasetLabels.end()) {
            throw DatabaseHttpError(422, "INVALID_SAMPLE_LABEL:" + *new_label);
        }
        auto new_color = opt_str_field("pipe_color");
        if (new_color.has_value() && *new_color != "yellow" && *new_color != "blue") {
            throw DatabaseHttpError(422, "INVALID_PIPE_COLOR:" + *new_color);
        }

        try {
            send_json(res, 200, training_.update_sample_annotation(
                sample_id,
                new_label,
                opt_str_field("physical_tube_id"),
                opt_str_field("batch_id"),
                new_color
            ));
        } catch (const std::out_of_range&) {
            throw DatabaseHttpError(404, "SAMPLE_NOT_FOUND");
        }
    }));

    svr_.Delete("/api/dataset/sample/:sample_id", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        try {
            send_json(res, 200, training_.delete_sample(req.path_params.at("sample_id")));
        } catch (const std::out_of_range&) {
            throw DatabaseHttpError(404, "SAMPLE_NOT_FOUND");
        }
    }));

    svr_.Get("/api/training/status", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, training_.status());
    }));

    svr_.Post("/api/training/start", wrap([this](const httplib::Request& req, httplib::Response& res) {
        auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        json body = req.body.empty() ? json::object() : parse_json_body_or_throw(req);
        if (!body.is_object()) throw DatabaseHttpError(422, "INVALID_TRAINING_PAYLOAD");
        static const std::set<std::string> kAllowed = {
            "epochs", "input_size", "batch_size", "learning_rate", "resume_from_latest", "use_augmentation"
        };
        for (auto it = body.begin(); it != body.end(); ++it) {
            if (kAllowed.find(it.key()) == kAllowed.end()) {
                throw DatabaseHttpError(422, "EXTRA_FIELD_FORBIDDEN:" + it.key());
            }
        }

        auto get_int_bounded = [&](const char* k, int def_v, int min_v, int max_v) -> int {
            if (!body.contains(k)) return def_v;
            if (!body[k].is_number_integer() || body[k].is_boolean()) {
                throw DatabaseHttpError(422, std::string("INVALID_INT_FIELD:") + k);
            }
            int v = body[k].get<int>();
            if (v < min_v || v > max_v) {
                throw DatabaseHttpError(422, std::string("FIELD_OUT_OF_RANGE:") + k);
            }
            return v;
        };

        const int epochs = get_int_bounded("epochs", 8, 2, 50);
        const int input_size = get_int_bounded("input_size", kDefaultInputSize, 64, 320);
        const int batch_size = get_int_bounded("batch_size", 8, 2, 64);

        double lr = 0.0025;
        if (body.contains("learning_rate")) {
            if (!body["learning_rate"].is_number() || body["learning_rate"].is_boolean()) {
                throw DatabaseHttpError(422, "INVALID_LEARNING_RATE");
            }
            lr = body["learning_rate"].get<double>();
            if (!std::isfinite(lr) || lr <= 0.0 || lr > 0.1) {
                throw DatabaseHttpError(422, "LEARNING_RATE_OUT_OF_RANGE");
            }
        }

        bool resume_from_latest = true;
        if (body.contains("resume_from_latest")) {
            if (!body["resume_from_latest"].is_boolean()) throw DatabaseHttpError(422, "INVALID_RESUME_FLAG");
            resume_from_latest = body["resume_from_latest"].get<bool>();
        }
        bool use_augmentation = true;
        if (body.contains("use_augmentation")) {
            if (!body["use_augmentation"].is_boolean()) throw DatabaseHttpError(422, "INVALID_AUGMENTATION_FLAG");
            use_augmentation = body["use_augmentation"].get<bool>();
        }

        try {
            json started = training_.start_training(
                epochs,
                input_size,
                batch_size,
                lr,
                resume_from_latest,
                use_augmentation,
                [this](const json& summary) { return on_training_completed(summary); }
            );
            send_json(res, 200, started);
        } catch (const std::exception& exc) {
            throw DatabaseHttpError(422, exc.what());
        }
    }));

    // 10. Red Rabbit Golden-Sample Challenge & Hash-Chain Audit (v4.3 AEC-Q101)
    svr_.Get("/api/red_rabbit/status", wrap([this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, red_rabbit_.status_json());
    }));

    svr_.Get("/api/red_rabbit/samples", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const bool active_only = get_int_query(req, "active_only", 0) != 0;
        send_json(res, 200, json{{"samples", db_.rr_list_samples(active_only)}});
    }));

    svr_.Post("/api/red_rabbit/samples", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        send_json(res, 200, db_.rr_upsert_sample(body, actor));
    }));

    svr_.Post("/api/red_rabbit/trigger", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = req.body.empty() ? json::object() : parse_json_body_or_throw(req);
        if (!body.is_object()) {
            throw DatabaseHttpError(422, "INVALID_TRIGGER_PAYLOAD");
        }
        const std::string reason = body.value("reason", "MANUAL_AUDIT");
        const std::string lot_id = body.value("lot_id", "");
        send_json(res, 200, red_rabbit_.trigger_challenge(reason, lot_id, actor));
    }));

    svr_.Post("/api/red_rabbit/sessions/start", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const json body = req.body.empty() ? json::object() : parse_json_body_or_throw(req);
        if (!body.is_object()) {
            throw DatabaseHttpError(422, "INVALID_SESSION_START_PAYLOAD");
        }
        const std::string operator_id = body.value("operator_id", "OP_DEFAULT");
        const std::string lot_id = body.value("lot_id", "");
        const json cam_state = camera_.get_state_json();
        std::string model_sha = cam_state.value("model_sha256", "");
        if (model_sha.empty()) {
            model_sha = cam_state.value("model_version", "heuristic");
        }
        send_json(res, 200, red_rabbit_.start_session(operator_id, lot_id, model_sha));
    }));

    svr_.Post("/api/red_rabbit/sessions/:session_id/arm", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string session_id = req.path_params.at("session_id");
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || !body.contains("sample_uid") || !body["sample_uid"].is_string()) {
            throw DatabaseHttpError(422, "MISSING_SAMPLE_UID");
        }
        send_json(res, 200, red_rabbit_.arm_sample(session_id, body["sample_uid"].get<std::string>()));
    }));

    svr_.Post("/api/red_rabbit/sessions/:session_id/attempt", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string session_id = req.path_params.at("session_id");
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object() || !body.contains("sample_uid") || !body["sample_uid"].is_string()) {
            throw DatabaseHttpError(422, "MISSING_SAMPLE_UID");
        }
        const std::string sample_uid = body["sample_uid"].get<std::string>();
        const bool use_live_state = body.value("use_live_state", false);
        std::string actual_verdict = body.value("actual_verdict", "");
        std::string evidence_dir = body.value("evidence_dir", "");

        if (use_live_state) {
            const json live = camera_.get_state_json();
            const std::string live_outcome = live.value("outcome", "");
            if (live_outcome != "PASS" && live_outcome != "ALARM") {
                throw DatabaseHttpError(409, "RED_RABBIT_LIVE_VERDICT_NOT_READY");
            }
            actual_verdict = live_outcome;
            if (evidence_dir.empty() && live.contains("latest_evidence_url") && live["latest_evidence_url"].is_string()) {
                evidence_dir = live["latest_evidence_url"].get<std::string>();
            }
        }

        std::optional<double> ood_energy;
        if (body.contains("ood_energy") && body["ood_energy"].is_number()) {
            ood_energy = body["ood_energy"].get<double>();
        }
        std::optional<double> ood_mahalanobis;
        if (body.contains("ood_mahalanobis") && body["ood_mahalanobis"].is_number()) {
            ood_mahalanobis = body["ood_mahalanobis"].get<double>();
        }
        send_json(
            res,
            200,
            red_rabbit_.record_attempt(
                session_id,
                sample_uid,
                actual_verdict,
                evidence_dir,
                ood_energy,
                ood_mahalanobis
            )
        );
    }));

    svr_.Post("/api/red_rabbit/unlock", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string admin_actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const json body = parse_json_body_or_throw(req);
        if (!body.is_object()) {
            throw DatabaseHttpError(422, "INVALID_UNLOCK_PAYLOAD");
        }
        std::string actor_id = body.value("actor_id", "");
        if (actor_id.empty()) actor_id = admin_actor;
        const std::string actor_role = body.value("actor_role", "ENGINEER");
        const std::string reason_code = body.value("reason_code", "");
        const std::string comment = body.value("comment", "");
        send_json(res, 200, red_rabbit_.unlock_hard_lock(actor_id, actor_role, reason_code, comment));
    }));

    svr_.Get("/api/red_rabbit/audit", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const int limit = get_int_query(req, "limit", 30);
        send_json(res, 200, red_rabbit_.audit_summary_json(limit));
    }));

    // 11. Industrial HAL DIO & Fail-Safe Watchdog Telemetry (v4.3 ADAM-6050/6060)
    svr_.Get("/api/hal/dio", wrap([this](const httplib::Request&, httplib::Response& res) {
        const DioHealthStatus dio = interlock_ ? interlock_->health_status() : DioHealthStatus{};
        send_json(res, 200, dio_health_to_json(dio));
    }));

    svr_.Post("/api/hal/dio/reset_weld_fault", wrap([this](const httplib::Request& req, httplib::Response& res) {
        const std::string actor = auth_.require_admin_token(get_opt_header(req, "X-Admin-Token"));
        const bool reset_done = interlock_ ? interlock_->acknowledge_relay_fault() : false;
        if (reset_done) {
            try {
                db_.rr_append_lock_audit(
                    "ENGINEER_UNLOCK",
                    actor,
                    "ENGINEER",
                    "RELAY_WELD_FAULT_RESET",
                    "工程师现场检查继电器触点状态并复位粘连告警",
                    std::nullopt,
                    default_wall_clock_ms()
                );
            } catch (...) {
            }
        }
        const DioHealthStatus dio = interlock_ ? interlock_->health_status() : DioHealthStatus{};
        json out = dio_health_to_json(dio);
        out["reset_applied"] = reset_done;
        send_json(res, 200, out);
    }));
}

void StationHttpServer::register_static_and_spa_routes() {
    auto serve_index_html = [this](httplib::Response& res) {
        const auto index_path = options_.web_dist_dir / "index.html";
        std::vector<uint8_t> bytes;
        if (!read_file_bytes(index_path, bytes) || bytes.empty()) {
            send_error(res, 503, "FRONTEND_BUILD_MISSING: run npm ci and npm run build in frontend/igbt-station-ui");
            return;
        }
        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "text/html; charset=utf-8");
    };

    svr_.Get("/", [serve_index_html](const httplib::Request&, httplib::Response& res) {
        serve_index_html(res);
    });

    // Unicode-safe static file & SPA fallback handler (handles Chinese Windows workspace paths cleanly)
    svr_.set_error_handler([this, serve_index_html](const httplib::Request& req, httplib::Response& res) {
        if (res.status != 404) {
            return;
        }
        if (req.path.rfind("/api/", 0) == 0 || req.path == "/api") {
            if (res.body.empty()) {
                send_error(res, 404, "NOT_FOUND");
            }
            return;
        }
        if (req.method != "GET" && req.method != "HEAD") {
            return;
        }

        std::string rel = req.path;
        while (!rel.empty() && rel.front() == '/') rel.erase(rel.begin());
        if (!rel.empty() && rel.find("..") == std::string::npos) {
            const auto candidate = options_.web_dist_dir / path_from_utf8(rel);
            std::error_code ec;
            if (is_path_within(options_.web_dist_dir, candidate) &&
                std::filesystem::is_regular_file(candidate, ec)) {
                std::vector<uint8_t> bytes;
                if (read_file_bytes(candidate, bytes)) {
                    res.status = 200;
                    if (rel.rfind("assets/", 0) == 0) {
                        res.set_header("Cache-Control", "public, max-age=31536000, immutable");
                    } else {
                        res.set_header("Cache-Control", "no-cache");
                    }
                    res.set_content(
                        reinterpret_cast<const char*>(bytes.data()),
                        bytes.size(),
                        guess_mime_type(candidate)
                    );
                    return;
                }
            }
        }

        // SPA fallback for client-side routes (e.g., /showcase)
        if (rel.find('.') == std::string::npos) {
            serve_index_html(res);
        }
    });
}

bool StationHttpServer::listen_blocking() {
    stopping_.store(false);
    autoload_latest_verified_model();
    if (options_.enable_live_camera) {
        camera_.start();
    }

    int port = options_.port;
    if (port <= 0) {
        port = svr_.bind_to_any_port(options_.host.c_str());
        if (port <= 0) return false;
    } else {
        if (!svr_.bind_to_port(options_.host.c_str(), port)) {
            return false;
        }
    }
    bound_port_.store(port);
    running_.store(true);
    start_maintenance_thread();
    const bool ok = svr_.listen_after_bind();
    running_.store(false);
    stop_maintenance_thread();
    if (options_.enable_live_camera) {
        camera_.stop();
    }
    return ok;
}

int StationHttpServer::start_background() {
    if (running_.load()) {
        return bound_port_.load();
    }
    stopping_.store(false);
    autoload_latest_verified_model();
    if (options_.enable_live_camera) {
        camera_.start();
    }

    int port = options_.port;
    if (port <= 0) {
        port = svr_.bind_to_any_port(options_.host.c_str());
        if (port <= 0) return -1;
    } else {
        if (!svr_.bind_to_port(options_.host.c_str(), port)) {
            return -1;
        }
    }
    bound_port_.store(port);
    running_.store(true);
    start_maintenance_thread();

    server_thread_ = std::thread([this]() {
        svr_.listen_after_bind();
        running_.store(false);
    });

    for (int i = 0; i < 50 && !svr_.is_running() && running_.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return port;
}

void StationHttpServer::start_maintenance_thread() {
    stop_maintenance_thread();
    maintenance_thread_ = std::thread([this]() {
        std::unique_lock<std::mutex> lk(maintenance_mutex_);
        while (!stopping_.load()) {
            if (maintenance_cv_.wait_for(lk, std::chrono::minutes(5), [this]() {
                    return stopping_.load();
                })) {
                break;
            }
            lk.unlock();
            try {
                (void)db_.enforce_evidence_quota();
            } catch (...) {
            }
            try {
                red_rabbit_.evaluate_periodic_triggers();
            } catch (...) {
            }
            lk.lock();
        }
    });
}

void StationHttpServer::stop_maintenance_thread() {
    maintenance_cv_.notify_all();
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.join();
    }
}

void StationHttpServer::stop() {
    stopping_.store(true);
    maintenance_cv_.notify_all();
    if (interlock_) {
        interlock_->stop();
    }
    if (svr_.is_running()) {
        svr_.stop();
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    stop_maintenance_thread();
    running_.store(false);
    if (options_.enable_live_camera && camera_.is_running()) {
        camera_.stop();
    }
}

}  // namespace igbt
