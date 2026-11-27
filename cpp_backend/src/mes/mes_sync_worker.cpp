#include <httplib.h>
#ifdef ERROR
#undef ERROR
#endif
#ifdef DELETE
#undef DELETE
#endif
#ifdef OPTIONAL
#undef OPTIONAL
#endif

#include "igbt/mes/mes_sync_worker.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace igbt {

namespace {

std::string trim_mes(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string to_upper_mes(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

std::string to_lower_mes(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string normalize_path_mes(const std::string& p, const std::string& fallback) {
    std::string out = trim_mes(p);
    if (out.empty()) out = fallback;
    if (!out.empty() && out.front() != '/') {
        out.insert(out.begin(), '/');
    }
    return out;
}

std::string url_encode_query_val(const std::string& s) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(kHex[(c >> 4) & 0x0F]);
            out.push_back(kHex[c & 0x0F]);
        }
    }
    return out;
}

std::string mask_secret_preview(const std::string& secret) {
    if (secret.empty()) return "";
    if (secret.size() <= 4) return "****";
    return "****" + secret.substr(secret.size() - 4);
}

void configure_http_client(httplib::Client& cli, const MesClientConfig& cfg) {
    cli.set_connection_timeout(std::chrono::milliseconds(std::clamp(cfg.connect_timeout_ms, 100, 10000)));
    cli.set_read_timeout(std::chrono::milliseconds(std::clamp(cfg.io_timeout_ms, 100, 15000)));
    cli.set_write_timeout(std::chrono::milliseconds(std::clamp(cfg.io_timeout_ms, 100, 15000)));
    cli.set_follow_location(false);
}

bool is_recall_lot_status(const std::string& raw) {
    const std::string u = to_upper_mes(trim_mes(raw));
    return u == "HOLD" || u == "SCRAP" || u == "RECALL";
}

std::string normalize_recall_status(const std::string& raw) {
    const std::string u = to_upper_mes(trim_mes(raw));
    if (u == "SCRAP") return "SCRAP";
    return "HOLD";
}

std::string json_str(const json& obj, const char* key, const std::string& fallback = "") {
    if (obj.is_object() && obj.contains(key) && obj[key].is_string()) {
        return obj[key].get<std::string>();
    }
    return fallback;
}

bool json_bool(const json& obj, const char* key, bool fallback = false) {
    if (obj.is_object() && obj.contains(key) && obj[key].is_boolean()) {
        return obj[key].get<bool>();
    }
    return fallback;
}

}  // namespace

int64_t mes_default_wall_clock_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

json mes_config_to_json(const MesClientConfig& cfg, bool mask_secrets) {
    return json{
        {"enabled", cfg.enabled},
        {"base_url", cfg.base_url},
        {"station_id", cfg.station_id},
        {"api_key_configured", !cfg.api_key.empty()},
        {"api_key_preview", mask_secrets ? mask_secret_preview(cfg.api_key) : cfg.api_key},
        {"heartbeat_path", cfg.heartbeat_path},
        {"validate_lot_path", cfg.validate_lot_path},
        {"pass_record_path", cfg.pass_record_path},
        {"connect_timeout_ms", cfg.connect_timeout_ms},
        {"io_timeout_ms", cfg.io_timeout_ms},
        {"heartbeat_period_ms", cfg.heartbeat_period_ms},
        {"heartbeat_fail_lock_ms", cfg.heartbeat_fail_lock_ms},
        {"outbox_poll_ms", cfg.outbox_poll_ms},
        {"base_backoff_ms", cfg.base_backoff_ms},
        {"max_backoff_ms", cfg.max_backoff_ms},
        {"max_retries", cfg.max_retries},
        {"whitelist_max_tubes", std::clamp(cfg.whitelist_max_tubes, 1, kMaxWhitelistTubes)},
        {"whitelist_max_duration_ms", std::clamp<int64_t>(cfg.whitelist_max_duration_ms, 60000, kMaxWhitelistDurationMs)},
    };
}

MesClientConfig mes_config_from_json(const json& j, const MesClientConfig& base) {
    MesClientConfig cfg = base;
    if (!j.is_object()) return cfg;

    auto get_bool = [&](const char* k, bool& dst) {
        if (j.contains(k) && j[k].is_boolean()) dst = j[k].get<bool>();
    };
    auto get_str = [&](const char* k, std::string& dst) {
        if (j.contains(k) && j[k].is_string()) dst = trim_mes(j[k].get<std::string>());
    };
    auto get_int = [&](const char* k, int& dst, int lo, int hi) {
        if (j.contains(k) && j[k].is_number_integer()) {
            dst = std::clamp(j[k].get<int>(), lo, hi);
        }
    };
    auto get_i64 = [&](const char* k, int64_t& dst, int64_t lo, int64_t hi) {
        if (j.contains(k) && j[k].is_number_integer()) {
            dst = std::clamp<int64_t>(j[k].get<int64_t>(), lo, hi);
        }
    };

    get_bool("enabled", cfg.enabled);
    get_str("base_url", cfg.base_url);
    get_str("station_id", cfg.station_id);
    if (cfg.station_id.empty()) cfg.station_id = "STATION_01";
    get_str("api_key", cfg.api_key);
    get_str("signing_secret", cfg.signing_secret);
    if (cfg.signing_secret.empty()) cfg.signing_secret = base.signing_secret;

    if (j.contains("heartbeat_path") && j["heartbeat_path"].is_string()) {
        cfg.heartbeat_path = normalize_path_mes(j["heartbeat_path"].get<std::string>(), "/api/mes/v1/heartbeat");
    }
    if (j.contains("validate_lot_path") && j["validate_lot_path"].is_string()) {
        cfg.validate_lot_path = normalize_path_mes(j["validate_lot_path"].get<std::string>(), "/api/mes/v1/lots/validate");
    }
    if (j.contains("pass_record_path") && j["pass_record_path"].is_string()) {
        cfg.pass_record_path = normalize_path_mes(j["pass_record_path"].get<std::string>(), "/api/mes/v1/pass_records");
    }

    get_int("connect_timeout_ms", cfg.connect_timeout_ms, 100, 10000);
    get_int("io_timeout_ms", cfg.io_timeout_ms, 100, 15000);
    get_int("heartbeat_period_ms", cfg.heartbeat_period_ms, 500, 60000);
    get_i64("heartbeat_fail_lock_ms", cfg.heartbeat_fail_lock_ms, 1000, 600000);
    get_int("outbox_poll_ms", cfg.outbox_poll_ms, 100, 30000);
    get_i64("base_backoff_ms", cfg.base_backoff_ms, 100, 30000);
    get_i64("max_backoff_ms", cfg.max_backoff_ms, 1000, 300000);
    get_int("max_retries", cfg.max_retries, 1, 100);
    get_int("whitelist_max_tubes", cfg.whitelist_max_tubes, 1, kMaxWhitelistTubes);
    get_i64("whitelist_max_duration_ms", cfg.whitelist_max_duration_ms, 60000, kMaxWhitelistDurationMs);

    return cfg;
}

MesClientConfig load_mes_config_file(const std::filesystem::path& path) {
    MesClientConfig cfg;
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, bytes) || bytes.empty()) {
        return cfg;
    }
    try {
        json parsed = json::parse(std::string(bytes.begin(), bytes.end()));
        return mes_config_from_json(parsed, cfg);
    } catch (...) {
        return cfg;
    }
}

bool save_mes_config_file(const std::filesystem::path& path, const MesClientConfig& cfg) {
    if (path.empty()) return false;
    json payload = mes_config_to_json(cfg, false);
    payload["api_key"] = cfg.api_key;
    payload["signing_secret"] = cfg.signing_secret;
    const std::string dumped = payload.dump(2) + "\n";
    return write_file_bytes_atomic(path, dumped.data(), dumped.size());
}

std::string compute_mes_idempotency_key(
    const std::string& station_id,
    const std::string& lot_id,
    const std::string& event_uuid,
    int64_t tube_seq
) {
    const std::string canonical =
        station_id + "|" + lot_id + "|" + event_uuid + "|" + std::to_string(tube_seq);
    return sha256_hex(canonical);
}

std::string compute_mes_lot_signature(
    const std::string& secret,
    const std::string& station_id,
    const std::string& work_order_id,
    const std::string& lot_id,
    const std::string& part_number,
    const std::string& expected_pipe_color,
    int64_t validated_at_ms,
    int64_t expires_at_ms
) {
    const std::string canonical =
        secret + "|" + station_id + "|" + work_order_id + "|" + lot_id + "|" +
        part_number + "|" + expected_pipe_color + "|" +
        std::to_string(validated_at_ms) + "|" + std::to_string(expires_at_ms);
    return sha256_hex(canonical);
}

MesSyncWorker::ParsedEndpoint MesSyncWorker::parse_http_endpoint(const std::string& base_url) {
    const std::string raw = trim_mes(base_url);
    if (raw.empty()) {
        throw DatabaseHttpError(422, "MES_BASE_URL_REQUIRED");
    }
    if (raw.rfind("https://", 0) == 0 || raw.rfind("HTTPS://", 0) == 0) {
        throw DatabaseHttpError(422, "HTTPS_NOT_SUPPORTED_USE_HTTP_GATEWAY");
    }
    if (raw.rfind("http://", 0) != 0 && raw.rfind("HTTP://", 0) != 0) {
        throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
    }

    const std::string rest = raw.substr(7);
    const std::size_t slash_pos = rest.find('/');
    const std::string authority = (slash_pos == std::string::npos) ? rest : rest.substr(0, slash_pos);
    std::string base_path = (slash_pos == std::string::npos) ? "" : rest.substr(slash_pos);
    while (!base_path.empty() && base_path.back() == '/') {
        base_path.pop_back();
    }

    if (authority.empty()) {
        throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
    }

    ParsedEndpoint ep;
    ep.base_path = base_path;

    if (authority.front() == '[') {
        const std::size_t rb = authority.find(']');
        if (rb == std::string::npos || rb <= 1) {
            throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
        }
        ep.host = authority.substr(1, rb - 1);
        if (rb + 1 < authority.size()) {
            if (authority[rb + 1] != ':') {
                throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
            }
            const std::string port_str = authority.substr(rb + 2);
            try {
                std::size_t idx = 0;
                int p = std::stoi(port_str, &idx);
                if (idx != port_str.size() || p < 1 || p > 65535) {
                    throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
                }
                ep.port = p;
            } catch (const DatabaseHttpError&) {
                throw;
            } catch (...) {
                throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
            }
        }
    } else {
        const std::size_t colon = authority.rfind(':');
        if (colon != std::string::npos) {
            ep.host = authority.substr(0, colon);
            const std::string port_str = authority.substr(colon + 1);
            if (port_str.empty()) {
                throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
            }
            try {
                std::size_t idx = 0;
                int p = std::stoi(port_str, &idx);
                if (idx != port_str.size() || p < 1 || p > 65535) {
                    throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
                }
                ep.port = p;
            } catch (const DatabaseHttpError&) {
                throw;
            } catch (...) {
                throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
            }
        } else {
            ep.host = authority;
            ep.port = 80;
        }
    }

    if (ep.host.empty()) {
        throw DatabaseHttpError(422, "INVALID_MES_BASE_URL");
    }
    return ep;
}

std::string MesSyncWorker::join_endpoint_path(const std::string& base_path, const std::string& rel_path) {
    const std::string rel = normalize_path_mes(rel_path, "/");
    if (base_path.empty()) return rel;
    return base_path + rel;
}

MesSyncWorker::MesSyncWorker(
    StationDatabase& db,
    MesClientConfig config,
    std::filesystem::path config_path,
    WallClockMsFn wall_clock,
    InterlockCallback interlock_cb,
    LotChangeCallback lot_change_cb
)
    : db_(db),
      config_path_(std::move(config_path)),
      cfg_(std::move(config)),
      wall_clock_(wall_clock ? std::move(wall_clock) : mes_default_wall_clock_ms),
      interlock_cb_(std::move(interlock_cb)),
      lot_change_cb_(std::move(lot_change_cb)) {
    if (cfg_.enabled && !cfg_.base_url.empty()) {
        try {
            (void)parse_http_endpoint(cfg_.base_url);
        } catch (...) {
            cfg_.enabled = false;
        }
    }
}

MesSyncWorker::~MesSyncWorker() {
    stop();
}

void MesSyncWorker::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_.load()) return;
    stop_requested_ = false;
    wake_requested_ = false;
    running_.store(true);
    worker_ = std::thread(&MesSyncWorker::worker_loop, this);
}

void MesSyncWorker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
        wake_requested_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    running_.store(false);
}

void MesSyncWorker::set_wall_clock_for_test(WallClockMsFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    wall_clock_ = fn ? std::move(fn) : mes_default_wall_clock_ms;
}

void MesSyncWorker::set_interlock_callback(InterlockCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    interlock_cb_ = std::move(cb);
}

void MesSyncWorker::set_lot_change_callback(LotChangeCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    lot_change_cb_ = std::move(cb);
}

void MesSyncWorker::notify_outbox_enqueued() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        wake_requested_ = true;
    }
    cv_.notify_all();
}

int64_t MesSyncWorker::current_wall_ms() const {
    WallClockMsFn fn;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fn = wall_clock_;
    }
    return fn ? fn() : mes_default_wall_clock_ms();
}

void MesSyncWorker::emit_interlock(InterlockSignal sig) {
    InterlockCallback cb;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cb = interlock_cb_;
    }
    if (cb) {
        try {
            cb(sig);
        } catch (...) {
        }
    }
}

json MesSyncWorker::reconfigure(const MesClientConfig& new_cfg, bool persist_file) {
    MesClientConfig normalized = new_cfg;
    if (normalized.station_id.empty()) normalized.station_id = "STATION_01";
    normalized.whitelist_max_tubes = std::clamp(normalized.whitelist_max_tubes, 1, kMaxWhitelistTubes);
    normalized.whitelist_max_duration_ms =
        std::clamp<int64_t>(normalized.whitelist_max_duration_ms, 60000, kMaxWhitelistDurationMs);

    if (normalized.enabled) {
        (void)parse_http_endpoint(normalized.base_url);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool url_changed = (cfg_.base_url != normalized.base_url) || (cfg_.enabled != normalized.enabled);
        cfg_ = normalized;
        if (!cfg_.enabled || url_changed) {
            connected_ = false;
            last_heartbeat_attempt_ms_ = 0;
            last_heartbeat_ok_ms_ = 0;
            last_rtt_ms_ = 0;
            last_heartbeat_error_.clear();
            offline_since_ms_ = 0;
            offline_locked_ = false;
            offline_lock_reason_.clear();
            offline_lock_audit_emitted_ = false;
        }
        wake_requested_ = true;
    }

    if (persist_file && !config_path_.empty()) {
        (void)save_mes_config_file(config_path_, normalized);
    }

    cv_.notify_all();
    return status_json();
}

json MesSyncWorker::reconfigure_from_json(const json& patch, bool persist_file) {
    if (!patch.is_object()) {
        throw DatabaseHttpError(422, "INVALID_MES_CONFIG_PAYLOAD");
    }
    MesClientConfig cur = config();
    MesClientConfig next = mes_config_from_json(patch, cur);
    return reconfigure(next, persist_file);
}

MesClientConfig MesSyncWorker::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cfg_;
}

void MesSyncWorker::worker_loop() {
    while (true) {
        MesClientConfig cfg_snap;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const int wait_ms = cfg_.enabled
                ? std::clamp(std::min(cfg_.heartbeat_period_ms, cfg_.outbox_poll_ms), 50, 5000)
                : 500;
            cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [this]() {
                return stop_requested_ || wake_requested_;
            });
            if (stop_requested_) {
                break;
            }
            wake_requested_ = false;
            cfg_snap = cfg_;
        }

        if (!cfg_snap.enabled) {
            continue;
        }
        try {
            (void)tick_once(false);
        } catch (...) {
        }
    }
}

void MesSyncWorker::run_heartbeat_probe(
    const MesClientConfig& cfg,
    const ParsedEndpoint& ep,
    int64_t now_ms
) {
    std::string target = join_endpoint_path(ep.base_path, cfg.heartbeat_path);
    target += "?station_id=" + url_encode_query_val(cfg.station_id);
    std::string active_lot_id;
    std::string active_wo_id;
    if (auto lot = db_.mes_get_active_lot(); lot.has_value()) {
        active_lot_id = lot->value("lot_id", "");
        active_wo_id = lot->value("work_order_id", "");
        if (!active_lot_id.empty()) {
            target += "&lot_id=" + url_encode_query_val(active_lot_id);
        }
    }

    httplib::Client cli(ep.host, ep.port);
    configure_http_client(cli, cfg);
    httplib::Headers headers{
        {"X-Station-ID", cfg.station_id},
    };
    if (!cfg.api_key.empty()) {
        headers.emplace("X-MES-API-Key", cfg.api_key);
    }

    const int64_t t0_steady = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    auto res = cli.Get(target, headers);
    const int64_t rtt_ms = std::max<int64_t>(
        0,
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count() - t0_steady
    );

    if (res && res->status >= 200 && res->status < 300) {
        bool need_recall = false;
        std::string recall_status;
        std::string recall_lot = active_lot_id;
        std::string recall_wo = active_wo_id;
        std::string recall_reason = "MES 心跳对账返回批次冻结/报废状态";

        if (!res->body.empty()) {
            try {
                json parsed = json::parse(res->body);
                if (parsed.is_object()) {
                    const std::string ls = json_str(parsed, "lot_status", json_str(parsed, "action"));
                    if (is_recall_lot_status(ls)) {
                        const std::string resp_lot = json_str(parsed, "lot_id");
                        if (resp_lot.empty() || resp_lot == active_lot_id) {
                            need_recall = true;
                            recall_status = normalize_recall_status(ls);
                            if (!resp_lot.empty()) recall_lot = resp_lot;
                            if (parsed.contains("work_order_id") && parsed["work_order_id"].is_string()) {
                                recall_wo = parsed["work_order_id"].get<std::string>();
                            }
                            if (parsed.contains("reason") && parsed["reason"].is_string() &&
                                !parsed["reason"].get<std::string>().empty()) {
                                recall_reason = parsed["reason"].get<std::string>();
                            }
                        }
                    }
                }
            } catch (...) {
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            connected_ = true;
            ever_connected_ = true;
            last_heartbeat_attempt_ms_ = now_ms;
            last_heartbeat_ok_ms_ = now_ms;
            last_rtt_ms_ = rtt_ms;
            last_heartbeat_error_.clear();
            offline_since_ms_ = 0;
            offline_locked_ = false;
            offline_lock_reason_.clear();
            offline_lock_audit_emitted_ = false;
        }
        try {
            db_.mes_reset_offline_usage(now_ms);
        } catch (...) {
        }
        if (need_recall) {
            trigger_recall_lock(recall_status, recall_lot, recall_wo, recall_reason, now_ms);
        }
        return;
    }

    std::string err_msg;
    if (!res) {
        err_msg = "HTTP_UNREACHABLE: " + httplib::to_string(res.error());
    } else {
        err_msg = "HTTP_" + std::to_string(res->status);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        connected_ = false;
        last_heartbeat_attempt_ms_ = now_ms;
        last_rtt_ms_ = rtt_ms;
        last_heartbeat_error_ = err_msg;
        if (offline_since_ms_ == 0) {
            offline_since_ms_ = now_ms;
        }
    }
}

int MesSyncWorker::drain_outbox_once(
    const MesClientConfig& cfg,
    const ParsedEndpoint& ep,
    int64_t now_ms
) {
    int acked_count = 0;
    constexpr int kMaxBatchPerTick = 20;

    for (int i = 0; i < kMaxBatchPerTick; ++i) {
        auto item = db_.mes_claim_next_pending(now_ms, 30000);
        if (!item.has_value()) {
            break;
        }

        const int64_t row_id = (*item)["id"].get<int64_t>();
        const std::string idem_key = (*item).value("idempotency_key", "");
        const std::string lot_id = (*item).value("lot_id", "");
        const std::string wo_id = (*item).value("work_order_id", "");
        const std::string rel_path = (*item).value("endpoint_path", cfg.pass_record_path);
        const std::string body = (*item).value("payload_json", "{}");
        const int retry_count = (*item).value("retry_count", 0);

        httplib::Client cli(ep.host, ep.port);
        configure_http_client(cli, cfg);
        httplib::Headers headers{
            {"X-Idempotency-Key", idem_key},
            {"X-Station-ID", cfg.station_id},
        };
        if (!cfg.api_key.empty()) {
            headers.emplace("X-MES-API-Key", cfg.api_key);
        }

        const std::string full_path = join_endpoint_path(ep.base_path, rel_path);
        auto res = cli.Post(full_path, headers, body, "application/json");
        if (!res) {
            const std::string err_msg = "TRANSPORT_ERR: " + httplib::to_string(res.error());
            const int shift = std::min(retry_count, 10);
            const int64_t delay_ms = std::min<int64_t>(
                cfg.max_backoff_ms,
                std::max<int64_t>(100, cfg.base_backoff_ms) * (int64_t{1} << shift)
            );
            const bool dead_letter = (retry_count + 1) >= std::max(1, cfg.max_retries);
            db_.mes_mark_retry(row_id, 0, err_msg, now_ms + delay_ms, dead_letter);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                connected_ = false;
                last_heartbeat_error_ = err_msg;
                if (offline_since_ms_ == 0) {
                    offline_since_ms_ = now_ms;
                }
            }
            break;
        }

        json resp_json = json::object();
        if (!res->body.empty()) {
            try {
                json parsed = json::parse(res->body);
                if (parsed.is_object()) resp_json = std::move(parsed);
            } catch (...) {
            }
        }
        const std::string resp_lot_status = json_str(resp_json, "lot_status", json_str(resp_json, "action"));
        const bool is_recall = is_recall_lot_status(resp_lot_status);

        if (res->status >= 200 && res->status < 300) {
            db_.mes_mark_acked(row_id, res->status, res->body, now_ms);
            ++acked_count;
            if (is_recall) {
                trigger_recall_lock(
                    normalize_recall_status(resp_lot_status),
                    json_str(resp_json, "lot_id", lot_id),
                    json_str(resp_json, "work_order_id", wo_id),
                    json_str(resp_json, "reason", "MES 补传对账返回批次冻结/报废状态"),
                    now_ms
                );
            }
            continue;
        }

        const bool idempotent_dup =
            (res->status == 409) &&
            (json_bool(resp_json, "duplicate") || json_bool(resp_json, "already_acked"));
        if (idempotent_dup) {
            db_.mes_mark_acked(row_id, res->status, res->body, now_ms);
            ++acked_count;
            if (is_recall) {
                trigger_recall_lock(
                    normalize_recall_status(resp_lot_status),
                    json_str(resp_json, "lot_id", lot_id),
                    json_str(resp_json, "work_order_id", wo_id),
                    json_str(resp_json, "reason", "MES 幂等对账返回批次冻结/报废状态"),
                    now_ms
                );
            }
            continue;
        }

        if ((res->status == 409 || res->status == 410 || res->status == 423) && is_recall) {
            db_.mes_mark_retry(
                row_id,
                res->status,
                json_str(resp_json, "reason", "MES_LOT_RECALLED"),
                now_ms,
                true
            );
            trigger_recall_lock(
                normalize_recall_status(resp_lot_status),
                json_str(resp_json, "lot_id", lot_id),
                json_str(resp_json, "work_order_id", wo_id),
                json_str(resp_json, "reason", "MES 补传对账拒绝：批次已冻结或报废"),
                now_ms
            );
            continue;
        }

        if (res->status == 400 || res->status == 422) {
            db_.mes_mark_retry(
                row_id,
                res->status,
                json_str(resp_json, "detail", "HTTP_" + std::to_string(res->status)),
                now_ms,
                true
            );
            continue;
        }

        const int shift = std::min(retry_count, 10);
        const int64_t delay_ms = std::min<int64_t>(
            cfg.max_backoff_ms,
            std::max<int64_t>(100, cfg.base_backoff_ms) * (int64_t{1} << shift)
        );
        const bool dead_letter = (retry_count + 1) >= std::max(1, cfg.max_retries);
        db_.mes_mark_retry(
            row_id,
            res->status,
            "HTTP_" + std::to_string(res->status),
            now_ms + delay_ms,
            dead_letter
        );
        if (res->status >= 500) {
            break;
        }
    }

    return acked_count;
}

void MesSyncWorker::trigger_recall_lock(
    const std::string& status,
    const std::string& lot_id,
    const std::string& work_order_id,
    const std::string& reason,
    int64_t now_ms
) {
    const std::string norm_status = normalize_recall_status(status);
    json cur_policy = db_.mes_get_runtime_policy();
    const bool already_locked_same =
        cur_policy.value("recall_locked", false) &&
        cur_policy.value("recall_lot_id", "") == lot_id &&
        cur_policy.value("recall_status", "") == norm_status;

    const std::vector<std::string> keys = db_.mes_list_lot_outbox_keys(lot_id, 200);
    json keys_json = json::array();
    for (const auto& k : keys) {
        keys_json.push_back(k);
    }

    cur_policy["recall_locked"] = true;
    cur_policy["recall_status"] = norm_status;
    cur_policy["recall_lot_id"] = lot_id;
    cur_policy["recall_work_order_id"] = work_order_id;
    cur_policy["recall_reason"] = reason;
    cur_policy["recall_affected_keys"] = std::move(keys_json);
    if (!already_locked_same || cur_policy.value("recall_triggered_at_ms", int64_t{0}) <= 0) {
        cur_policy["recall_triggered_at_ms"] = now_ms;
    }
    cur_policy["updated_at_ms"] = now_ms;
    (void)db_.mes_save_runtime_policy(cur_policy);

    try {
        if (auto lot = db_.mes_get_active_lot(); lot.has_value() && lot->value("lot_id", "") == lot_id) {
            db_.mes_update_active_lot_status(norm_status, now_ms);
        }
    } catch (...) {
    }

    if (!already_locked_same) {
        const std::string reason_code =
            (norm_status == "SCRAP") ? "MES_LOT_SCRAP_RECALL" : "MES_LOT_HOLD_RECALL";
        try {
            db_.rr_append_lock_audit(
                "MES_OFFLINE_LOCK",
                "SYSTEM",
                "SYSTEM",
                reason_code,
                reason + " [Lot=" + lot_id + ", 受影响记录=" + std::to_string(keys.size()) + "]",
                std::nullopt,
                now_ms
            );
        } catch (...) {
        }
        emit_interlock(InterlockSignal::ALARM_LATCHED);
    }
}

MesSyncWorker::EvaluatedPolicyState MesSyncWorker::evaluate_policy_snapshot(
    int64_t now_ms,
    bool emit_lock_effects
) {
    MesClientConfig cfg;
    bool connected = false;
    int64_t offline_since = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg = cfg_;
        connected = connected_;
        offline_since = offline_since_ms_;
    }

    EvaluatedPolicyState out;
    out.enabled = cfg.enabled;
    out.connected = connected;
    out.policy = db_.mes_get_runtime_policy();
    out.active_lot = db_.mes_get_active_lot();

    const bool wl_approved = out.policy.value("whitelist_approved", false);
    out.whitelist_max_tubes = std::clamp(
        out.policy.value("whitelist_max_tubes", cfg.whitelist_max_tubes),
        1,
        kMaxWhitelistTubes
    );
    out.whitelist_max_duration_ms = std::clamp<int64_t>(
        out.policy.value("whitelist_max_duration_ms", cfg.whitelist_max_duration_ms),
        60000,
        kMaxWhitelistDurationMs
    );

    if (!cfg.enabled) {
        out.degradation_mode = "DISABLED";
        out.blocking = false;
        return out;
    }

    if (out.policy.value("recall_locked", false)) {
        out.recall_locked = true;
        out.blocking = true;
        const std::string rs = out.policy.value("recall_status", "HOLD");
        out.block_reason = (rs == "SCRAP") ? "MES_LOT_SCRAP_RECALL" : "MES_LOT_HOLD_RECALL";
        out.degradation_mode = "RECALL_LOCKED";
    }

    if (!out.active_lot.has_value() || out.active_lot->value("lot_id", "").empty()) {
        out.lot_bound = false;
        if (!out.blocking) {
            out.blocking = true;
            out.block_reason = "MES_LOT_UNBOUND_LOCK";
            out.degradation_mode = "LOT_UNBOUND";
        }
    } else {
        out.lot_bound = true;
        const auto& lot = *out.active_lot;
        const std::string expected_sig = compute_mes_lot_signature(
            cfg.signing_secret,
            cfg.station_id,
            lot.value("work_order_id", ""),
            lot.value("lot_id", ""),
            lot.value("part_number", ""),
            lot.value("expected_pipe_color", ""),
            lot.value("validated_at_ms", int64_t{0}),
            lot.value("expires_at_ms", int64_t{0})
        );
        const std::string actual_sig = lot.value("signature_sha256", "");
        out.signature_valid = !actual_sig.empty() && constant_time_equals(actual_sig, expected_sig);

        const int64_t exp_ms = lot.value("expires_at_ms", int64_t{0});
        out.lot_expired = (exp_ms > 0 && now_ms > exp_ms);

        const std::string lot_status = lot.value("lot_status", "ACTIVE");
        if ((lot_status == "HOLD" || lot_status == "SCRAP") && !out.blocking) {
            out.recall_locked = true;
            out.blocking = true;
            out.block_reason = (lot_status == "SCRAP") ? "MES_LOT_SCRAP_RECALL" : "MES_LOT_HOLD_RECALL";
            out.degradation_mode = "RECALL_LOCKED";
        }
    }

    if (!connected) {
        int64_t persisted_offline_anchor = 0;
        if (out.active_lot.has_value() &&
            out.active_lot->contains("offline_started_at_ms") &&
            (*out.active_lot)["offline_started_at_ms"].is_number_integer()) {
            persisted_offline_anchor = (*out.active_lot)["offline_started_at_ms"].get<int64_t>();
        }
        int64_t anchor = 0;
        if (offline_since > 0 && persisted_offline_anchor > 0) {
            anchor = std::min(offline_since, persisted_offline_anchor);
        } else if (offline_since > 0) {
            anchor = offline_since;
        } else if (persisted_offline_anchor > 0) {
            anchor = persisted_offline_anchor;
        }
        out.offline_elapsed_ms = (anchor > 0) ? std::max<int64_t>(0, now_ms - anchor) : 0;

        if (!wl_approved) {
            if (anchor > 0 && out.offline_elapsed_ms >= cfg.heartbeat_fail_lock_ms) {
                out.offline_locked = true;
                if (!out.blocking) {
                    out.blocking = true;
                    out.block_reason = "MES_HEARTBEAT_TIMEOUT";
                    out.degradation_mode = "FAIL_CLOSED_LOCKED";
                }
            } else if (!out.blocking) {
                out.degradation_mode = "OFFLINE_GRACE";
            }
        } else {
            const int used = out.active_lot.has_value() ? out.active_lot->value("offline_used_count", 0) : 0;
            out.whitelist_remaining_tubes = std::max(0, out.whitelist_max_tubes - used);
            out.whitelist_remaining_ms = (anchor > 0)
                ? std::max<int64_t>(0, out.whitelist_max_duration_ms - out.offline_elapsed_ms)
                : out.whitelist_max_duration_ms;

            if (out.lot_bound && !out.signature_valid) {
                out.offline_locked = true;
                if (!out.blocking) {
                    out.blocking = true;
                    out.block_reason = "MES_LOT_SIGNATURE_INVALID";
                    out.degradation_mode = "WHITELIST_LOCKED";
                }
            } else if (out.lot_bound && out.lot_expired) {
                out.offline_locked = true;
                if (!out.blocking) {
                    out.blocking = true;
                    out.block_reason = "MES_LOT_CACHE_EXPIRED";
                    out.degradation_mode = "WHITELIST_LOCKED";
                }
            } else if (used >= out.whitelist_max_tubes) {
                out.offline_locked = true;
                if (!out.blocking) {
                    out.blocking = true;
                    out.block_reason = "MES_WHITELIST_QUOTA_EXCEEDED";
                    out.degradation_mode = "WHITELIST_LOCKED";
                }
            } else if (anchor > 0 && out.offline_elapsed_ms >= out.whitelist_max_duration_ms) {
                out.offline_locked = true;
                if (!out.blocking) {
                    out.blocking = true;
                    out.block_reason = "MES_WHITELIST_DURATION_EXCEEDED";
                    out.degradation_mode = "WHITELIST_LOCKED";
                }
            } else if (!out.blocking) {
                out.degradation_mode = "WHITELIST_ACTIVE";
            }
        }
    } else {
        const int used = out.active_lot.has_value() ? out.active_lot->value("offline_used_count", 0) : 0;
        out.whitelist_remaining_tubes = std::max(0, out.whitelist_max_tubes - used);
        out.whitelist_remaining_ms = out.whitelist_max_duration_ms;
        if (!out.blocking) {
            out.degradation_mode = wl_approved ? "ONLINE_WHITELIST_ARMED" : "ONLINE";
        }
    }

    if (emit_lock_effects) {
        bool need_emit_audit = false;
        std::string audit_reason;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            offline_locked_ = out.offline_locked;
            offline_lock_reason_ = out.offline_locked ? out.block_reason : "";
            if (out.offline_locked) {
                if (!offline_lock_audit_emitted_) {
                    offline_lock_audit_emitted_ = true;
                    need_emit_audit = true;
                    audit_reason = out.block_reason.empty() ? "MES_HEARTBEAT_TIMEOUT" : out.block_reason;
                }
            } else {
                offline_lock_audit_emitted_ = false;
            }
        }
        if (need_emit_audit) {
            try {
                db_.rr_append_lock_audit(
                    "MES_OFFLINE_LOCK",
                    "SYSTEM",
                    "SYSTEM",
                    audit_reason,
                    "MES 心跳断联或离线白名单配额/时长越界触发工位落锁",
                    std::nullopt,
                    now_ms
                );
            } catch (...) {
            }
            emit_interlock(InterlockSignal::LOCKED);
        }
    }

    return out;
}

json MesSyncWorker::tick_once(bool force_heartbeat) {
    MesClientConfig cfg;
    int64_t last_hb_attempt = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg = cfg_;
        last_hb_attempt = last_heartbeat_attempt_ms_;
    }

    const int64_t now_ms = current_wall_ms();
    if (!cfg.enabled) {
        return status_json();
    }

    const ParsedEndpoint ep = parse_http_endpoint(cfg.base_url);
    if (force_heartbeat || last_hb_attempt == 0 ||
        (now_ms - last_hb_attempt) >= std::max(500, cfg.heartbeat_period_ms)) {
        run_heartbeat_probe(cfg, ep, now_ms);
    }

    (void)drain_outbox_once(cfg, ep, now_ms);
    (void)evaluate_policy_snapshot(now_ms, true);
    return status_json();
}

void MesSyncWorker::evaluate_offline_policy_now() {
    const int64_t now_ms = current_wall_ms();
    (void)evaluate_policy_snapshot(now_ms, true);
}

json MesSyncWorker::bind_work_order_online(const json& payload, const std::string& default_actor) {
    if (!payload.is_object()) {
        throw DatabaseHttpError(422, "INVALID_LOT_BIND_PAYLOAD");
    }

    MesClientConfig cfg;
    bool was_connected = false;
    int64_t offline_since = 0;
    LotChangeCallback lot_cb;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg = cfg_;
        was_connected = connected_;
        offline_since = offline_since_ms_;
        lot_cb = lot_change_cb_;
    }

    if (!cfg.enabled) {
        throw DatabaseHttpError(409, "MES_DISABLED");
    }

    const std::string work_order_id = trim_mes(json_str(payload, "work_order_id"));
    const std::string lot_id = trim_mes(json_str(payload, "lot_id"));
    std::string part_number = trim_mes(json_str(payload, "part_number"));
    std::string expected_pipe_color = to_lower_mes(trim_mes(json_str(payload, "expected_pipe_color")));
    const std::string actor_id = trim_mes(json_str(payload, "actor_id", default_actor)).empty()
                                     ? "operator"
                                     : trim_mes(json_str(payload, "actor_id", default_actor));

    if (work_order_id.empty() || lot_id.empty()) {
        throw DatabaseHttpError(422, "WORK_ORDER_AND_LOT_REQUIRED");
    }
    if (!expected_pipe_color.empty() &&
        expected_pipe_color != "yellow" &&
        expected_pipe_color != "blue" &&
        expected_pipe_color != "other") {
        throw DatabaseHttpError(422, "INVALID_EXPECTED_PIPE_COLOR");
    }

    if (!was_connected && offline_since > 0) {
        throw DatabaseHttpError(409, "MES_OFFLINE_LOT_CHANGE_FORBIDDEN");
    }

    const int64_t now_ms = current_wall_ms();
    const ParsedEndpoint ep = parse_http_endpoint(cfg.base_url);
    const std::string full_path = join_endpoint_path(ep.base_path, cfg.validate_lot_path);

    json req_body{
        {"station_id", cfg.station_id},
        {"work_order_id", work_order_id},
        {"lot_id", lot_id},
        {"part_number", part_number},
        {"expected_pipe_color", expected_pipe_color},
        {"operator_id", actor_id},
    };

    httplib::Client cli(ep.host, ep.port);
    configure_http_client(cli, cfg);
    httplib::Headers headers{
        {"X-Station-ID", cfg.station_id},
    };
    if (!cfg.api_key.empty()) {
        headers.emplace("X-MES-API-Key", cfg.api_key);
    }

    auto res = cli.Post(full_path, headers, req_body.dump(), "application/json");
    if (!res || res->status >= 500) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            connected_ = false;
            last_heartbeat_attempt_ms_ = now_ms;
            last_heartbeat_error_ = !res
                ? ("HTTP_UNREACHABLE: " + httplib::to_string(res.error()))
                : ("HTTP_" + std::to_string(res->status));
            if (offline_since_ms_ == 0) {
                offline_since_ms_ = now_ms;
            }
        }
        (void)evaluate_policy_snapshot(now_ms, true);
        throw DatabaseHttpError(409, "MES_OFFLINE_LOT_CHANGE_FORBIDDEN");
    }

    json resp = json::object();
    if (!res->body.empty()) {
        try {
            json parsed = json::parse(res->body);
            if (parsed.is_object()) resp = std::move(parsed);
        } catch (...) {
        }
    }

    if (res->status < 200 || res->status >= 300) {
        const std::string ls = json_str(resp, "lot_status", json_str(resp, "action"));
        if (is_recall_lot_status(ls)) {
            const std::string norm = normalize_recall_status(ls);
            trigger_recall_lock(
                norm,
                lot_id,
                work_order_id,
                json_str(resp, "reason", "MES 批次核验返回冻结/报废状态"),
                now_ms
            );
            throw DatabaseHttpError(409, norm == "SCRAP" ? "MES_LOT_SCRAP_RECALL" : "MES_LOT_HOLD_RECALL");
        }
        throw DatabaseHttpError(
            res->status == 422 ? 422 : 409,
            json_str(resp, "detail", json_str(resp, "reason", "MES_LOT_VALIDATION_REJECTED"))
        );
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        connected_ = true;
        ever_connected_ = true;
        last_heartbeat_attempt_ms_ = now_ms;
        last_heartbeat_ok_ms_ = now_ms;
        last_heartbeat_error_.clear();
        offline_since_ms_ = 0;
        offline_locked_ = false;
        offline_lock_reason_.clear();
        offline_lock_audit_emitted_ = false;
    }

    const std::string lot_status = to_upper_mes(json_str(resp, "lot_status", "ACTIVE"));
    const bool allowed = json_bool(resp, "allowed", true) && json_bool(resp, "ok", true);
    if (!allowed || is_recall_lot_status(lot_status)) {
        const std::string norm = is_recall_lot_status(lot_status) ? normalize_recall_status(lot_status) : "HOLD";
        trigger_recall_lock(
            norm,
            lot_id,
            work_order_id,
            json_str(resp, "reason", "MES 拒绝放行该批次"),
            now_ms
        );
        throw DatabaseHttpError(409, norm == "SCRAP" ? "MES_LOT_SCRAP_RECALL" : "MES_LOT_HOLD_RECALL");
    }

    if (resp.contains("part_number") && resp["part_number"].is_string() &&
        !resp["part_number"].get<std::string>().empty()) {
        part_number = trim_mes(resp["part_number"].get<std::string>());
    }
    if (part_number.empty()) part_number = "IGBT-TO247";

    if (resp.contains("expected_pipe_color") && resp["expected_pipe_color"].is_string() &&
        !resp["expected_pipe_color"].get<std::string>().empty()) {
        expected_pipe_color = to_lower_mes(trim_mes(resp["expected_pipe_color"].get<std::string>()));
    }
    if (expected_pipe_color.empty()) expected_pipe_color = "yellow";

    int64_t expires_at_ms = now_ms + 24LL * 3600 * 1000;
    if (resp.contains("expires_at_ms") && resp["expires_at_ms"].is_number_integer()) {
        expires_at_ms = resp["expires_at_ms"].get<int64_t>();
    }

    const std::optional<json> prev_lot = db_.mes_get_active_lot();
    const std::string prev_lot_id = prev_lot.has_value() ? prev_lot->value("lot_id", "") : "";
    const bool lot_changed = (prev_lot_id != lot_id);
    const int64_t preserved_seq = (!lot_changed && prev_lot.has_value())
                                      ? prev_lot->value("tube_seq", int64_t{0})
                                      : (db_.mes_peek_next_tube_seq() - 1);

    const std::string sig = compute_mes_lot_signature(
        cfg.signing_secret,
        cfg.station_id,
        work_order_id,
        lot_id,
        part_number,
        expected_pipe_color,
        now_ms,
        expires_at_ms
    );

    json lot_row{
        {"work_order_id", work_order_id},
        {"lot_id", lot_id},
        {"part_number", part_number},
        {"expected_pipe_color", expected_pipe_color},
        {"lot_status", "ACTIVE"},
        {"signature_sha256", sig},
        {"validated_at_ms", now_ms},
        {"expires_at_ms", expires_at_ms},
        {"tube_seq", std::max<int64_t>(0, preserved_seq)},
        {"offline_used_count", 0},
        {"offline_started_at_ms", nullptr},
        {"updated_at_ms", now_ms},
    };
    (void)db_.mes_save_active_lot(lot_row);

    if (lot_changed && lot_cb) {
        try {
            lot_cb(lot_id, actor_id);
        } catch (...) {
        }
    }

    json out = status_json();
    out["lot_changed"] = lot_changed;
    out["red_rabbit_triggered"] = lot_changed;
    return out;
}

json MesSyncWorker::set_whitelist_policy(const json& payload, const std::string& default_actor) {
    if (!payload.is_object()) {
        throw DatabaseHttpError(422, "INVALID_WHITELIST_PAYLOAD");
    }

    const bool approved = json_bool(payload, "approved", false);
    const std::string actor_id = trim_mes(json_str(payload, "actor_id", default_actor)).empty()
                                     ? "qe"
                                     : trim_mes(json_str(payload, "actor_id", default_actor));
    const std::string actor_role = to_upper_mes(trim_mes(json_str(payload, "actor_role", "QE")));
    if (actor_role != "QE" && actor_role != "ENGINEER") {
        throw DatabaseHttpError(422, "INVALID_WHITELIST_ACTOR_ROLE");
    }
    const std::string reason = trim_mes(json_str(payload, "reason"));
    if (approved && reason.size() < 2) {
        throw DatabaseHttpError(422, "WHITELIST_REASON_REQUIRED");
    }

    const int requested_tubes = (payload.contains("max_tubes") && payload["max_tubes"].is_number_integer())
                                    ? payload["max_tubes"].get<int>()
                                    : kMaxWhitelistTubes;
    const int clamped_tubes = std::clamp(requested_tubes, 1, kMaxWhitelistTubes);

    int64_t requested_dur_ms = kMaxWhitelistDurationMs;
    if (payload.contains("max_duration_minutes") && payload["max_duration_minutes"].is_number()) {
        requested_dur_ms = static_cast<int64_t>(payload["max_duration_minutes"].get<double>() * 60000.0);
    } else if (payload.contains("max_duration_ms") && payload["max_duration_ms"].is_number_integer()) {
        requested_dur_ms = payload["max_duration_ms"].get<int64_t>();
    }
    const int64_t clamped_dur_ms = std::clamp<int64_t>(requested_dur_ms, 60000, kMaxWhitelistDurationMs);

    const int64_t now_ms = current_wall_ms();
    json policy = db_.mes_get_runtime_policy();
    policy["whitelist_approved"] = approved;
    policy["whitelist_actor_id"] = actor_id;
    policy["whitelist_actor_role"] = actor_role;
    policy["whitelist_reason"] = reason.empty() ? (approved ? "QE受控离线白名单启用" : "白名单已撤回") : reason;
    policy["whitelist_max_tubes"] = clamped_tubes;
    policy["whitelist_max_duration_ms"] = clamped_dur_ms;
    policy["whitelist_approved_at_ms"] = now_ms;
    policy["updated_at_ms"] = now_ms;
    (void)db_.mes_save_runtime_policy(policy);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        offline_lock_audit_emitted_ = false;
    }

    const std::string audit_comment =
        (approved ? "启用受控离线白名单" : "撤回受控离线白名单") +
        std::string(" [max_tubes=") + std::to_string(clamped_tubes) +
        ", max_min=" + std::to_string(clamped_dur_ms / 60000) +
        "] " + policy["whitelist_reason"].get<std::string>();
    (void)db_.rr_append_lock_audit(
        "POLICY_OVERRIDE",
        actor_id,
        actor_role,
        approved ? "MES_WHITELIST_ENABLED" : "MES_WHITELIST_REVOKED",
        audit_comment,
        std::nullopt,
        now_ms
    );

    (void)evaluate_policy_snapshot(now_ms, true);
    return status_json();
}

json MesSyncWorker::unlock_recall_lock(const json& payload, const std::string& default_actor) {
    if (!payload.is_object()) {
        throw DatabaseHttpError(422, "INVALID_RECALL_UNLOCK_PAYLOAD");
    }

    const std::string actor_id = trim_mes(json_str(payload, "actor_id", default_actor)).empty()
                                     ? "qe"
                                     : trim_mes(json_str(payload, "actor_id", default_actor));
    const std::string actor_role = to_upper_mes(trim_mes(json_str(payload, "actor_role", "QE")));
    if (actor_role != "QE" && actor_role != "ENGINEER") {
        throw DatabaseHttpError(422, "INVALID_UNLOCK_ACTOR_ROLE");
    }
    const std::string reason_code = trim_mes(json_str(payload, "reason_code", "MES_RECALL_QUARANTINED"));
    const std::string comment = trim_mes(json_str(payload, "comment"));
    if (reason_code.empty() || comment.size() < 4) {
        throw DatabaseHttpError(422, "UNLOCK_REASON_AND_COMMENT_REQUIRED");
    }

    const int64_t now_ms = current_wall_ms();
    json policy = db_.mes_get_runtime_policy();
    const std::string recalled_lot = policy.value("recall_lot_id", "");
    const std::string recall_status = policy.value("recall_status", "");

    policy["recall_locked"] = false;
    policy["recall_status"] = "";
    policy["recall_reason"] = "";
    policy["recall_affected_keys"] = json::array();
    policy["updated_at_ms"] = now_ms;
    (void)db_.mes_save_runtime_policy(policy);

    try {
        if (auto lot = db_.mes_get_active_lot(); lot.has_value() &&
            (!recalled_lot.empty() && lot->value("lot_id", "") == recalled_lot)) {
            db_.mes_update_active_lot_status("ACTIVE", now_ms);
        }
    } catch (...) {
    }

    (void)db_.rr_append_lock_audit(
        "ENGINEER_UNLOCK",
        actor_id,
        actor_role,
        reason_code,
        comment + " [Lot=" + recalled_lot + ", 原状态=" + recall_status + "]",
        std::nullopt,
        now_ms
    );

    const EvaluatedPolicyState ev = evaluate_policy_snapshot(now_ms, false);
    if (!ev.blocking) {
        emit_interlock(InterlockSignal::LOCKED);
    }
    return status_json();
}

std::optional<MesOutboxEnqueueRequest> MesSyncWorker::prepare_cycle_outbox_request(
    const json& snap,
    const std::string& model_sha256
) {
    MesClientConfig cfg;
    bool connected = false;
    int64_t offline_since = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg = cfg_;
        connected = connected_;
        offline_since = offline_since_ms_;
    }

    if (!cfg.enabled) {
        return std::nullopt;
    }
    const std::string outcome = json_str(snap, "outcome");
    if (outcome != "PASS" && outcome != "ALARM") {
        return std::nullopt;
    }

    const auto lot_opt = db_.mes_get_active_lot();
    if (!lot_opt.has_value()) {
        return std::nullopt;
    }
    const std::string lot_id = json_str(*lot_opt, "lot_id");
    const std::string work_order_id = json_str(*lot_opt, "work_order_id");
    if (lot_id.empty()) {
        return std::nullopt;
    }

    const int64_t now_ms = current_wall_ms();
    const int64_t tube_seq = db_.mes_peek_next_tube_seq();
    const std::string cycle_id = json_str(snap, "cycle_id");
    std::string event_uuid = json_str(snap, "transition_event_id");
    if (event_uuid.empty()) {
        event_uuid = !cycle_id.empty() ? cycle_id : random_token_hex(8);
    }
    const std::string idem_key = compute_mes_idempotency_key(cfg.station_id, lot_id, event_uuid, tube_seq);

    MesOutboxEnqueueRequest req;
    req.idempotency_key = idem_key;
    req.cycle_id = cycle_id;
    req.work_order_id = work_order_id;
    req.lot_id = lot_id;
    req.tube_seq = tube_seq;
    req.endpoint_path = cfg.pass_record_path;
    req.created_at_ms = now_ms;
    req.increment_offline_used = !connected;
    if (!connected) {
        req.offline_started_at_ms = (offline_since > 0) ? offline_since : now_ms;
    }

    req.payload = json{
        {"event_id", idem_key},
        {"idempotency_key", idem_key},
        {"station_id", cfg.station_id},
        {"cycle_id", cycle_id},
        {"work_order_id", work_order_id},
        {"lot_id", lot_id},
        {"part_number", json_str(*lot_opt, "part_number")},
        {"expected_pipe_color", json_str(*lot_opt, "expected_pipe_color")},
        {"observed_pipe_color", json_str(snap, "pipe_color", "unknown")},
        {"tube_seq", tube_seq},
        {"outcome", outcome},
        {"result_reason", snap.contains("result_reason") ? snap["result_reason"] : json(nullptr)},
        {"label_status", snap.contains("label_status") ? snap["label_status"] : json(nullptr)},
        {"model_version", json_str(snap, "model_version", "heuristic")},
        {"model_sha256", model_sha256},
        {"offline_mode", !connected},
        {"recorded_at_ms", now_ms},
    };
    return req;
}

bool MesSyncWorker::check_pipe_color_mismatch(const std::string& observed_pipe_color) const {
    if (!is_enabled()) {
        return false;
    }
    const auto lot_opt = db_.mes_get_active_lot();
    if (!lot_opt.has_value()) {
        return false;
    }
    const std::string expected = to_lower_mes(trim_mes(lot_opt->value("expected_pipe_color", "")));
    const std::string observed = to_lower_mes(trim_mes(observed_pipe_color));
    if ((expected == "yellow" || expected == "blue") &&
        (observed == "yellow" || observed == "blue") &&
        expected != observed) {
        return true;
    }
    return false;
}

bool MesSyncWorker::is_enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cfg_.enabled;
}

bool MesSyncWorker::is_blocking() const {
    if (!is_enabled()) return false;
    return const_cast<MesSyncWorker*>(this)->evaluate_policy_snapshot(current_wall_ms(), false).blocking;
}

bool MesSyncWorker::is_recall_locked() const {
    if (!is_enabled()) return false;
    return const_cast<MesSyncWorker*>(this)->evaluate_policy_snapshot(current_wall_ms(), false).recall_locked;
}

bool MesSyncWorker::is_offline_locked() const {
    if (!is_enabled()) return false;
    return const_cast<MesSyncWorker*>(this)->evaluate_policy_snapshot(current_wall_ms(), false).offline_locked;
}

std::string MesSyncWorker::block_reason() const {
    if (!is_enabled()) return "";
    return const_cast<MesSyncWorker*>(this)->evaluate_policy_snapshot(current_wall_ms(), false).block_reason;
}

json MesSyncWorker::status_json() const {
    MesClientConfig cfg;
    bool connected = false;
    bool ever_connected = false;
    int64_t last_hb_attempt = 0;
    int64_t last_hb_ok = 0;
    int64_t last_rtt = 0;
    std::string last_err;
    int64_t offline_since = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg = cfg_;
        connected = connected_;
        ever_connected = ever_connected_;
        last_hb_attempt = last_heartbeat_attempt_ms_;
        last_hb_ok = last_heartbeat_ok_ms_;
        last_rtt = last_rtt_ms_;
        last_err = last_heartbeat_error_;
        offline_since = offline_since_ms_;
    }

    const int64_t now_ms = current_wall_ms();
    const EvaluatedPolicyState ev =
        const_cast<MesSyncWorker*>(this)->evaluate_policy_snapshot(now_ms, false);
    const json counts = db_.mes_outbox_counts();

    json active_lot_out = nullptr;
    if (ev.active_lot.has_value()) {
        active_lot_out = *ev.active_lot;
        active_lot_out["signature_valid"] = ev.signature_valid;
        active_lot_out["expired"] = ev.lot_expired;
    }

    return json{
        {"backend", backend_name()},
        {"enabled", cfg.enabled},
        {"connected", connected},
        {"ever_connected", ever_connected},
        {"blocking", ev.blocking},
        {"block_reason", ev.block_reason},
        {"degradation_mode", ev.degradation_mode},
        {"lot_bound", ev.lot_bound},
        {"signature_valid", ev.signature_valid},
        {"lot_expired", ev.lot_expired},
        {"recall_locked", ev.recall_locked},
        {"offline_locked", ev.offline_locked},
        {"offline_since_ms", offline_since},
        {"offline_elapsed_ms", ev.offline_elapsed_ms},
        {"last_heartbeat_attempt_ms", last_hb_attempt},
        {"last_heartbeat_ok_ms", last_hb_ok},
        {"last_rtt_ms", last_rtt},
        {"last_heartbeat_error", last_err},
        {"config", mes_config_to_json(cfg, true)},
        {"active_lot", active_lot_out},
        {"whitelist", {
            {"approved", ev.policy.value("whitelist_approved", false)},
            {"actor_id", ev.policy.value("whitelist_actor_id", "")},
            {"actor_role", ev.policy.value("whitelist_actor_role", "")},
            {"reason", ev.policy.value("whitelist_reason", "")},
            {"max_tubes", ev.whitelist_max_tubes},
            {"max_duration_ms", ev.whitelist_max_duration_ms},
            {"used_tubes", ev.active_lot.has_value() ? ev.active_lot->value("offline_used_count", 0) : 0},
            {"remaining_tubes", ev.whitelist_remaining_tubes},
            {"remaining_ms", ev.whitelist_remaining_ms},
            {"approved_at_ms", ev.policy.value("whitelist_approved_at_ms", int64_t{0})},
        }},
        {"recall", {
            {"locked", ev.recall_locked},
            {"status", ev.policy.value("recall_status", "")},
            {"lot_id", ev.policy.value("recall_lot_id", "")},
            {"work_order_id", ev.policy.value("recall_work_order_id", "")},
            {"reason", ev.policy.value("recall_reason", "")},
            {"affected_keys", ev.policy.value("recall_affected_keys", json::array())},
            {"triggered_at_ms", ev.policy.value("recall_triggered_at_ms", int64_t{0})},
        }},
        {"outbox_counts", counts},
    };
}

std::optional<MesWorkOrder> MesSyncWorker::active_work_order() const {
    const auto lot_opt = db_.mes_get_active_lot();
    if (!lot_opt.has_value()) {
        return std::nullopt;
    }
    MesWorkOrder wo;
    wo.work_order_id = lot_opt->value("work_order_id", "");
    wo.part_number = lot_opt->value("part_number", "");
    wo.expected_lot_id = lot_opt->value("lot_id", "");
    wo.expected_pipe_color = lot_opt->value("expected_pipe_color", "");
    wo.lot_status = lot_opt->value("lot_status", "ACTIVE");
    wo.server_signature = lot_opt->value("signature_sha256", "");
    wo.validated_at_ms = lot_opt->value("validated_at_ms", int64_t{0});
    wo.expires_at_ms = lot_opt->value("expires_at_ms", int64_t{0});
    return wo;
}

bool MesSyncWorker::verify_label_payload(
    const std::string& decoded_text,
    const MesWorkOrder& expected
) const {
    if (expected.expected_lot_id.empty()) return false;
    return decoded_text.find(expected.expected_lot_id) != std::string::npos;
}

bool MesSyncWorker::report_cycle_result(
    const std::string& cycle_id,
    const std::string& outcome,
    const std::string& reason
) {
    if (!is_enabled()) return false;
    json snap{
        {"cycle_id", cycle_id},
        {"transition_event_id", cycle_id + "_" + random_token_hex(4)},
        {"outcome", outcome},
        {"result_reason", reason},
    };
    auto req = prepare_cycle_outbox_request(snap, "manual_report");
    if (!req.has_value()) return false;
    try {
        (void)db_.mes_enqueue_outbox(*req);
        notify_outbox_enqueued();
        return true;
    } catch (...) {
        return false;
    }
}

bool MesSyncWorker::is_connected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cfg_.enabled && connected_;
}

}  // namespace igbt
