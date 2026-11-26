#pragma once

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

#include "igbt/common.hpp"
#include "igbt/database.hpp"
#include "igbt/hal/adam_modbus_dio.hpp"
#include "igbt/industrial_hal.hpp"
#include "igbt/red_rabbit.hpp"
#include "igbt/training_manager.hpp"
#include "igbt/vision_engine.hpp"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace igbt {

class EngineerSessionAuth {
public:
    static constexpr int kDefaultTtlSeconds = 1800;
    static constexpr std::size_t kMaxActiveSessions = 256;

    EngineerSessionAuth() = default;

    json status_json() const;
    json verify_passcode(const std::string& passcode);

    std::string require_admin_token(const std::optional<std::string>& supplied);
    std::string require_station_token(const std::optional<std::string>& supplied) const;
    std::string require_station_or_admin_token(
        const std::optional<std::string>& x_station_token,
        const std::optional<std::string>& x_admin_token
    );

    bool verify_session_token(const std::string& token);
    void clear_sessions();
    void expire_all_sessions();
    void set_session_ttl_seconds(int ttl_seconds);
    void set_lockout_policy(int max_attempts, double lockout_seconds);
    void reset_rate_limit();

    // Optional test environment overrides (empty map falls back to process environment)
    void set_env_override(const std::string& key, const std::optional<std::string>& value);
    void clear_env_overrides();
    std::string get_env_value(const std::string& key) const;

private:
    void prune_expired_locked(double now_mono);
    std::string mint_session(int ttl_seconds);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, double> sessions_;
    std::unordered_map<std::string, std::optional<std::string>> env_overrides_;
    int ttl_seconds_{kDefaultTtlSeconds};
    int failed_attempts_{0};
    double lockout_until_mono_{0.0};
    int max_failed_attempts_{5};
    double lockout_seconds_{60.0};
};

struct StationServerOptions {
    std::string host{"127.0.0.1"};
    int port{8765};
    std::filesystem::path data_dir;
    std::filesystem::path web_dist_dir;
    std::filesystem::path assets_dir;
    bool enable_live_camera{false};
    std::string initial_work_mode{"strict"};
    std::optional<AdamModbusConfig> adam_config;
};

class StationHttpServer {
public:
    StationHttpServer(
        StationDatabase& db,
        LiveCameraEngine& camera,
        TrainingManager& training,
        StationServerOptions options = {}
    );
    ~StationHttpServer();

    StationHttpServer(const StationHttpServer&) = delete;
    StationHttpServer& operator=(const StationHttpServer&) = delete;

    // Auto-load newest verified ANN_MLP (.xml) or ONNX (.onnx) model run from disk
    bool autoload_latest_verified_model();

    // Start listening synchronously (blocking) on 127.0.0.1:port
    bool listen_blocking();

    // Start listening on a background thread (returns bound port, or -1 on failure; port=0 binds ephemeral port)
    int start_background();
    void stop();

    bool is_running() const noexcept { return running_.load(); }
    int bound_port() const noexcept { return bound_port_.load(); }

    EngineerSessionAuth& auth() noexcept { return auth_; }
    const EngineerSessionAuth& auth() const noexcept { return auth_; }
    RedRabbitController& red_rabbit() noexcept { return red_rabbit_; }
    const RedRabbitController& red_rabbit() const noexcept { return red_rabbit_; }
    const IDioInterlock& interlock() const noexcept { return *interlock_; }
    const IMesClient& mes_client() const noexcept { return mes_client_; }
    httplib::Server& raw_server() noexcept { return svr_; }

    static constexpr int kMaxConcurrentMjpegStreams = 6;

    static std::filesystem::path detect_default_package_root();
    static std::filesystem::path detect_default_data_dir();
    static std::filesystem::path detect_default_web_dist_dir();
    static std::filesystem::path detect_default_assets_dir();

private:
    void register_routes();
    void register_static_and_spa_routes();

    bool on_training_completed(const json& run_summary);
    json load_verified_model_into_camera(
        const std::filesystem::path& model_path,
        const std::string& artifact_format,
        const std::string& model_version,
        const std::string& expected_sha256,
        int input_size,
        int input_height = 0,
        const std::filesystem::path& ood_calibration_path = {}
    );

    json build_current_state_response();
    json build_offline_live_state() const;
    static json apply_validity_gate(json state);

    StationDatabase& db_;
    LiveCameraEngine& camera_;
    TrainingManager& training_;
    StationServerOptions options_;
    EngineerSessionAuth auth_;
    std::unique_ptr<IDioInterlock> interlock_;
    OfflineNoopMesClient mes_client_;
    RedRabbitController red_rabbit_;

    void start_maintenance_thread();
    void stop_maintenance_thread();

    httplib::Server svr_;
    std::thread server_thread_;
    std::thread maintenance_thread_;
    std::mutex maintenance_mutex_;
    std::condition_variable maintenance_cv_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<int> bound_port_{0};
    std::atomic<int> active_mjpeg_streams_{0};

    mutable std::mutex state_mutex_;
    double sim_override_until_mono_{0.0};
};

}  // namespace igbt
