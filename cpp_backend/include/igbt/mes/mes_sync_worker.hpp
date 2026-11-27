#pragma once

#include "igbt/common.hpp"
#include "igbt/database.hpp"
#include "igbt/industrial_hal.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace igbt {

inline constexpr int kMaxWhitelistTubes = 50;
inline constexpr int64_t kMaxWhitelistDurationMs = 30LL * 60 * 1000;

struct MesClientConfig {
    bool enabled{false};
    std::string base_url;
    std::string station_id{"STATION_01"};
    std::string api_key;
    std::string signing_secret{"IGBT_MES_LOCAL_CACHE_V46"};

    std::string heartbeat_path{"/api/mes/v1/heartbeat"};
    std::string validate_lot_path{"/api/mes/v1/lots/validate"};
    std::string pass_record_path{"/api/mes/v1/pass_records"};

    int connect_timeout_ms{1500};
    int io_timeout_ms{3000};
    int heartbeat_period_ms{5000};
    int64_t heartbeat_fail_lock_ms{60000};
    int outbox_poll_ms{1000};
    int64_t base_backoff_ms{1000};
    int64_t max_backoff_ms{60000};
    int max_retries{12};

    int whitelist_max_tubes{kMaxWhitelistTubes};
    int64_t whitelist_max_duration_ms{kMaxWhitelistDurationMs};
};

int64_t mes_default_wall_clock_ms();

json mes_config_to_json(const MesClientConfig& cfg, bool mask_secrets = false);
MesClientConfig mes_config_from_json(
    const json& j,
    const MesClientConfig& base = MesClientConfig{}
);
MesClientConfig load_mes_config_file(const std::filesystem::path& path);
bool save_mes_config_file(const std::filesystem::path& path, const MesClientConfig& cfg);

std::string compute_mes_idempotency_key(
    const std::string& station_id,
    const std::string& lot_id,
    const std::string& event_uuid,
    int64_t tube_seq
);

std::string compute_mes_lot_signature(
    const std::string& secret,
    const std::string& station_id,
    const std::string& work_order_id,
    const std::string& lot_id,
    const std::string& part_number,
    const std::string& expected_pipe_color,
    int64_t validated_at_ms,
    int64_t expires_at_ms
);

class MesSyncWorker final : public IMesClient {
public:
    using WallClockMsFn = std::function<int64_t()>;
    using InterlockCallback = std::function<void(InterlockSignal)>;
    using LotChangeCallback = std::function<void(const std::string& lot_id, const std::string& actor_id)>;

    MesSyncWorker(
        StationDatabase& db,
        MesClientConfig config = {},
        std::filesystem::path config_path = {},
        WallClockMsFn wall_clock = mes_default_wall_clock_ms,
        InterlockCallback interlock_cb = nullptr,
        LotChangeCallback lot_change_cb = nullptr
    );
    ~MesSyncWorker() override;

    MesSyncWorker(const MesSyncWorker&) = delete;
    MesSyncWorker& operator=(const MesSyncWorker&) = delete;

    void start();
    void stop() override;

    void set_wall_clock_for_test(WallClockMsFn fn);
    void set_interlock_callback(InterlockCallback cb);
    void set_lot_change_callback(LotChangeCallback cb);
    void notify_outbox_enqueued();

    json reconfigure(const MesClientConfig& new_cfg, bool persist_file = true);
    json reconfigure_from_json(const json& patch, bool persist_file = true);

    json tick_once(bool force_heartbeat = false);
    void evaluate_offline_policy_now();

    json bind_work_order_online(const json& payload, const std::string& default_actor);
    json set_whitelist_policy(const json& payload, const std::string& default_actor);
    json unlock_recall_lock(const json& payload, const std::string& default_actor);

    std::optional<MesOutboxEnqueueRequest> prepare_cycle_outbox_request(
        const json& snap,
        const std::string& model_sha256
    );
    bool check_pipe_color_mismatch(const std::string& observed_pipe_color) const;

    bool is_enabled() const;
    bool is_blocking() const;
    bool is_recall_locked() const;
    bool is_offline_locked() const;
    std::string block_reason() const;

    MesClientConfig config() const;
    json status_json() const override;

    // IMesClient interface
    std::optional<MesWorkOrder> active_work_order() const override;
    bool verify_label_payload(const std::string& decoded_text, const MesWorkOrder& expected) const override;
    bool report_cycle_result(
        const std::string& cycle_id,
        const std::string& outcome,
        const std::string& reason
    ) override;
    bool is_connected() const override;
    std::string backend_name() const override { return "mes_outbox_http"; }

private:
    struct ParsedEndpoint {
        std::string host;
        int port{80};
        std::string base_path;
    };

    struct EvaluatedPolicyState {
        bool enabled{false};
        bool connected{false};
        bool lot_bound{false};
        bool signature_valid{false};
        bool lot_expired{false};
        bool recall_locked{false};
        bool offline_locked{false};
        bool blocking{false};
        std::string block_reason;
        std::string degradation_mode{"DISABLED"};
        int64_t offline_elapsed_ms{0};
        int64_t whitelist_remaining_ms{0};
        int whitelist_remaining_tubes{0};
        int whitelist_max_tubes{kMaxWhitelistTubes};
        int64_t whitelist_max_duration_ms{kMaxWhitelistDurationMs};
        std::optional<json> active_lot;
        json policy;
    };

    static ParsedEndpoint parse_http_endpoint(const std::string& base_url);
    static std::string join_endpoint_path(const std::string& base_path, const std::string& rel_path);

    int64_t current_wall_ms() const;
    void worker_loop();
    void run_heartbeat_probe(const MesClientConfig& cfg, const ParsedEndpoint& ep, int64_t now_ms);
    int drain_outbox_once(const MesClientConfig& cfg, const ParsedEndpoint& ep, int64_t now_ms);

    void trigger_recall_lock(
        const std::string& status,
        const std::string& lot_id,
        const std::string& work_order_id,
        const std::string& reason,
        int64_t now_ms
    );

    EvaluatedPolicyState evaluate_policy_snapshot(int64_t now_ms, bool emit_lock_effects);
    void emit_interlock(InterlockSignal sig);

    StationDatabase& db_;
    std::filesystem::path config_path_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    bool stop_requested_{false};
    bool wake_requested_{false};

    MesClientConfig cfg_;
    WallClockMsFn wall_clock_;
    InterlockCallback interlock_cb_;
    LotChangeCallback lot_change_cb_;

    bool connected_{false};
    bool ever_connected_{false};
    int64_t last_heartbeat_attempt_ms_{0};
    int64_t last_heartbeat_ok_ms_{0};
    int64_t last_rtt_ms_{0};
    std::string last_heartbeat_error_;
    int64_t offline_since_ms_{0};
    bool offline_locked_{false};
    std::string offline_lock_reason_;
    bool offline_lock_audit_emitted_{false};
};

}  // namespace igbt
