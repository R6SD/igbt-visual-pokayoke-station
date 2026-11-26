#pragma once

#include "igbt/common.hpp"
#include "igbt/industrial_hal.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace igbt {

struct AdamModbusConfig {
    bool enabled{false};
    std::string host{"127.0.0.1"};
    uint16_t port{502};
    uint8_t unit_id{1};

    uint16_t di_base_addr{0x0000};
    uint16_t do_base_addr{0x0010};
    uint16_t di_count{6};
    uint16_t do_count{6};

    // Positive-logic fail-safe DO channels (DO=1 energized healthy; DO=0 de-energized NC lock)
    int do_ch_system_healthy{0};
    int do_ch_pass_allow{1};
    int do_ch_heartbeat_toggle{2};
    int do_ch_alarm_lamp{3};

    // DI channels (E-Stop active-low by default so broken wire trips Fail-Safe halt)
    int di_ch_estop{0};
    bool estop_active_low{true};
    int di_ch_part_sensor{1};
    int di_ch_relay_aux{2};
    int di_ch_esd_ground{3};
    bool enable_estop_di_check{true};
    bool enable_relay_aux_check{true};
    bool enable_esd_di_check{false};

    // Deterministic non-blocking timeouts & watchdog parameters
    int connect_timeout_ms{500};
    int io_timeout_ms{200};
    int heartbeat_period_ms{500};
    int max_consecutive_timeouts{2};
    int vision_stall_timeout_ms{1000};
};

json adam_config_to_json(const AdamModbusConfig& cfg);
AdamModbusConfig adam_config_from_json(
    const json& j,
    const AdamModbusConfig& base = AdamModbusConfig{}
);
AdamModbusConfig load_adam_config_file(const std::filesystem::path& path);
json dio_health_to_json(const DioHealthStatus& st);

class AdamModbusDioDriver final : public IDioInterlock {
public:
    using VisionRunningFn = std::function<bool()>;
    using VisionTickMsFn = std::function<int64_t()>;
    using FaultCallback = std::function<void(bool fault_locked, const std::string& reason)>;

    explicit AdamModbusDioDriver(
        AdamModbusConfig config,
        VisionRunningFn vision_running_fn = nullptr,
        VisionTickMsFn vision_tick_fn = nullptr,
        FaultCallback fault_callback = nullptr
    );
    ~AdamModbusDioDriver() override;

    AdamModbusDioDriver(const AdamModbusDioDriver&) = delete;
    AdamModbusDioDriver& operator=(const AdamModbusDioDriver&) = delete;

    void start();
    void stop() override;

    bool set_interlock(InterlockSignal signal) override;
    bool pulse_watchdog() override;
    bool is_hardware_ready() const override;
    bool is_esd_ground_ok() const override;
    std::string backend_name() const override { return "adam_modbus_tcp"; }

    DioHealthStatus health_status() const override;
    bool acknowledge_relay_fault() override;

    const AdamModbusConfig& config() const noexcept { return cfg_; }

private:
    class ModbusTcpSocket;

    void worker_loop();
    void run_heartbeat_tick();
    uint8_t compute_do_mask_locked(bool force_safe_zero) const;
    void decode_di_and_update_locked(uint16_t raw_di_mask, bool pass_allow_cmd, bool& need_immediate_safe_zero);
    void notify_fault_if_changed(bool prev_locked, const std::string& prev_reason);

    AdamModbusConfig cfg_;
    VisionRunningFn vision_running_fn_;
    VisionTickMsFn vision_tick_fn_;
    FaultCallback fault_cb_;

    std::unique_ptr<ModbusTcpSocket> sock_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    bool stop_requested_{false};
    bool wake_requested_{false};

    InterlockSignal commanded_signal_{InterlockSignal::LOCKED};
    bool heartbeat_toggle_{false};
    uint8_t last_do_mask_{0};

    bool connected_{false};
    bool fault_locked_{false};
    std::string fault_reason_;
    bool vision_stalled_{false};
    bool stall_safe_zero_sent_{false};
    bool relay_weld_fault_{false};
    int relay_weld_streak_{0};

    DioInputSnapshot inputs_{};
    int consecutive_timeouts_{0};
    int64_t total_heartbeats_{0};
    int64_t total_comm_errors_{0};
    int64_t last_rtt_ms_{0};
    int64_t last_ok_mono_ms_{0};
};

}  // namespace igbt
