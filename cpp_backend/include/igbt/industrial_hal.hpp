#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace igbt {

/**
 * Physical hardware interlock state sent to an external PLC / isolated DIO module.
 */
enum class InterlockSignal {
    LOCKED,
    PASS_ALLOW,
    ALARM_LATCHED,
    FAULT_HALT
};

inline std::string interlock_signal_to_string(InterlockSignal sig) {
    switch (sig) {
        case InterlockSignal::LOCKED: return "LOCKED";
        case InterlockSignal::PASS_ALLOW: return "PASS_ALLOW";
        case InterlockSignal::ALARM_LATCHED: return "ALARM_LATCHED";
        case InterlockSignal::FAULT_HALT: return "FAULT_HALT";
    }
    return "LOCKED";
}

struct DioInputSnapshot {
    bool estop_active{false};
    bool part_sensor_active{false};
    bool relay_aux_closed{false};
    bool esd_ground_ok{true};
    uint16_t raw_di_mask{0};
};

struct DioHealthStatus {
    std::string backend{"shadow_noop"};
    bool configured{false};
    bool connected{false};
    bool hardware_ready{false};
    bool fault_locked{false};
    std::string fault_reason;
    InterlockSignal commanded_signal{InterlockSignal::LOCKED};
    uint8_t commanded_do_mask{0};
    DioInputSnapshot inputs{};
    int consecutive_timeouts{0};
    int64_t total_heartbeats{0};
    int64_t total_comm_errors{0};
    int64_t last_rtt_ms{0};
    int64_t last_ok_mono_ms{0};
    bool vision_stalled{false};
    bool relay_weld_fault{false};
};

/**
 * Abstract hardware digital I/O interlock and watchdog interface (Wave 3 HAL).
 */
class IDioInterlock {
public:
    virtual ~IDioInterlock() = default;
    virtual bool set_interlock(InterlockSignal signal) = 0;
    virtual bool pulse_watchdog() = 0;
    virtual bool is_hardware_ready() const = 0;
    virtual bool is_esd_ground_ok() const = 0;
    virtual std::string backend_name() const = 0;

    virtual DioHealthStatus health_status() const {
        DioHealthStatus st;
        st.backend = backend_name();
        st.hardware_ready = is_hardware_ready();
        st.inputs.esd_ground_ok = is_esd_ground_ok();
        return st;
    }
    virtual bool acknowledge_relay_fault() { return false; }
    virtual void stop() {}
};

/**
 * Default non-actuating shadow interlock used until physical DIO hardware is attached.
 */
class ShadowNoopInterlock final : public IDioInterlock {
public:
    bool set_interlock(InterlockSignal) override { return false; }
    bool pulse_watchdog() override { return false; }
    bool is_hardware_ready() const override { return false; }
    bool is_esd_ground_ok() const override { return true; }
    std::string backend_name() const override { return "shadow_noop"; }
};

/**
 * Active manufacturing work order metadata provided by a plant MES endpoint.
 */
struct MesWorkOrder {
    std::string work_order_id;
    std::string part_number;
    std::string expected_lot_id;
    std::string expected_pipe_color;
};

/**
 * Abstract Manufacturing Execution System (MES) client interface (Wave 3 HAL).
 */
class IMesClient {
public:
    virtual ~IMesClient() = default;
    virtual std::optional<MesWorkOrder> active_work_order() const = 0;
    virtual bool verify_label_payload(const std::string& decoded_text, const MesWorkOrder& expected) const = 0;
    virtual bool report_cycle_result(
        const std::string& cycle_id,
        const std::string& outcome,
        const std::string& reason
    ) = 0;
    virtual bool is_connected() const = 0;
    virtual std::string backend_name() const = 0;
};

/**
 * Default disconnected MES stub used when no external plant MES is configured.
 */
class OfflineNoopMesClient final : public IMesClient {
public:
    std::optional<MesWorkOrder> active_work_order() const override { return std::nullopt; }
    bool verify_label_payload(const std::string&, const MesWorkOrder&) const override { return false; }
    bool report_cycle_result(const std::string&, const std::string&, const std::string&) override { return false; }
    bool is_connected() const override { return false; }
    std::string backend_name() const override { return "offline_noop"; }
};

}  // namespace igbt
