#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "igbt/hal/adam_modbus_dio.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <vector>

namespace igbt {

namespace {

int64_t steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

void ensure_winsock_once() {
#ifdef _WIN32
    static std::once_flag flag;
    std::call_once(flag, []() {
        WSADATA wsa{};
        (void)WSAStartup(MAKEWORD(2, 2), &wsa);
    });
#endif
}

bool channel_bit(uint16_t mask, int ch) {
    if (ch < 0 || ch >= 16) return false;
    return (mask & static_cast<uint16_t>(1U << ch)) != 0;
}

void set_do_bit(uint8_t& mask, int ch, bool val) {
    if (ch < 0 || ch >= 8) return;
    if (val) {
        mask = static_cast<uint8_t>(mask | static_cast<uint8_t>(1U << ch));
    } else {
        mask = static_cast<uint8_t>(mask & ~static_cast<uint8_t>(1U << ch));
    }
}

uint16_t read_be16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}

void write_be16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[1] = static_cast<uint8_t>(v & 0xFF);
}

}  // namespace

class AdamModbusDioDriver::ModbusTcpSocket {
public:
    ModbusTcpSocket() {
        ensure_winsock_once();
    }

    ~ModbusTcpSocket() {
        close_socket();
    }

    bool is_connected() const noexcept {
#ifdef _WIN32
        return sock_ != INVALID_SOCKET;
#else
        return false;
#endif
    }

    void close_socket() noexcept {
#ifdef _WIN32
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
#endif
    }

    bool ensure_connected(const std::string& host, uint16_t port, int connect_timeout_ms) {
#ifdef _WIN32
        if (sock_ != INVALID_SOCKET) return true;
        if (host.empty() || port == 0) return false;

        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        addrinfo* res = nullptr;
        const std::string port_str = std::to_string(port);
        if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0 || !res) {
            return false;
        }

        const int64_t deadline = steady_now_ms() + std::max(50, connect_timeout_ms);
        SOCKET connected_sock = INVALID_SOCKET;

        for (addrinfo* ptr = res; ptr != nullptr; ptr = ptr->ai_next) {
            SOCKET s = ::socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
            if (s == INVALID_SOCKET) continue;

            u_long nonblock = 1;
            if (ioctlsocket(s, FIONBIO, &nonblock) != 0) {
                closesocket(s);
                continue;
            }
            int nodelay = 1;
            (void)setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

            const int rc = ::connect(s, ptr->ai_addr, static_cast<int>(ptr->ai_addrlen));
            if (rc == 0) {
                connected_sock = s;
                break;
            }
            const int err = WSAGetLastError();
            if (err != WSAEWOULDBLOCK && err != WSAEINPROGRESS) {
                closesocket(s);
                continue;
            }

            const int rem = static_cast<int>(std::max<int64_t>(1, deadline - steady_now_ms()));
            WSAPOLLFD pfd{};
            pfd.fd = s;
            pfd.events = POLLWRNORM;
            const int prc = WSAPoll(&pfd, 1, rem);
            if (prc <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
                closesocket(s);
                continue;
            }
            int so_err = 0;
            int so_len = sizeof(so_err);
            if (getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so_err), &so_len) != 0 || so_err != 0) {
                closesocket(s);
                continue;
            }
            connected_sock = s;
            break;
        }
        freeaddrinfo(res);
        sock_ = connected_sock;
        return sock_ != INVALID_SOCKET;
#else
        (void)host;
        (void)port;
        (void)connect_timeout_ms;
        return false;
#endif
    }

    bool write_multiple_coils(
        uint8_t unit_id,
        uint16_t start_addr,
        uint16_t count,
        uint8_t bit_mask,
        int io_timeout_ms
    ) {
        if (count == 0 || count > 8) return false;
        const uint16_t tx_id = next_tx_id();
        std::array<uint8_t, 14> req{};
        write_be16(&req[0], tx_id);
        write_be16(&req[2], 0x0000);
        write_be16(&req[4], 8);
        req[6] = unit_id;
        req[7] = 0x0F;  // FC15: Write Multiple Coils
        write_be16(&req[8], start_addr);
        write_be16(&req[10], count);
        req[12] = 1;    // byte count
        req[13] = bit_mask;

        const int64_t deadline = steady_now_ms() + std::max(50, io_timeout_ms);
        if (!send_all(req.data(), req.size(), deadline)) {
            close_socket();
            return false;
        }

        std::array<uint8_t, 7> mbap{};
        if (!recv_exact(mbap.data(), mbap.size(), deadline)) {
            close_socket();
            return false;
        }
        const uint16_t rx_tx = read_be16(&mbap[0]);
        const uint16_t rx_proto = read_be16(&mbap[2]);
        const uint16_t rx_len = read_be16(&mbap[4]);
        const uint8_t rx_unit = mbap[6];
        if (rx_tx != tx_id || rx_proto != 0x0000 || rx_unit != unit_id || rx_len < 2 || rx_len > 64) {
            close_socket();
            return false;
        }

        std::vector<uint8_t> pdu(rx_len - 1);
        if (!recv_exact(pdu.data(), pdu.size(), deadline)) {
            close_socket();
            return false;
        }
        if (pdu.size() != 5 || pdu[0] != 0x0F) {
            close_socket();
            return false;
        }
        const uint16_t echo_addr = read_be16(&pdu[1]);
        const uint16_t echo_cnt = read_be16(&pdu[3]);
        if (echo_addr != start_addr || echo_cnt != count) {
            close_socket();
            return false;
        }
        return true;
    }

    bool read_discrete_inputs(
        uint8_t unit_id,
        uint16_t start_addr,
        uint16_t count,
        uint16_t& out_mask,
        int io_timeout_ms
    ) {
        if (count == 0 || count > 16) return false;
        const uint16_t tx_id = next_tx_id();
        std::array<uint8_t, 12> req{};
        write_be16(&req[0], tx_id);
        write_be16(&req[2], 0x0000);
        write_be16(&req[4], 6);
        req[6] = unit_id;
        req[7] = 0x02;  // FC02: Read Discrete Inputs
        write_be16(&req[8], start_addr);
        write_be16(&req[10], count);

        const int64_t deadline = steady_now_ms() + std::max(50, io_timeout_ms);
        if (!send_all(req.data(), req.size(), deadline)) {
            close_socket();
            return false;
        }

        std::array<uint8_t, 7> mbap{};
        if (!recv_exact(mbap.data(), mbap.size(), deadline)) {
            close_socket();
            return false;
        }
        const uint16_t rx_tx = read_be16(&mbap[0]);
        const uint16_t rx_proto = read_be16(&mbap[2]);
        const uint16_t rx_len = read_be16(&mbap[4]);
        const uint8_t rx_unit = mbap[6];
        const uint8_t expected_bytes = static_cast<uint8_t>((count + 7) / 8);
        if (rx_tx != tx_id || rx_proto != 0x0000 || rx_unit != unit_id || rx_len < 2 || rx_len > 64) {
            close_socket();
            return false;
        }

        std::vector<uint8_t> pdu(rx_len - 1);
        if (!recv_exact(pdu.data(), pdu.size(), deadline)) {
            close_socket();
            return false;
        }
        if (pdu.size() != static_cast<std::size_t>(2 + expected_bytes) ||
            pdu[0] != 0x02 || pdu[1] != expected_bytes) {
            close_socket();
            return false;
        }

        uint16_t mask = 0;
        for (uint8_t b = 0; b < expected_bytes; ++b) {
            mask |= static_cast<uint16_t>(static_cast<uint16_t>(pdu[2 + b]) << (b * 8));
        }
        if (count < 16) {
            mask &= static_cast<uint16_t>((1U << count) - 1U);
        }
        out_mask = mask;
        return true;
    }

private:
    uint16_t next_tx_id() noexcept {
        ++tx_seq_;
        if (tx_seq_ == 0) tx_seq_ = 1;
        return tx_seq_;
    }

    bool send_all(const uint8_t* data, std::size_t len, int64_t deadline_ms) {
#ifdef _WIN32
        std::size_t sent = 0;
        while (sent < len) {
            const int64_t rem = deadline_ms - steady_now_ms();
            if (rem <= 0) return false;

            WSAPOLLFD pfd{};
            pfd.fd = sock_;
            pfd.events = POLLWRNORM;
            const int prc = WSAPoll(&pfd, 1, static_cast<int>(rem));
            if (prc <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
                return false;
            }
            const int n = ::send(
                sock_,
                reinterpret_cast<const char*>(data + sent),
                static_cast<int>(len - sent),
                0
            );
            if (n > 0) {
                sent += static_cast<std::size_t>(n);
                continue;
            }
            const int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK || err == WSAEINTR) {
                continue;
            }
            return false;
        }
        return true;
#else
        (void)data;
        (void)len;
        (void)deadline_ms;
        return false;
#endif
    }

    bool recv_exact(uint8_t* out, std::size_t len, int64_t deadline_ms) {
#ifdef _WIN32
        std::size_t got = 0;
        while (got < len) {
            const int64_t rem = deadline_ms - steady_now_ms();
            if (rem <= 0) return false;

            WSAPOLLFD pfd{};
            pfd.fd = sock_;
            pfd.events = POLLRDNORM;
            const int prc = WSAPoll(&pfd, 1, static_cast<int>(rem));
            if (prc <= 0 || (pfd.revents & (POLLERR | POLLNVAL))) {
                return false;
            }
            const int n = ::recv(
                sock_,
                reinterpret_cast<char*>(out + got),
                static_cast<int>(len - got),
                0
            );
            if (n > 0) {
                got += static_cast<std::size_t>(n);
                continue;
            }
            if (n == 0) {
                return false;
            }
            const int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK || err == WSAEINTR) {
                continue;
            }
            return false;
        }
        return true;
#else
        (void)out;
        (void)len;
        (void)deadline_ms;
        return false;
#endif
    }

#ifdef _WIN32
    SOCKET sock_{INVALID_SOCKET};
#endif
    uint16_t tx_seq_{0};
};

json adam_config_to_json(const AdamModbusConfig& cfg) {
    return json{
        {"enabled", cfg.enabled},
        {"host", cfg.host},
        {"port", cfg.port},
        {"unit_id", cfg.unit_id},
        {"di_base_addr", cfg.di_base_addr},
        {"do_base_addr", cfg.do_base_addr},
        {"di_count", cfg.di_count},
        {"do_count", cfg.do_count},
        {"do_ch_system_healthy", cfg.do_ch_system_healthy},
        {"do_ch_pass_allow", cfg.do_ch_pass_allow},
        {"do_ch_heartbeat_toggle", cfg.do_ch_heartbeat_toggle},
        {"do_ch_alarm_lamp", cfg.do_ch_alarm_lamp},
        {"di_ch_estop", cfg.di_ch_estop},
        {"estop_active_low", cfg.estop_active_low},
        {"di_ch_part_sensor", cfg.di_ch_part_sensor},
        {"di_ch_relay_aux", cfg.di_ch_relay_aux},
        {"di_ch_esd_ground", cfg.di_ch_esd_ground},
        {"enable_estop_di_check", cfg.enable_estop_di_check},
        {"enable_relay_aux_check", cfg.enable_relay_aux_check},
        {"enable_esd_di_check", cfg.enable_esd_di_check},
        {"connect_timeout_ms", cfg.connect_timeout_ms},
        {"io_timeout_ms", cfg.io_timeout_ms},
        {"heartbeat_period_ms", cfg.heartbeat_period_ms},
        {"max_consecutive_timeouts", cfg.max_consecutive_timeouts},
        {"vision_stall_timeout_ms", cfg.vision_stall_timeout_ms},
    };
}

AdamModbusConfig adam_config_from_json(const json& j, const AdamModbusConfig& base) {
    AdamModbusConfig cfg = base;
    if (!j.is_object()) return cfg;

    auto read_bool = [&](const char* k, bool& dst) {
        if (j.contains(k) && j[k].is_boolean()) dst = j[k].get<bool>();
    };
    auto read_int = [&](const char* k, int& dst, int min_v, int max_v) {
        if (j.contains(k) && j[k].is_number_integer() && !j[k].is_boolean()) {
            dst = std::clamp(j[k].get<int>(), min_v, max_v);
        }
    };
    auto read_u16 = [&](const char* k, uint16_t& dst, int min_v, int max_v) {
        if (j.contains(k) && j[k].is_number_integer() && !j[k].is_boolean()) {
            dst = static_cast<uint16_t>(std::clamp(j[k].get<int>(), min_v, max_v));
        }
    };

    read_bool("enabled", cfg.enabled);
    if (j.contains("host") && j["host"].is_string()) {
        cfg.host = j["host"].get<std::string>();
    }
    read_u16("port", cfg.port, 1, 65535);
    if (j.contains("unit_id") && j["unit_id"].is_number_integer() && !j["unit_id"].is_boolean()) {
        cfg.unit_id = static_cast<uint8_t>(std::clamp(j["unit_id"].get<int>(), 0, 255));
    }
    read_u16("di_base_addr", cfg.di_base_addr, 0, 65535);
    read_u16("do_base_addr", cfg.do_base_addr, 0, 65535);
    read_u16("di_count", cfg.di_count, 1, 16);
    read_u16("do_count", cfg.do_count, 1, 8);

    read_int("do_ch_system_healthy", cfg.do_ch_system_healthy, 0, 7);
    read_int("do_ch_pass_allow", cfg.do_ch_pass_allow, 0, 7);
    read_int("do_ch_heartbeat_toggle", cfg.do_ch_heartbeat_toggle, 0, 7);
    read_int("do_ch_alarm_lamp", cfg.do_ch_alarm_lamp, -1, 7);

    read_int("di_ch_estop", cfg.di_ch_estop, 0, 15);
    read_bool("estop_active_low", cfg.estop_active_low);
    read_int("di_ch_part_sensor", cfg.di_ch_part_sensor, -1, 15);
    read_int("di_ch_relay_aux", cfg.di_ch_relay_aux, -1, 15);
    read_int("di_ch_esd_ground", cfg.di_ch_esd_ground, -1, 15);
    read_bool("enable_estop_di_check", cfg.enable_estop_di_check);
    read_bool("enable_relay_aux_check", cfg.enable_relay_aux_check);
    read_bool("enable_esd_di_check", cfg.enable_esd_di_check);

    read_int("connect_timeout_ms", cfg.connect_timeout_ms, 50, 5000);
    read_int("io_timeout_ms", cfg.io_timeout_ms, 50, 5000);
    read_int("heartbeat_period_ms", cfg.heartbeat_period_ms, 20, 5000);
    read_int("max_consecutive_timeouts", cfg.max_consecutive_timeouts, 1, 20);
    read_int("vision_stall_timeout_ms", cfg.vision_stall_timeout_ms, 100, 30000);
    return cfg;
}

AdamModbusConfig load_adam_config_file(const std::filesystem::path& path) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, bytes) || bytes.empty()) {
        return {};
    }
    try {
        json parsed = json::parse(bytes.begin(), bytes.end());
        return adam_config_from_json(parsed);
    } catch (...) {
        return {};
    }
}

json dio_health_to_json(const DioHealthStatus& st) {
    return json{
        {"backend", st.backend},
        {"configured", st.configured},
        {"connected", st.connected},
        {"hardware_ready", st.hardware_ready},
        {"fault_locked", st.fault_locked},
        {"fault_reason", st.fault_reason.empty() ? json(nullptr) : json(st.fault_reason)},
        {"commanded_signal", interlock_signal_to_string(st.commanded_signal)},
        {"commanded_do_mask", st.commanded_do_mask},
        {"inputs", {
            {"estop_active", st.inputs.estop_active},
            {"part_sensor_active", st.inputs.part_sensor_active},
            {"relay_aux_closed", st.inputs.relay_aux_closed},
            {"esd_ground_ok", st.inputs.esd_ground_ok},
            {"raw_di_mask", st.inputs.raw_di_mask},
        }},
        {"consecutive_timeouts", st.consecutive_timeouts},
        {"total_heartbeats", st.total_heartbeats},
        {"total_comm_errors", st.total_comm_errors},
        {"last_rtt_ms", st.last_rtt_ms},
        {"last_ok_mono_ms", st.last_ok_mono_ms},
        {"vision_stalled", st.vision_stalled},
        {"relay_weld_fault", st.relay_weld_fault},
    };
}

AdamModbusDioDriver::AdamModbusDioDriver(
    AdamModbusConfig config,
    VisionRunningFn vision_running_fn,
    VisionTickMsFn vision_tick_fn,
    FaultCallback fault_callback
)
    : cfg_(std::move(config)),
      vision_running_fn_(std::move(vision_running_fn)),
      vision_tick_fn_(std::move(vision_tick_fn)),
      fault_cb_(std::move(fault_callback)),
      sock_(std::make_unique<ModbusTcpSocket>()) {
    cfg_.di_count = std::clamp<uint16_t>(cfg_.di_count, 1, 16);
    cfg_.do_count = std::clamp<uint16_t>(cfg_.do_count, 1, 8);
    cfg_.connect_timeout_ms = std::clamp(cfg_.connect_timeout_ms, 50, 5000);
    cfg_.io_timeout_ms = std::clamp(cfg_.io_timeout_ms, 50, 5000);
    cfg_.heartbeat_period_ms = std::clamp(cfg_.heartbeat_period_ms, 20, 5000);
    cfg_.max_consecutive_timeouts = std::clamp(cfg_.max_consecutive_timeouts, 1, 20);
    cfg_.vision_stall_timeout_ms = std::clamp(cfg_.vision_stall_timeout_ms, 100, 30000);
}

AdamModbusDioDriver::~AdamModbusDioDriver() {
    stop();
}

void AdamModbusDioDriver::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_.load()) return;
    stop_requested_ = false;
    wake_requested_ = false;
    running_.store(true);
    worker_ = std::thread([this]() { worker_loop(); });
}

void AdamModbusDioDriver::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_.load() && !worker_.joinable()) {
            sock_->close_socket();
            return;
        }
        stop_requested_ = true;
        wake_requested_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    running_.store(false);
    // Final Fail-Safe zero DO command on orderly shutdown
    if (sock_ && sock_->is_connected()) {
        (void)sock_->write_multiple_coils(cfg_.unit_id, cfg_.do_base_addr, cfg_.do_count, 0x00, cfg_.io_timeout_ms);
        sock_->close_socket();
    }
}

bool AdamModbusDioDriver::set_interlock(InterlockSignal signal) {
    bool ready = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        commanded_signal_ = signal;
        wake_requested_ = true;
        ready = connected_ && !fault_locked_;
    }
    cv_.notify_one();
    return ready;
}

bool AdamModbusDioDriver::pulse_watchdog() {
    bool ready = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        wake_requested_ = true;
        ready = connected_ && !fault_locked_;
    }
    cv_.notify_one();
    return ready;
}

bool AdamModbusDioDriver::is_hardware_ready() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return connected_ && !fault_locked_;
}

bool AdamModbusDioDriver::is_esd_ground_ok() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return inputs_.esd_ground_ok;
}

DioHealthStatus AdamModbusDioDriver::health_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    DioHealthStatus st;
    st.backend = "adam_modbus_tcp";
    st.configured = cfg_.enabled;
    st.connected = connected_;
    st.hardware_ready = connected_ && !fault_locked_;
    st.fault_locked = fault_locked_;
    st.fault_reason = fault_reason_;
    st.commanded_signal = commanded_signal_;
    st.commanded_do_mask = last_do_mask_;
    st.inputs = inputs_;
    st.consecutive_timeouts = consecutive_timeouts_;
    st.total_heartbeats = total_heartbeats_;
    st.total_comm_errors = total_comm_errors_;
    st.last_rtt_ms = last_rtt_ms_;
    st.last_ok_mono_ms = last_ok_mono_ms_;
    st.vision_stalled = vision_stalled_;
    st.relay_weld_fault = relay_weld_fault_;
    return st;
}

bool AdamModbusDioDriver::acknowledge_relay_fault() {
    bool prev_locked = false;
    std::string prev_reason;
    bool cleared = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        prev_locked = fault_locked_;
        prev_reason = fault_reason_;
        if (relay_weld_fault_) {
            relay_weld_fault_ = false;
            relay_weld_streak_ = 0;
            cleared = true;
            if (fault_reason_ == "RELAY_CONTACT_WELD_FAULT") {
                fault_locked_ = false;
                fault_reason_.clear();
            }
            wake_requested_ = true;
        }
    }
    if (cleared) {
        cv_.notify_one();
        notify_fault_if_changed(prev_locked, prev_reason);
    }
    return cleared;
}

void AdamModbusDioDriver::notify_fault_if_changed(bool prev_locked, const std::string& prev_reason) {
    FaultCallback cb;
    bool cur_locked = false;
    std::string cur_reason;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cb = fault_cb_;
        cur_locked = fault_locked_;
        cur_reason = fault_reason_;
    }
    if (cb && (cur_locked != prev_locked || cur_reason != prev_reason)) {
        try {
            cb(cur_locked, cur_reason);
        } catch (...) {
        }
    }
}

uint8_t AdamModbusDioDriver::compute_do_mask_locked(bool force_safe_zero) const {
    if (force_safe_zero || fault_locked_ || inputs_.estop_active ||
        relay_weld_fault_ || vision_stalled_ || commanded_signal_ == InterlockSignal::FAULT_HALT) {
        return 0x00;
    }

    uint8_t mask = 0;
    set_do_bit(mask, cfg_.do_ch_system_healthy, true);
    set_do_bit(mask, cfg_.do_ch_pass_allow, commanded_signal_ == InterlockSignal::PASS_ALLOW);
    set_do_bit(mask, cfg_.do_ch_heartbeat_toggle, heartbeat_toggle_);
    set_do_bit(mask, cfg_.do_ch_alarm_lamp, commanded_signal_ == InterlockSignal::ALARM_LATCHED);
    return mask;
}

void AdamModbusDioDriver::decode_di_and_update_locked(
    uint16_t raw_di_mask,
    bool pass_allow_cmd,
    bool& need_immediate_safe_zero
) {
    need_immediate_safe_zero = false;
    inputs_.raw_di_mask = raw_di_mask;

    if (cfg_.enable_estop_di_check) {
        const bool bit = channel_bit(raw_di_mask, cfg_.di_ch_estop);
        inputs_.estop_active = cfg_.estop_active_low ? !bit : bit;
    } else {
        inputs_.estop_active = false;
    }

    inputs_.part_sensor_active = (cfg_.di_ch_part_sensor >= 0)
                                     ? channel_bit(raw_di_mask, cfg_.di_ch_part_sensor)
                                     : false;
    inputs_.relay_aux_closed = (cfg_.di_ch_relay_aux >= 0)
                                   ? channel_bit(raw_di_mask, cfg_.di_ch_relay_aux)
                                   : false;
    inputs_.esd_ground_ok = (cfg_.enable_esd_di_check && cfg_.di_ch_esd_ground >= 0)
                                ? channel_bit(raw_di_mask, cfg_.di_ch_esd_ground)
                                : true;

    if (cfg_.enable_relay_aux_check && cfg_.di_ch_relay_aux >= 0) {
        if (!pass_allow_cmd && inputs_.relay_aux_closed) {
            ++relay_weld_streak_;
            if (relay_weld_streak_ >= 2 && !relay_weld_fault_) {
                relay_weld_fault_ = true;
                need_immediate_safe_zero = true;
            }
        } else if (!relay_weld_fault_) {
            relay_weld_streak_ = 0;
        }
    }

    if (inputs_.estop_active) {
        fault_locked_ = true;
        fault_reason_ = "HARDWARE_ESTOP_ACTIVE";
        need_immediate_safe_zero = true;
    } else if (relay_weld_fault_) {
        fault_locked_ = true;
        fault_reason_ = "RELAY_CONTACT_WELD_FAULT";
        need_immediate_safe_zero = true;
    } else if (!vision_stalled_) {
        fault_locked_ = false;
        fault_reason_.clear();
    }
}

void AdamModbusDioDriver::worker_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_requested_) break;
        }

        run_heartbeat_tick();

        std::unique_lock<std::mutex> lock(mutex_);
        if (stop_requested_) break;
        if (!wake_requested_) {
            cv_.wait_for(
                lock,
                std::chrono::milliseconds(cfg_.heartbeat_period_ms),
                [this]() { return stop_requested_ || wake_requested_; }
            );
        }
        wake_requested_ = false;
        if (stop_requested_) break;
    }
}

void AdamModbusDioDriver::run_heartbeat_tick() {
    bool prev_locked = false;
    std::string prev_reason;
    bool vision_running = false;
    int64_t last_tick_ms = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        prev_locked = fault_locked_;
        prev_reason = fault_reason_;
    }

    if (vision_running_fn_) {
        vision_running = vision_running_fn_();
    }
    if (vision_tick_fn_) {
        last_tick_ms = vision_tick_fn_();
    }

    const int64_t now_ms = steady_now_ms();
    const bool is_stalled_now =
        vision_running && last_tick_ms > 0 && (now_ms - last_tick_ms) > cfg_.vision_stall_timeout_ms;

    if (is_stalled_now) {
        bool send_zero_now = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!vision_stalled_ || !stall_safe_zero_sent_) {
                send_zero_now = true;
                stall_safe_zero_sent_ = true;
            }
            vision_stalled_ = true;
            fault_locked_ = true;
            fault_reason_ = "VISION_LOOP_STALL_WATCHDOG_STOP";
            last_do_mask_ = 0x00;
        }
        if (send_zero_now && sock_->is_connected()) {
            (void)sock_->write_multiple_coils(
                cfg_.unit_id,
                cfg_.do_base_addr,
                cfg_.do_count,
                0x00,
                cfg_.io_timeout_ms
            );
        }
        notify_fault_if_changed(prev_locked, prev_reason);
        return;
    }

    uint8_t do_mask = 0;
    bool pass_allow_cmd = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (vision_stalled_) {
            vision_stalled_ = false;
            stall_safe_zero_sent_ = false;
            if (fault_reason_ == "VISION_LOOP_STALL_WATCHDOG_STOP") {
                fault_locked_ = false;
                fault_reason_.clear();
            }
        }
        heartbeat_toggle_ = !heartbeat_toggle_;
        do_mask = compute_do_mask_locked(false);
        pass_allow_cmd = channel_bit(do_mask, cfg_.do_ch_pass_allow);
    }

    const int64_t t_start = steady_now_ms();
    bool comm_ok = sock_->ensure_connected(cfg_.host, cfg_.port, cfg_.connect_timeout_ms);
    uint16_t raw_di_mask = 0;

    if (comm_ok) {
        comm_ok = sock_->write_multiple_coils(
            cfg_.unit_id,
            cfg_.do_base_addr,
            cfg_.do_count,
            do_mask,
            cfg_.io_timeout_ms
        );
    }
    if (comm_ok) {
        comm_ok = sock_->read_discrete_inputs(
            cfg_.unit_id,
            cfg_.di_base_addr,
            cfg_.di_count,
            raw_di_mask,
            cfg_.io_timeout_ms
        );
    }
    const int64_t rtt_ms = std::max<int64_t>(0, steady_now_ms() - t_start);

    if (!comm_ok) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            connected_ = false;
            ++consecutive_timeouts_;
            ++total_comm_errors_;
            last_do_mask_ = 0x00;
            if (consecutive_timeouts_ >= cfg_.max_consecutive_timeouts) {
                fault_locked_ = true;
                fault_reason_ = "MODBUS_WATCHDOG_TIMEOUT";
            }
        }
        notify_fault_if_changed(prev_locked, prev_reason);
        return;
    }

    bool need_safe_zero = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        connected_ = true;
        consecutive_timeouts_ = 0;
        ++total_heartbeats_;
        last_rtt_ms_ = rtt_ms;
        last_ok_mono_ms_ = steady_now_ms();
        last_do_mask_ = do_mask;
        decode_di_and_update_locked(raw_di_mask, pass_allow_cmd, need_safe_zero);
        if (need_safe_zero) {
            last_do_mask_ = 0x00;
        }
    }

    if (need_safe_zero && sock_->is_connected()) {
        (void)sock_->write_multiple_coils(
            cfg_.unit_id,
            cfg_.do_base_addr,
            cfg_.do_count,
            0x00,
            cfg_.io_timeout_ms
        );
    }

    notify_fault_if_changed(prev_locked, prev_reason);
}

}  // namespace igbt
