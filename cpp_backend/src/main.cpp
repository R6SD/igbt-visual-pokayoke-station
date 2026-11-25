#include "igbt/http_server.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_shutdown_requested{false};

std::string env_or_default(const wchar_t* name, const std::string& fallback) {
    if (const wchar_t* val = _wgetenv(name); val != nullptr && val[0] != L'\0') {
        return igbt::path_to_utf8(std::filesystem::path(val));
    }
    DWORD needed = GetEnvironmentVariableW(name, nullptr, 0);
    if (needed > 1) {
        std::wstring buf(needed, L'\0');
        DWORD copied = GetEnvironmentVariableW(name, buf.data(), needed);
        if (copied > 0) {
            buf.resize(copied);
            return igbt::path_to_utf8(std::filesystem::path(buf));
        }
    }
    return fallback;
}

void signal_handler(int) {
    g_shutdown_requested.store(true);
}

BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            g_shutdown_requested.store(true);
            return TRUE;
        default:
            return FALSE;
    }
}

}  // namespace

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

    try {
        int port = 8765;
        try {
            port = std::stoi(env_or_default(L"IGBT_UI_PORT", "8765"));
            if (port < 1 || port > 65535) port = 8765;
        } catch (...) {
            port = 8765;
        }

        int cam_index = 0;
        try {
            cam_index = std::stoi(env_or_default(L"IGBT_CAMERA_INDEX", "0"));
            if (cam_index < 0) cam_index = 0;
        } catch (...) {
            cam_index = 0;
        }

        const bool enable_live_camera = (env_or_default(L"IGBT_ENABLE_LIVE_CAMERA", "1") == "1");
        std::string work_mode = env_or_default(L"IGBT_WORK_MODE", "strict");
        if (work_mode != "strict" && work_mode != "debug") {
            work_mode = "strict";
        }

        const auto data_dir = igbt::StationHttpServer::detect_default_data_dir();
        const auto web_dist_dir = igbt::StationHttpServer::detect_default_web_dist_dir();
        const auto assets_dir = igbt::StationHttpServer::detect_default_assets_dir();

        igbt::StationDatabase db(data_dir);
        igbt::LiveCameraEngine camera(cam_index);
        igbt::TrainingManager training(data_dir);

        igbt::StationServerOptions opts;
        opts.host = "127.0.0.1";  // Strictly loopback-only by design
        opts.port = port;
        opts.data_dir = data_dir;
        opts.web_dist_dir = web_dist_dir;
        opts.assets_dir = assets_dir;
        opts.enable_live_camera = enable_live_camera;
        opts.initial_work_mode = work_mode;

        igbt::StationHttpServer server(db, camera, training, opts);

        std::cout << "[IGBT-C++20] Starting Visual Poka-Yoke Station v4.0 on http://127.0.0.1:" << port
                  << " (camera=" << (enable_live_camera ? "ON" : "OFF")
                  << ", mode=" << work_mode
                  << ", res=1280x720@25fps)" << std::endl;

        const int bound = server.start_background();
        if (bound <= 0) {
            std::cerr << "[IGBT-C++20] Failed to bind http://127.0.0.1:" << port << std::endl;
            return 1;
        }

        while (!g_shutdown_requested.load() && server.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        server.stop();
        std::cout << "[IGBT-C++20] Server stopped cleanly." << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[IGBT-C++20] Fatal startup error: " << exc.what() << std::endl;
        return 1;
    }
}
