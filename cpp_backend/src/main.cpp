#include "igbt/http_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <dbghelp.h>
#endif

namespace {

std::atomic<bool> g_shutdown_requested{false};
#ifdef _WIN32
wchar_t g_dump_dir_w[1024] = {0};

void prune_old_dumps(const std::filesystem::path& dumps_dir) {
    struct DumpFile {
        std::filesystem::file_time_type mtime{};
        std::filesystem::path path;
    };
    std::vector<DumpFile> files;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dumps_dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        if (it->path().extension() != L".dmp") continue;
        files.push_back(DumpFile{it->last_write_time(fec), it->path()});
    }
    if (files.size() <= 5) return;
    std::sort(files.begin(), files.end(), [](const DumpFile& a, const DumpFile& b) {
        return a.mtime > b.mtime;
    });
    for (std::size_t i = 5; i < files.size(); ++i) {
        std::error_code rme;
        std::filesystem::remove(files[i].path, rme);
    }
}

LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* exc_info) {
    if (g_dump_dir_w[0] != L'\0') {
        FILETIME ft{};
        GetSystemTimeAsFileTime(&ft);
        ULARGE_INTEGER uli{};
        uli.LowPart = ft.dwLowDateTime;
        uli.HighPart = ft.dwHighDateTime;
        // Convert Windows 100ns FILETIME ticks (since 1601) to Unix epoch milliseconds
        const unsigned long long epoch_ms =
            (uli.QuadPart >= 116444736000000000ULL)
                ? ((uli.QuadPart - 116444736000000000ULL) / 10000ULL)
                : 0ULL;

        wchar_t dump_path[1280] = {0};
        std::swprintf(
            dump_path,
            sizeof(dump_path) / sizeof(dump_path[0]),
            L"%ls\\crash_%lu_%llu.dmp",
            g_dump_dir_w,
            static_cast<unsigned long>(GetCurrentProcessId()),
            epoch_ms
        );

        HANDLE hFile = CreateFileW(
            dump_path,
            GENERIC_WRITE,
            FILE_SHARE_READ,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (hFile != INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION mei{};
            mei.ThreadId = GetCurrentThreadId();
            mei.ExceptionPointers = exc_info;
            mei.ClientPointers = FALSE;
            MiniDumpWriteDump(
                GetCurrentProcess(),
                GetCurrentProcessId(),
                hFile,
                MiniDumpWithIndirectlyReferencedMemory,
                exc_info ? &mei : nullptr,
                nullptr,
                nullptr
            );
            CloseHandle(hFile);
            try {
                prune_old_dumps(std::filesystem::path(g_dump_dir_w));
            } catch (...) {
            }
        }
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

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

#ifdef _WIN32
        {
            std::error_code dec;
            const auto dumps_dir = data_dir / "dumps";
            std::filesystem::create_directories(dumps_dir, dec);
            const std::wstring wdir = dumps_dir.wstring();
            if (!wdir.empty() && wdir.size() < (sizeof(g_dump_dir_w) / sizeof(g_dump_dir_w[0]))) {
                std::wmemcpy(g_dump_dir_w, wdir.c_str(), wdir.size() + 1);
                SetUnhandledExceptionFilter(unhandled_exception_filter);
            }
        }
#endif

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
