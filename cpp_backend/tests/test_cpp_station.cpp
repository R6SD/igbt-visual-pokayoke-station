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

#include "igbt/http_server.hpp"

#include <sqlite3.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace igbt;

#define REQUIRE(cond)                                                                                  \
    do {                                                                                               \
        if (!(cond)) {                                                                                 \
            std::ostringstream _oss;                                                                   \
            _oss << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " -> " << #cond;          \
            throw std::runtime_error(_oss.str());                                                      \
        }                                                                                              \
    } while (0)

#define REQUIRE_EQ(actual, expected)                                                                   \
    do {                                                                                               \
        auto _a = (actual);                                                                            \
        auto _e = (expected);                                                                          \
        if (!(_a == _e)) {                                                                             \
            std::ostringstream _oss;                                                                   \
            _oss << "Equality failed at " << __FILE__ << ":" << __LINE__                               \
                 << " (" << #actual << " == " << #expected << "): got [" << _a << "] vs [" << _e << "]"; \
            throw std::runtime_error(_oss.str());                                                      \
        }                                                                                              \
    } while (0)

class ScopedTempDir {
public:
    ScopedTempDir() {
        std::error_code ec;
        auto base = std::filesystem::temp_directory_path(ec);
        std::wstring folder = L"igbt_测试_cpp_" + std::to_wstring(GetCurrentProcessId()) + L"_" +
                              path_from_utf8(random_token_hex(4)).wstring();
        path_ = base / folder;
        std::filesystem::create_directories(path_, ec);
    }
    ~ScopedTempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

cv::Mat make_synthetic_station_frame(
    int width = kDefaultCameraWidth,
    int height = kDefaultCameraHeight,
    const std::string& label_kind = "complete",
    bool with_glare = false,
    bool with_bare_skin = false
) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(48, 52, 56));
    // Subtle background texture so Laplacian sharpness easily passes >= 120
    for (int y = 0; y < height; y += 8) {
        cv::line(img, cv::Point(0, y), cv::Point(width, y), cv::Scalar(58, 62, 66), 1);
    }

    // Yellow transparent IGBT tube across middle
    cv::Rect tube_rect(
        static_cast<int>(0.17 * width),
        static_cast<int>(0.43 * height),
        static_cast<int>(0.66 * width),
        static_cast<int>(0.14 * height)
    );
    cv::rectangle(img, tube_rect, cv::Scalar(35, 195, 235), cv::FILLED);
    cv::rectangle(img, tube_rect, cv::Scalar(20, 130, 175), 3);

    // White ESD gloves on left and right grip zones (unless testing bare skin)
    cv::Rect left_hand(
        static_cast<int>(0.18 * width),
        static_cast<int>(0.40 * height),
        static_cast<int>(0.11 * width),
        static_cast<int>(0.36 * height)
    );
    cv::Rect right_hand(
        static_cast<int>(0.71 * width),
        static_cast<int>(0.40 * height),
        static_cast<int>(0.11 * width),
        static_cast<int>(0.36 * height)
    );
    if (with_bare_skin) {
        cv::rectangle(img, left_hand, cv::Scalar(115, 150, 215), cv::FILLED);
        cv::rectangle(img, right_hand, cv::Scalar(115, 150, 215), cv::FILLED);
    } else {
        cv::rectangle(img, left_hand, cv::Scalar(235, 238, 240), cv::FILLED);
        cv::rectangle(img, right_hand, cv::Scalar(235, 238, 240), cv::FILLED);
    }

    // Label ROI area (0.43..0.57 x 0.45..0.55)
    cv::Rect label_rect(
        static_cast<int>(0.43 * width),
        static_cast<int>(0.45 * height),
        static_cast<int>(0.14 * width),
        static_cast<int>(0.10 * height)
    );

    if (label_kind == "complete" || label_kind == "crossed_out") {
        cv::rectangle(img, label_rect, cv::Scalar(212, 215, 218), cv::FILLED);
        cv::rectangle(img, label_rect, cv::Scalar(65, 70, 75), 2);
        // Crisp mid-dark barcode stripes (above underexposure threshold 40, > 250 Laplacian variance)
        for (int x = label_rect.x + 10; x < label_rect.x + label_rect.width - 10; x += 8) {
            cv::line(
                img,
                cv::Point(x, label_rect.y + 10),
                cv::Point(x, label_rect.y + label_rect.height - 10),
                cv::Scalar(60, 62, 65),
                2
            );
        }
        if (label_kind == "crossed_out") {
            cv::line(
                img,
                cv::Point(label_rect.x + 6, label_rect.y + 6),
                cv::Point(label_rect.x + label_rect.width - 6, label_rect.y + label_rect.height - 6),
                cv::Scalar(50, 50, 185),
                5
            );
            cv::line(
                img,
                cv::Point(label_rect.x + label_rect.width - 6, label_rect.y + 6),
                cv::Point(label_rect.x + 6, label_rect.y + label_rect.height - 6),
                cv::Scalar(50, 50, 185),
                5
            );
        }
    } else if (label_kind == "covered") {
        cv::rectangle(img, label_rect, cv::Scalar(55, 58, 62), cv::FILLED);
        for (int x = label_rect.x + 6; x < label_rect.x + label_rect.width - 6; x += 8) {
            cv::line(
                img,
                cv::Point(x, label_rect.y + 6),
                cv::Point(x, label_rect.y + label_rect.height - 6),
                cv::Scalar(95, 98, 102),
                1
            );
        }
    } else {
        // "absent": bare yellow tube + internal chip outlines for sharpness
        for (int x = label_rect.x + 8; x < label_rect.x + label_rect.width - 8; x += 10) {
            cv::rectangle(
                img,
                cv::Rect(x, label_rect.y + 12, 6, label_rect.height - 24),
                cv::Scalar(58, 70, 80),
                cv::FILLED
            );
        }
    }

    if (with_glare) {
        cv::Rect glare_rect(label_rect.x, label_rect.y, label_rect.width, label_rect.height / 2);
        cv::rectangle(img, glare_rect, cv::Scalar(255, 255, 255), cv::FILLED);
    }

    return img;
}

FSMEvent make_valid_obs(
    const std::string& cycle_id,
    int64_t frame_id,
    double ts,
    const std::string& label_class,
    bool marker = false
) {
    FSMEvent ev;
    ev.kind = "observation";
    ev.cycle_id = cycle_id;
    ev.frame_id = frame_id;
    ev.capture_monotonic_s = ts;
    ev.label_class = label_class;
    ev.view_valid = true;
    ev.quality_ok = true;
    ev.camera_healthy = true;
    ev.camera_settings_locked = true;
    ev.config_match = true;
    ev.tube_present = true;
    ev.label_window_present = true;
    ev.label_roi_clear = true;
    ev.hand_occluded = false;
    ev.hand_contact_candidate = false;
    ev.hand_detection_valid = true;
    ev.hand_in_contact_roi = false;
    ev.hand_in_label_contact_roi = false;
    ev.hand_in_left_grip_zone = true;
    ev.hand_in_right_grip_zone = true;
    ev.marker_detected = marker;
    ev.pipe_color = "yellow";
    ev.operator_anon_id = "OP_TEST_01";
    return ev;
}

// ----------------------------------------------------------------------------
// Test 1: InspectionFSM Rules (A / B / C 1200ms / Initial Absent Alarm / Reset)
// ----------------------------------------------------------------------------
void test_fsm_rules() {
    double now = 100.0;
    FSMConfig cfg;
    cfg.initial_complete_ms = 400;
    cfg.stable_state_ms = 400;
    cfg.min_stable_frames = 5;
    cfg.c_absent_duration_ms = 1200;
    cfg.hand_leave_stable_ms = 200;

    // 1A. Class C 1200ms full lifecycle + mid-window occlusion reset
    {
        InspectionFSM fsm(cfg, [&]() { return now; });
        REQUIRE_EQ(fsm.state_str(), "WAIT_PART");

        FSMEvent start_ev;
        start_ev.kind = "new_cycle";
        start_ev.cycle_id = "CYC_FSM_C01";
        start_ev.capture_monotonic_s = now;
        fsm.process(start_ev, now);
        REQUIRE_EQ(fsm.state_str(), "VERIFY_INITIAL");

        int64_t fid = 0;
        for (int i = 0; i < 6; ++i) {
            now += 0.1;
            ++fid;
            fsm.process(make_valid_obs("CYC_FSM_C01", fid, now, "complete"), now);
        }
        REQUIRE_EQ(fsm.state_str(), "READY");

        FSMEvent insp_ev;
        insp_ev.kind = "start_inspection";
        insp_ev.capture_monotonic_s = now;
        fsm.process(insp_ev, now);
        REQUIRE_EQ(fsm.state_str(), "INSPECTING");

        // Central hand contact enters -> PROCESSING
        now += 0.1;
        ++fid;
        auto contact_ev = make_valid_obs("CYC_FSM_C01", fid, now, "complete");
        contact_ev.hand_contact_candidate = true;
        contact_ev.hand_in_contact_roi = true;
        contact_ev.hand_in_label_contact_roi = true;
        contact_ev.hand_in_left_grip_zone = false;
        contact_ev.hand_in_right_grip_zone = false;
        contact_ev.hand_occluded = true;
        contact_ev.label_roi_clear = false;
        fsm.process(contact_ev, now);
        REQUIRE(fsm.contact_seen());

        // Hand leaves, label absent for 600ms (< 1200ms) -> must NOT pass yet
        for (int i = 0; i < 8; ++i) {
            now += 0.1;
            ++fid;
            fsm.process(make_valid_obs("CYC_FSM_C01", fid, now, "absent"), now);
        }
        REQUIRE(fsm.state() != FSMState::PASS);
        REQUIRE(fsm.c_timer_ms() > 0 && fsm.c_timer_ms() < 1200);

        // Brief hand occlusion resets C timer to 0!
        now += 0.1;
        ++fid;
        auto occ_ev = make_valid_obs("CYC_FSM_C01", fid, now, "absent");
        occ_ev.hand_occluded = true;
        occ_ev.label_roi_clear = false;
        fsm.process(occ_ev, now);
        REQUIRE_EQ(fsm.c_timer_ms(), 0);

        // Re-stabilize hand leave + full >= 1200ms continuous absence -> PASS
        for (int i = 0; i < 18; ++i) {
            now += 0.1;
            ++fid;
            fsm.process(make_valid_obs("CYC_FSM_C01", fid, now, "absent"), now);
            if (fsm.state() == FSMState::PASS) break;
        }
        REQUIRE_EQ(fsm.state_str(), "PASS");
        REQUIRE_EQ(fsm.outcome(), "PASS");
        REQUIRE_EQ(fsm.c_timer_ms(), 1200);
    }

    // 1B. Initial absent -> ALARM, unauthorized reset blocked, authorized reset clears
    {
        InspectionFSM fsm(cfg, [&]() { return now; });
        FSMEvent start_ev;
        start_ev.kind = "new_cycle";
        start_ev.cycle_id = "CYC_FSM_ALM";
        start_ev.capture_monotonic_s = now;
        fsm.process(start_ev, now);

        int64_t fid = 0;
        for (int i = 0; i < 6; ++i) {
            now += 0.1;
            ++fid;
            fsm.process(make_valid_obs("CYC_FSM_ALM", fid, now, "absent"), now);
            if (fsm.state() == FSMState::ALARM) break;
        }
        REQUIRE_EQ(fsm.state_str(), "ALARM");
        REQUIRE_EQ(fsm.outcome(), "ALARM");

        // Unauthorized reset must NOT clear ALARM
        FSMEvent unauth_rst;
        unauth_rst.kind = "reset";
        unauth_rst.reset_authorized = false;
        unauth_rst.capture_monotonic_s = now + 0.1;
        now += 0.1;
        fsm.process(unauth_rst, now);
        REQUIRE_EQ(fsm.state_str(), "ALARM");

        // Authorized reset returns to WAIT_PART
        FSMEvent auth_rst;
        auth_rst.kind = "reset";
        auth_rst.reset_authorized = true;
        auth_rst.reason = "AUTHORIZED_RESET";
        auth_rst.capture_monotonic_s = now + 0.1;
        now += 0.1;
        fsm.process(auth_rst, now);
        REQUIRE_EQ(fsm.state_str(), "WAIT_PART");
        REQUIRE_EQ(fsm.outcome(), "WAITING");
    }
}

// ----------------------------------------------------------------------------
// Test 2: 720p@25fps Pacer, ViewValid Occlusion/Glare & Bare-Hand Simulation
// ----------------------------------------------------------------------------
void test_vision_720p_25fps_and_pokayoke() {
    // Default ViewConfig is 1280 x 720
    ViewConfig vcfg = default_view_config();
    REQUIRE_EQ(vcfg.calibrated_width, 1280);
    REQUIRE_EQ(vcfg.calibrated_height, 720);

    // 25.0 FPS pacer: exact 40,000 us step when on time, catch-up when lagging
    auto t0 = std::chrono::steady_clock::now();
    auto d1 = LiveCameraEngine::next_pacer_deadline(t0, t0 + std::chrono::milliseconds(12));
    auto step_us = std::chrono::duration_cast<std::chrono::microseconds>(d1 - t0).count();
    REQUIRE_EQ(step_us, 40000);

    auto late_now = t0 + std::chrono::milliseconds(95);
    auto d2 = LiveCameraEngine::next_pacer_deadline(t0, late_now);
    REQUIRE(d2 >= late_now);

    // ViewValidEvaluator: glare rejection & bidirectional occluder overlap
    cv::Mat clean_frame = make_synthetic_station_frame(1280, 720, "complete", false, false);
    cv::Mat glare_frame = make_synthetic_station_frame(1280, 720, "complete", true, false);

    ViewValidEvaluator evaluator(vcfg);
    double ts = 10.0;
    ViewResult vr{};
    for (int i = 1; i <= 12; ++i) {
        ts += 0.04;
        FrameObservation obs;
        obs.frame_id = i;
        obs.captured_monotonic_s = ts;
        obs.tube_center_norm = {0.50, 0.50};
        obs.label_center_norm = {0.50, 0.50};
        vr = evaluator.evaluate(clean_frame, obs);
    }
    REQUIRE(vr.view_valid);

    // Glare invalidates view
    ts += 0.04;
    FrameObservation glare_obs;
    glare_obs.frame_id = 13;
    glare_obs.captured_monotonic_s = ts;
    glare_obs.tube_center_norm = {0.50, 0.50};
    glare_obs.label_center_norm = {0.50, 0.50};
    ViewResult glare_res = evaluator.evaluate(glare_frame, glare_obs);
    REQUIRE(!glare_res.view_valid);

    // Occluder overlapping label ROI invalidates view
    ts += 0.04;
    FrameObservation occ_obs;
    occ_obs.frame_id = 14;
    occ_obs.captured_monotonic_s = ts;
    occ_obs.tube_center_norm = {0.50, 0.50};
    occ_obs.label_center_norm = {0.50, 0.50};
    occ_obs.occluder_boxes_norm = {Box{0.44, 0.45, 0.56, 0.55}};
    ViewResult occ_res = evaluator.evaluate(clean_frame, occ_obs);
    REQUIRE(!occ_res.view_valid);

    // Simulation scenarios: a_pass, b_pass, c_pass, bare_hand_alarm
    auto sim_a = run_simulation_scenario("a_pass", StationHttpServer::detect_default_assets_dir());
    REQUIRE_EQ(sim_a.snapshot.value("outcome", ""), "PASS");
    auto sim_c = run_simulation_scenario("c_pass", StationHttpServer::detect_default_assets_dir());
    REQUIRE_EQ(sim_c.snapshot.value("outcome", ""), "PASS");
    REQUIRE_EQ(sim_c.snapshot.value("c_timer_ms", 0), 1200);
    auto sim_bare = run_simulation_scenario("bare_hand_alarm", StationHttpServer::detect_default_assets_dir());
    REQUIRE_EQ(sim_bare.snapshot.value("outcome", ""), "ALARM");
}

// ----------------------------------------------------------------------------
// Test 3: SQLite3 Migration, CRUD, Purge & Active Model 409 Protection
// ----------------------------------------------------------------------------
void test_sqlite_storage_and_migration() {
    ScopedTempDir tmp;
    const auto legacy_db_path = tmp.path() / "station.db";

    // Create a pre-v4 SQLite database WITHOUT cycles.source column
    {
        sqlite3* raw_db = nullptr;
        REQUIRE_EQ(
            sqlite3_open_v2(
                path_to_utf8(legacy_db_path).c_str(),
                &raw_db,
                SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                nullptr
            ),
            SQLITE_OK
        );
        const char* legacy_sql = R"SQL(
            CREATE TABLE cycles (
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
                fsm_state TEXT NOT NULL DEFAULT 'WAIT_PART'
            );
            CREATE TABLE events (
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
                payload_json TEXT NOT NULL
            );
            INSERT INTO cycles VALUES('CYC_LEGACY_SIM','2026-11-20T00:00:00Z','2026-11-20T00:00:01Z','complete','PASS','A_CROSSED_OUT','yellow','OP_01','v1','ROI_V001','PASS');
            INSERT INTO events VALUES('EV_SIM_1','CYC_LEGACY_SIM','2026-11-20T00:00:01Z',1.0,1,'A_CROSSED_OUT','INSPECTING','PASS','PASS','A_CROSSED_OUT','{"record_type":"STATE_TRANSITION"}');
        )SQL";
        REQUIRE_EQ(sqlite3_exec(raw_db, legacy_sql, nullptr, nullptr, nullptr), SQLITE_OK);
        sqlite3_close_v2(raw_db);
    }

    // StationDatabase auto-detects station.db, migrates cycles.source, and backfills 'simulated'
    StationDatabase db(tmp.path());
    REQUIRE_EQ(db.db_path().filename().string(), "station.db");

    json def_cfg = db.get_config();
    REQUIRE_EQ(def_cfg.value("camera_width", 0), 1280);
    REQUIRE_EQ(def_cfg.value("camera_height", 0), 720);

    json cycles = db.list_cycles(10);
    REQUIRE_EQ(cycles.size(), 1u);
    REQUIRE_EQ(cycles[0].value("cycle_id", ""), "CYC_LEGACY_SIM");
    REQUIRE_EQ(cycles[0].value("source", ""), "simulated");

    // Save a live manual capture and verify simulated_only purge preserves live cycle
    cv::Mat frame = make_synthetic_station_frame();
    std::vector<uint8_t> jpg;
    cv::imencode(".jpg", frame, jpg);
    json live_st{
        {"cycle_id", "LIVE_TEST_001"},
        {"outcome", "PASS"},
        {"fsm_state", "PASS"},
        {"label_status", "crossed_out"},
        {"pipe_color", "yellow"},
        {"camera_width", 1280},
        {"camera_height", 720},
        {"frame_id", 12},
        {"capture_monotonic_s", 55.2}
    };
    json cap_res = db.save_manual_capture(jpg, live_st, "LIVE_FALLBACK");
    const std::string ev_id = cap_res.value("evidence_id", "");
    REQUIRE(!ev_id.empty());
    auto ev_file = db.get_evidence_file(ev_id);
    REQUIRE(std::filesystem::is_regular_file(ev_file.file_path));

    json purge_sim = db.purge_cycles("simulated_only");
    REQUIRE_EQ(purge_sim.value("deleted_cycles", 0), 1);
    REQUIRE_EQ(purge_sim.value("vacuumed", true), false);
    REQUIRE_EQ(db.list_cycles(10).size(), 1u);
    REQUIRE(std::filesystem::is_regular_file(ev_file.file_path));

    // Active model 409 delete protection
    json reg = db.register_model(
        "opencv-mlp-cpp-igbt",
        "mlp_active_v1",
        std::string(64, 'a'),
        "reviewed",
        "unit test active model",
        "tester"
    );
    const std::string mid = reg.value("model_id", "");
    bool got_409 = false;
    try {
        db.delete_model(mid, "mlp_active_v1");
    } catch (const DatabaseHttpError& err) {
        got_409 = (err.status_code() == 409);
    }
    REQUIRE(got_409);

    // Deleting when inactive succeeds
    json del_m = db.delete_model(mid, "other_version");
    REQUIRE(del_m.value("deleted", false));

    // All-scope purge deletes live cycle + unlinks disk JPEG + runs VACUUM
    json purge_all = db.purge_cycles("all");
    REQUIRE_EQ(purge_all.value("deleted_cycles", 0), 1);
    REQUIRE_EQ(purge_all.value("vacuumed", false), true);
    REQUIRE(!std::filesystem::exists(ev_file.file_path));
    REQUIRE_EQ(db.cycle_summary_all().value("total", -1), 0);
}

// ----------------------------------------------------------------------------
// Test 4: Full HTTP Loopback Server, Auth RBAC, Skip-Frame Auto-Capture,
//         Pure C++ ANN_MLP Training + Hot-Load, ONNX Import & MJPEG Streaming
// ----------------------------------------------------------------------------
void test_http_server_auth_autocapture_training_and_mjpeg() {
    ScopedTempDir tmp;
    const auto web_dir = tmp.path() / "web_dist";
    std::filesystem::create_directories(web_dir / "assets");
    const std::string html = "<!doctype html><html><body>IGBT C++20 Station UI</body></html>";
    REQUIRE(write_file_bytes_atomic(web_dir / "index.html", html.data(), html.size()));

    StationDatabase db(tmp.path());
    LiveCameraEngine camera(0);
    TrainingManager training(tmp.path());

    StationServerOptions opts;
    opts.host = "127.0.0.1";
    opts.port = 0;  // bind ephemeral loopback port
    opts.data_dir = tmp.path();
    opts.web_dist_dir = web_dir;
    opts.assets_dir = StationHttpServer::detect_default_assets_dir();
    opts.enable_live_camera = false;
    opts.initial_work_mode = "debug";

    StationHttpServer server(db, camera, training, opts);

    // 4A. Zero-default-credential check: unconfigured env fails closed (401)
    server.auth().set_env_override("IGBT_ADMIN_TOKEN", "");
    server.auth().set_env_override("IGBT_ENGINEER_PIN", "");
    server.auth().set_env_override("IGBT_STATION_TOKEN", "");

    const int port = server.start_background();
    REQUIRE(port > 0);

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(15, 0);

    // Health & SPA static serving
    auto health_res = cli.Get("/api/health");
    REQUIRE(health_res != nullptr);
    REQUIRE_EQ(health_res->status, 200);
    json health_j = json::parse(health_res->body);
    REQUIRE_EQ(health_j.value("status", ""), "ok");
    REQUIRE_EQ(health_j.value("host_policy", ""), "loopback_only");
    REQUIRE_EQ(health_j.value("interlock_backend", ""), "shadow_noop");
    REQUIRE_EQ(health_j.value("mes_connected", true), false);

    auto spa_res = cli.Get("/showcase");
    REQUIRE(spa_res != nullptr);
    REQUIRE_EQ(spa_res->status, 200);
    REQUIRE(spa_res->body.find("IGBT C++20 Station UI") != std::string::npos);

    // Auth status when unconfigured -> default_pin_active MUST be false, "888888" MUST fail 401
    auto st_unconf = cli.Get("/api/auth/status");
    REQUIRE(st_unconf != nullptr);
    REQUIRE_EQ(st_unconf->status, 200);
    json st_unconf_j = json::parse(st_unconf->body);
    REQUIRE_EQ(st_unconf_j.value("auth_configured", true), false);
    REQUIRE_EQ(st_unconf_j.value("default_pin_active", true), false);

    auto ver_default_pin = cli.Post("/api/auth/verify", R"({"passcode":"888888"})", "application/json");
    REQUIRE(ver_default_pin != nullptr);
    REQUIRE_EQ(ver_default_pin->status, 401);

    auto mut_unconf = cli.Post("/api/camera/work_mode", R"({"work_mode":"strict"})", "application/json");
    REQUIRE(mut_unconf != nullptr);
    REQUIRE_EQ(mut_unconf->status, 401);

    // Configure PIN & Station Token -> verify session minting, 48-hex length, TTL expiry & 401 guards
    server.auth().set_env_override("IGBT_ENGINEER_PIN", "937462");
    server.auth().set_env_override("IGBT_STATION_TOKEN", "station-tok-v4");

    auto bad_pin = cli.Post("/api/auth/verify", R"({"passcode":"888888"})", "application/json");
    REQUIRE(bad_pin != nullptr);
    REQUIRE_EQ(bad_pin->status, 401);

    auto ok_pin = cli.Post("/api/auth/verify", R"({"passcode":"937462"})", "application/json");
    REQUIRE(ok_pin != nullptr);
    REQUIRE_EQ(ok_pin->status, 200);
    json auth_j = json::parse(ok_pin->body);
    REQUIRE_EQ(auth_j.value("authenticated", false), true);
    REQUIRE_EQ(auth_j.value("expires_in_seconds", 0), 1800);
    const std::string session_tok = auth_j.value("admin_token", "");
    REQUIRE_EQ(session_tok.size(), 48u);

    httplib::Headers admin_hdrs = {{"X-Admin-Token", session_tok}};

    // Unauthenticated protected mutations MUST return 401; read-only auto_capture status is public (200)
    REQUIRE_EQ(cli.Post("/api/cycles/purge", R"({"scope":"simulated_only","confirm":true})", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/dataset/clear", "", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/runtime/simulate/a_pass", "", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/camera/capture", "", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/camera/mirror", R"({"mirror":true})", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/camera/auxiliary_rois", R"({"show":false})", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/dataset/capture", R"({"label":"complete"})", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/dataset/bootstrap", "", "application/json")->status, 401);
    REQUIRE_EQ(cli.Post("/api/training/start", "{}", "application/json")->status, 401);
    REQUIRE_EQ(cli.Get("/api/dataset/auto_capture")->status, 200);
    REQUIRE_EQ(cli.Post("/api/dataset/auto_capture", R"({"enabled":true})", "application/json")->status, 401);

    // Authenticated work_mode succeeds; expiring sessions immediately fails closed with 401
    REQUIRE_EQ(cli.Post("/api/camera/work_mode", admin_hdrs, R"({"work_mode":"debug"})", "application/json")->status, 200);
    server.auth().expire_all_sessions();
    REQUIRE_EQ(cli.Post("/api/camera/work_mode", admin_hdrs, R"({"work_mode":"strict"})", "application/json")->status, 401);

    // Mint fresh session token for remaining integration steps
    auto ok_pin2 = cli.Post("/api/auth/verify", R"({"passcode":"937462"})", "application/json");
    REQUIRE(ok_pin2 != nullptr && ok_pin2->status == 200);
    const std::string tok = json::parse(ok_pin2->body).value("admin_token", "");
    admin_hdrs = {{"X-Admin-Token", tok}};

    // 4B. Inject synthetic 1280x720 frames & verify Skip-Frame Auto-Capture (every_n_frames=3, max_samples=4)
    cv::Mat synth_frame = make_synthetic_station_frame(1280, 720, "crossed_out", false, false);
    double inject_ts = 200.0;
    REQUIRE(camera.process_injected_frame(synth_frame, inject_ts));

    auto ac_set = cli.Post(
        "/api/dataset/auto_capture",
        admin_hdrs,
        R"({"enabled":true,"every_n_frames":3,"label":"crossed_out","pipe_color":"yellow","physical_tube_id":"TUBE_AUTO_01","max_samples":4})",
        "application/json"
    );
    REQUIRE(ac_set != nullptr);
    REQUIRE_EQ(ac_set->status, 200);
    REQUIRE_EQ(json::parse(ac_set->body).value("enabled", false), true);

    for (int i = 0; i < 15; ++i) {
        inject_ts += 0.04;
        REQUIRE(camera.process_injected_frame(synth_frame, inject_ts));
    }

    auto ac_get = cli.Get("/api/dataset/auto_capture", admin_hdrs);
    REQUIRE(ac_get != nullptr && ac_get->status == 200);
    json ac_state = json::parse(ac_get->body);
    REQUIRE_EQ(ac_state.value("captured_count", -1), 4);
    REQUIRE_EQ(ac_state.value("enabled", true), false);  // auto-stops at max_samples

    auto samples_res = cli.Get("/api/dataset/samples?label=crossed_out");
    REQUIRE(samples_res != nullptr && samples_res->status == 200);
    json samples_arr = json::parse(samples_res->body);
    REQUIRE_EQ(samples_arr.size(), 4u);

    // 4C. MJPEG Stream & Raw Frame Endpoints
    auto raw_jpg_res = cli.Get("/api/camera/frame.jpg");
    REQUIRE(raw_jpg_res != nullptr && raw_jpg_res->status == 200);
    REQUIRE(raw_jpg_res->body.size() > 512);
    REQUIRE(static_cast<uint8_t>(raw_jpg_res->body[0]) == 0xFF &&
            static_cast<uint8_t>(raw_jpg_res->body[1]) == 0xD8);

    // Offline cached frame MJPEG stream delivers 1 boundary frame and terminates cleanly
    camera.stop();
    std::string mjpeg_buf;
    auto stream_res = cli.Get("/api/camera/stream", [&](const char* data, std::size_t len) {
        mjpeg_buf.append(data, len);
        return false;  // disconnect after first chunk
    });
    REQUIRE(!mjpeg_buf.empty());
    REQUIRE(mjpeg_buf.find("--frame") != std::string::npos);

    // 4D. Bootstrap Standard Dataset -> Pure C++ ANN_MLP Training -> Auto Registry + Hot-Load
    REQUIRE_EQ(cli.Post("/api/dataset/clear", admin_hdrs, "", "application/json")->status, 200);
    auto boot_res = cli.Post("/api/dataset/bootstrap", admin_hdrs, "", "application/json");
    REQUIRE(boot_res != nullptr && boot_res->status == 200);

    auto train_start = cli.Post(
        "/api/training/start",
        admin_hdrs,
        R"({"epochs":18,"input_size":96,"batch_size":8,"learning_rate":0.005,"resume_from_latest":false,"use_augmentation":true})",
        "application/json"
    );
    REQUIRE(train_start != nullptr && train_start->status == 200);

    training.wait_for_completion();
    auto train_st = cli.Get("/api/training/status");
    REQUIRE(train_st != nullptr && train_st->status == 200);
    json train_st_j = json::parse(train_st->body);
    REQUIRE_EQ(train_st_j.value("state", ""), "COMPLETED");

    auto models_res = cli.Get("/api/models");
    REQUIRE(models_res != nullptr && models_res->status == 200);
    json models_arr = json::parse(models_res->body);
    REQUIRE(!models_arr.empty());
    const std::string trained_model_id = models_arr[0].value("model_id", "");
    const std::string trained_version = models_arr[0].value("model_version", "");
    REQUIRE(!trained_model_id.empty());

    // Active model must be hot-loaded into LiveCameraEngine & protected against deletion (409)
    REQUIRE_EQ(camera.get_state_json().value("model_version", ""), trained_version);
    auto del_active = cli.Delete(("/api/models/" + trained_model_id).c_str(), admin_hdrs);
    REQUIRE(del_active != nullptr);
    REQUIRE_EQ(del_active->status, 409);

    // Explicit re-activation via POST /api/models/:id/activate succeeds
    auto act_res = cli.Post(("/api/models/" + trained_model_id + "/activate").c_str(), admin_hdrs, "", "application/json");
    REQUIRE(act_res != nullptr && act_res->status == 200);
    REQUIRE_EQ(json::parse(act_res->body).value("activated", false), true);

    // Corrupt ONNX import rejected with 422; real ONNX import succeeds if reference .onnx exists
    auto bad_onnx = cli.Post("/api/models/import_onnx", admin_hdrs, "NOT_AN_ONNX_FILE", "application/octet-stream");
    REQUIRE(bad_onnx != nullptr);
    REQUIRE_EQ(bad_onnx->status, 422);

    const auto existing_runs = StationHttpServer::detect_default_package_root() / "data" / "runs" / "label_classifier";
    std::error_code ec;
    if (std::filesystem::is_directory(existing_runs, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(existing_runs, ec)) {
            auto candidate = entry.path() / "label_classifier.onnx";
            std::vector<uint8_t> onnx_bytes;
            if (std::filesystem::is_regular_file(candidate, ec) && read_file_bytes(candidate, onnx_bytes)) {
                httplib::Headers imp_hdrs = admin_hdrs;
                imp_hdrs.emplace("X-Model-Version", "imported_onnx_v4_test");
                std::string raw_body(reinterpret_cast<const char*>(onnx_bytes.data()), onnx_bytes.size());
                auto imp_res = cli.Post("/api/models/import_onnx", imp_hdrs, raw_body, "application/octet-stream");
                REQUIRE(imp_res != nullptr && imp_res->status == 200);
                json imp_j = json::parse(imp_res->body);
                REQUIRE_EQ(imp_j.value("imported", false), true);
                REQUIRE_EQ(imp_j.value("activated", false), true);
                REQUIRE_EQ(camera.get_state_json().value("model_version", ""), "imported_onnx_v4_test");
                break;
            }
        }
    }

    // Storage overview & simulated scenario persistence
    auto sim_res = cli.Post("/api/runtime/simulate/a_pass", admin_hdrs, "", "application/json");
    REQUIRE(sim_res != nullptr && sim_res->status == 200);
    REQUIRE_EQ(json::parse(sim_res->body).value("outcome", ""), "PASS");

    auto ov_res = cli.Get("/api/storage/overview");
    REQUIRE(ov_res != nullptr && ov_res->status == 200);
    json ov_j = json::parse(ov_res->body);
    REQUIRE(ov_j.contains("business_db") && ov_j.contains("training_dataset") && ov_j.contains("model_artifacts"));
    REQUIRE(ov_j["business_db"]["tables"].value("cycles", 0) >= 1);

    server.stop();
}

// ----------------------------------------------------------------------------
// Test 5: Wave 1 & Wave 2 Hardening (Strict Default, Auth Rate Limit, SQLite FULL, HAL)
// ----------------------------------------------------------------------------
void test_wave1_wave2_hardening() {
    // 5A. Default work_mode MUST be strict in both StationServerOptions and LiveCameraEngine
    StationServerOptions default_opts;
    REQUIRE_EQ(default_opts.initial_work_mode, "strict");
    LiveCameraEngine default_cam(0);
    REQUIRE_EQ(default_cam.get_work_mode(), "strict");

    // 5B. HAL stub defaults
    ShadowNoopInterlock noop_dio;
    REQUIRE(!noop_dio.is_hardware_ready());
    REQUIRE(noop_dio.is_esd_ground_ok());
    REQUIRE_EQ(noop_dio.backend_name(), "shadow_noop");
    OfflineNoopMesClient noop_mes;
    REQUIRE(!noop_mes.is_connected());
    REQUIRE(!noop_mes.active_work_order().has_value());
    REQUIRE_EQ(noop_mes.backend_name(), "offline_noop");

    // 5C. SQLite synchronous=FULL (2), journal_size_limit=67108864 & checkpoint
    ScopedTempDir tmp;
    StationDatabase db(tmp.path());
    db.run_wal_checkpoint_truncate();
    const json pragmas = db.pragma_diagnostics();
    REQUIRE_EQ(pragmas.value("journal_mode", ""), "wal");
    REQUIRE_EQ(pragmas.value("synchronous", -1), 2);
    REQUIRE_EQ(pragmas.value("journal_size_limit", int64_t{-1}), int64_t{67108864});

    // 5D. Strict optical quality gate enforces underexposure and strict defaults
    json cfg = StationDatabase::default_config();
    cfg["config_version"] = 1;
    cfg["roi_config_version"] = "ROI_V001";
    cfg["calibrated_at_utc"] = utc_now_iso();
    cfg["rois"] = json{
        {"tube_roi", {0.16, 0.42, 0.84, 0.58}},
        {"label_roi", {0.43, 0.45, 0.57, 0.55}},
        {"label_contact_roi", {0.40, 0.38, 0.60, 0.62}},
        {"left_grip_zone", {0.16, 0.34, 0.31, 0.82}},
        {"right_grip_zone", {0.69, 0.34, 0.84, 0.82}}
    };
    cfg["thresholds"] = json::object();  // force strict code-level defaults
    default_cam.update_config(cfg);
    cv::Mat dark_frame(720, 1280, CV_8UC3, cv::Scalar(10, 10, 10));
    REQUIRE(default_cam.process_injected_frame(dark_frame, 50.0));
    json dark_st = default_cam.get_state_json();
    REQUIRE_EQ(dark_st.value("quality_ok", true), false);
    REQUIRE_EQ(dark_st.value("view_valid", true), false);

    // 5E. PIN brute-force rate limit (5 bad attempts -> 429 lockout even for valid PIN)
    const auto web_dir = tmp.path() / "web_dist";
    std::filesystem::create_directories(web_dir);
    const std::string html = "<!doctype html><html><body>OK</body></html>";
    REQUIRE(write_file_bytes_atomic(web_dir / "index.html", html.data(), html.size()));

    TrainingManager training(tmp.path());
    StationServerOptions opts;
    opts.host = "127.0.0.1";
    opts.port = 0;
    opts.data_dir = tmp.path();
    opts.web_dist_dir = web_dir;
    opts.enable_live_camera = false;

    StationHttpServer server(db, default_cam, training, opts);
    server.auth().set_env_override("IGBT_ENGINEER_PIN", "642819");
    server.auth().set_lockout_policy(5, 30.0);
    const int port = server.start_background();
    REQUIRE(port > 0);

    httplib::Client cli("127.0.0.1", port);
    for (int i = 0; i < 4; ++i) {
        auto res = cli.Post("/api/auth/verify", R"({"passcode":"000000"})", "application/json");
        REQUIRE(res != nullptr);
        REQUIRE_EQ(res->status, 401);
    }
    auto res5 = cli.Post("/api/auth/verify", R"({"passcode":"000000"})", "application/json");
    REQUIRE(res5 != nullptr);
    REQUIRE_EQ(res5->status, 429);

    auto res_locked = cli.Post("/api/auth/verify", R"({"passcode":"642819"})", "application/json");
    REQUIRE(res_locked != nullptr);
    REQUIRE_EQ(res_locked->status, 429);

    server.auth().reset_rate_limit();
    auto res_ok = cli.Post("/api/auth/verify", R"({"passcode":"642819"})", "application/json");
    REQUIRE(res_ok != nullptr);
    REQUIRE_EQ(res_ok->status, 200);

    server.stop();
}

// ----------------------------------------------------------------------------
// Test 6: Skip-Frame Ring Buffer, Async Burst Writer, Diff-Gated Capture & Quota
// ----------------------------------------------------------------------------
void test_skip_frame_storage_and_diff_gate() {
    // 6A. SkipFrameRingBuffer (stride=5, capacity=4 -> keeps latest 4 at 5,10,15,20,25,30)
    SkipFrameRingBuffer ring(4, 5);
    cv::Mat sample_img(120, 160, CV_8UC3, cv::Scalar(40, 80, 120));
    int due_pushes = 0;
    for (int fid = 1; fid <= 30; ++fid) {
        if (ring.push_if_due(fid, fid * 0.04, sample_img)) {
            ++due_pushes;
        }
    }
    REQUIRE_EQ(due_pushes, 6);
    auto snap = ring.snapshot_ordered();
    REQUIRE_EQ(snap.size(), 4u);
    REQUIRE_EQ(snap[0].frame_id, int64_t{15});
    REQUIRE_EQ(snap[1].frame_id, int64_t{20});
    REQUIRE_EQ(snap[2].frame_id, int64_t{25});
    REQUIRE_EQ(snap[3].frame_id, int64_t{30});

    // 6B. AsyncEvidenceWriter burst flush + 10s coalescing window
    ScopedTempDir tmp;
    const auto ev_root = tmp.path() / "evidence";
    {
        AsyncEvidenceWriter writer(ev_root, 8, 10.0);
        REQUIRE(writer.enqueue_burst("CYC_BURST_01", "TUBE_SWAP_SUSPECTED", snap, 100.0));
        REQUIRE(!writer.enqueue_burst("CYC_BURST_01", "TUBE_SWAP_SUSPECTED", snap, 102.0));
        writer.stop_and_drain();
        json st = writer.stats_json();
        REQUIRE_EQ(st.value("written_bursts", uint64_t{0}), uint64_t{1});
        REQUIRE_EQ(st.value("written_frames", uint64_t{0}), uint64_t{4});
        REQUIRE_EQ(st.value("coalesced_bursts", uint64_t{0}), uint64_t{1});
    }
    const auto burst_dir = ev_root / "bursts" / "CYC_BURST_01";
    REQUIRE(std::filesystem::is_directory(burst_dir));
    int jpg_cnt = 0;
    int manifest_cnt = 0;
    for (const auto& entry : std::filesystem::directory_iterator(burst_dir)) {
        const auto ext = entry.path().extension().string();
        if (ext == ".jpg") ++jpg_cnt;
        if (ext == ".json") ++manifest_cnt;
    }
    REQUIRE_EQ(jpg_cnt, 4);
    REQUIRE_EQ(manifest_cnt, 1);

    // 6C. Visual-diff gated skip-frame AutoCapture (bypassed for unknown_challenge)
    TrainingManager tm(tmp.path());
    LiveCameraEngine cam(0);
    cam.set_auto_sample_callback(
        [&](const cv::Mat& fb,
            const cv::Mat& cb,
            const std::string& lbl,
            const std::string& pc,
            const std::string& tid,
            const std::string& bid,
            std::optional<Box> box) {
            tm.capture_sample(fb, cb, lbl, pc, tid, bid, box);
        }
    );
    AutoCaptureConfig ac{};
    ac.enabled = true;
    ac.every_n_frames = 2;
    ac.label = "complete";
    ac.max_samples = 10;
    ac.min_diff_mad = 4.0;
    cam.set_auto_capture(ac);

    cv::Mat f_complete = make_synthetic_station_frame(1280, 720, "complete", false, false);
    cv::Mat f_crossed = make_synthetic_station_frame(1280, 720, "crossed_out", false, false);
    double ts = 10.0;
    for (int i = 0; i < 6; ++i) {
        ts += 0.04;
        REQUIRE(cam.process_injected_frame(f_complete, ts));
    }
    REQUIRE_EQ(cam.get_auto_capture().captured_count, 1);

    ts += 0.04;
    REQUIRE(cam.process_injected_frame(f_crossed, ts));
    ts += 0.04;
    REQUIRE(cam.process_injected_frame(f_crossed, ts));
    REQUIRE_EQ(cam.get_auto_capture().captured_count, 2);

    AutoCaptureConfig ac_chal{};
    ac_chal.enabled = true;
    ac_chal.every_n_frames = 1;
    ac_chal.label = "unknown_challenge";
    ac_chal.max_samples = 5;
    ac_chal.min_diff_mad = 20.0;
    cam.set_auto_capture(ac_chal);
    for (int i = 0; i < 2; ++i) {
        ts += 0.04;
        REQUIRE(cam.process_injected_frame(f_complete, ts));
    }
    REQUIRE_EQ(cam.get_auto_capture().captured_count, 2);

    // 6D. Evidence FIFO quota prunes non-anomaly PASS/initial evidence before anomaly evidence
    StationDatabase db(tmp.path() / "quota_db");
    std::vector<uint8_t> jpg_bytes;
    REQUIRE(cv::imencode(".jpg", f_complete, jpg_bytes));
    REQUIRE(jpg_bytes.size() > 1024);

    db.upload_evidence_bytes(
        "CYC_QUOTA_PASS",
        "initial",
        jpg_bytes,
        "image/jpeg",
        1,
        10.0,
        std::string("yellow"),
        std::string("PASS_KEYFRAME"),
        std::string("OP_STATION_01"),
        json::object(),
        std::string("complete"),
        "2026-11-25T10:00:00.000Z"
    );
    db.upload_evidence_bytes(
        "CYC_QUOTA_ALM",
        "anomaly",
        jpg_bytes,
        "image/jpeg",
        2,
        11.0,
        std::string("yellow"),
        std::string("ALARM_CAPTURE"),
        std::string("OP_STATION_01"),
        json::object(),
        std::string("crossed_out"),
        "2026-11-25T10:00:01.000Z"
    );

    json quota_res = db.enforce_evidence_quota(static_cast<std::uint64_t>(jpg_bytes.size() + 256), 0.0);
    REQUIRE_EQ(quota_res.value("deleted_count", 0), 1);
    json rem = db.list_evidence(10);
    REQUIRE_EQ(rem.size(), 1u);
    REQUIRE_EQ(rem[0].value("kind", ""), "anomaly");
}

// ----------------------------------------------------------------------------
// Test 7: v4.2.0 Hardening (bursts/ FIFO Quota, 429 Retry-After, /api/diagnostics/runtime,
//         Bare-Hand Zone & 3-Frame Debounce, Post-Trigger Burst Flush)
// ----------------------------------------------------------------------------
void test_v42_hardening_and_diagnostics() {
    ScopedTempDir tmp;
    StationDatabase db(tmp.path());

    // 7A. Orphan bursts/ directories are FIFO-pruned by enforce_evidence_quota
    const auto bursts_root = db.evidence_root() / "bursts";
    const auto old_dir = bursts_root / "CYC_BURST_01_OLD";
    const auto new_dir = bursts_root / "CYC_BURST_02_NEW";
    std::filesystem::create_directories(old_dir);

    const std::vector<uint8_t> payload(2048, static_cast<uint8_t>('B'));
    const auto old_file = old_dir / "frame_001.jpg";
    REQUIRE(write_file_bytes_atomic(old_file, payload.data(), payload.size()));
    std::this_thread::sleep_for(std::chrono::milliseconds(60));

    std::filesystem::create_directories(new_dir);
    const auto new_file = new_dir / "frame_001.jpg";
    REQUIRE(write_file_bytes_atomic(new_file, payload.data(), payload.size()));

    json quota_res = db.enforce_evidence_quota(2500, 0.0);
    REQUIRE_EQ(quota_res.value("deleted_burst_dirs", 0), 1);
    REQUIRE(!std::filesystem::exists(old_dir));
    REQUIRE(std::filesystem::exists(new_dir));

    // 7B. GET /api/diagnostics/runtime & 429 Retry-After: 60 header
    const auto web_dir = tmp.path() / "web_dist";
    std::filesystem::create_directories(web_dir);
    const std::string html = "<!doctype html><html><body>v4.2 OK</body></html>";
    REQUIRE(write_file_bytes_atomic(web_dir / "index.html", html.data(), html.size()));

    LiveCameraEngine cam_srv(0);
    TrainingManager training(tmp.path());
    StationServerOptions opts;
    opts.host = "127.0.0.1";
    opts.port = 0;
    opts.data_dir = tmp.path();
    opts.web_dist_dir = web_dir;
    opts.enable_live_camera = false;

    StationHttpServer server(db, cam_srv, training, opts);
    server.auth().set_env_override("IGBT_ENGINEER_PIN", "314159");
    server.auth().set_lockout_policy(2, 60.0);
    const int port = server.start_background();
    REQUIRE(port > 0);

    httplib::Client cli("127.0.0.1", port);
    auto diag_res = cli.Get("/api/diagnostics/runtime");
    REQUIRE(diag_res != nullptr);
    REQUIRE_EQ(diag_res->status, 200);
    json diag_j = json::parse(diag_res->body);
    REQUIRE(diag_j.contains("async_evidence") && diag_j["async_evidence"].is_object());
    REQUIRE(diag_j.contains("db_pragmas") && diag_j["db_pragmas"].is_object());
    REQUIRE_EQ(diag_j["db_pragmas"].value("synchronous", 0), 2);
    REQUIRE_EQ(diag_j["db_pragmas"].value("journal_mode", ""), "wal");
    REQUIRE(diag_j.contains("hardware_3a") && diag_j["hardware_3a"].is_object());

    auto auth_fail1 = cli.Post("/api/auth/verify", R"({"passcode":"000000"})", "application/json");
    REQUIRE(auth_fail1 != nullptr);
    REQUIRE_EQ(auth_fail1->status, 401);

    auto auth_fail2 = cli.Post("/api/auth/verify", R"({"passcode":"000000"})", "application/json");
    REQUIRE(auth_fail2 != nullptr);
    REQUIRE_EQ(auth_fail2->status, 429);
    REQUIRE(auth_fail2->has_header("Retry-After"));
    REQUIRE_EQ(auth_fail2->get_header_value("Retry-After"), "60");
    server.stop();

    // 7C. Bare-hand work-zone filter (cy < 0.22 ignored) & 3-frame streak debounce + Post-trigger burst
    auto make_empty_bg = []() {
        cv::Mat bg(kDefaultCameraHeight, kDefaultCameraWidth, CV_8UC3, cv::Scalar(48, 52, 56));
        for (int y = 0; y < bg.rows; y += 8) {
            cv::line(bg, cv::Point(0, y), cv::Point(bg.cols, y), cv::Scalar(58, 62, 66), 1);
        }
        return bg;
    };
    auto make_tube_scene = [&](bool top_skin, bool bare_hand_on_tube) {
        cv::Mat img = make_empty_bg();
        // Continuous bright yellow tube (`>= 0.29` contiguous span) + high-contrast white label
        const cv::Rect tube_r(
            static_cast<int>(0.15 * kDefaultCameraWidth),
            static_cast<int>(0.44 * kDefaultCameraHeight),
            static_cast<int>(0.70 * kDefaultCameraWidth),
            static_cast<int>(0.12 * kDefaultCameraHeight)
        );
        cv::rectangle(img, tube_r, cv::Scalar(35, 195, 235), cv::FILLED);
        cv::rectangle(img, tube_r, cv::Scalar(20, 120, 165), 4);

        const cv::Rect lbl_r(
            static_cast<int>(0.43 * kDefaultCameraWidth),
            static_cast<int>(0.44 * kDefaultCameraHeight),
            static_cast<int>(0.14 * kDefaultCameraWidth),
            static_cast<int>(0.12 * kDefaultCameraHeight)
        );
        cv::rectangle(img, lbl_r, cv::Scalar(242, 244, 246), cv::FILLED);
        cv::rectangle(img, lbl_r, cv::Scalar(40, 42, 45), 4);
        cv::rectangle(
            img,
            cv::Rect(lbl_r.x + 28, lbl_r.y + 18, lbl_r.width - 56, lbl_r.height - 36),
            cv::Scalar(50, 52, 55),
            3
        );

        if (top_skin) {
            cv::rectangle(
                img,
                cv::Rect(static_cast<int>(0.18 * kDefaultCameraWidth), 12, static_cast<int>(0.11 * kDefaultCameraWidth), 115),
                cv::Scalar(115, 150, 215),
                cv::FILLED
            );
        }
        if (bare_hand_on_tube) {
            cv::rectangle(
                img,
                cv::Rect(
                    static_cast<int>(0.18 * kDefaultCameraWidth),
                    static_cast<int>(0.38 * kDefaultCameraHeight),
                    static_cast<int>(0.12 * kDefaultCameraWidth),
                    static_cast<int>(0.36 * kDefaultCameraHeight)
                ),
                cv::Scalar(115, 150, 215),
                cv::FILLED
            );
        }
        return img;
    };
    const cv::Mat empty_bg = make_empty_bg();

    LiveCameraEngine cam(0);
    const auto cam_ev_dir = tmp.path() / "cam_ev";
    cam.set_evidence_root(cam_ev_dir);
    double ts = 1.0;
    for (int i = 0; i < 4; ++i) {
        ts += 0.04;
        REQUIRE(cam.process_injected_frame(empty_bg, ts));
    }

    // Warm skin-colored box outside work-zone (cy < 0.22) must NOT trigger BARE_HAND_VIOLATION
    const cv::Mat f_top_skin = make_tube_scene(true, false);
    for (int i = 0; i < 5; ++i) {
        ts += 0.04;
        REQUIRE(cam.process_injected_frame(f_top_skin, ts));
        const json st = cam.get_state_json();
        REQUIRE_EQ(st.value("bare_hand_violation", true), false);
        REQUIRE_EQ(st.value("alarm_latched", true), false);
    }

    // In-work-zone bare skin touching tube requires 2 consecutive frames (fast trigger) before latching ALARM
    const cv::Mat f_bare = make_tube_scene(false, true);
    ts += 0.04;
    REQUIRE(cam.process_injected_frame(f_bare, ts));
    REQUIRE_EQ(cam.get_state_json().value("alarm_latched", true), false);

    ts += 0.04;
    REQUIRE(cam.process_injected_frame(f_bare, ts));
    {
        const json st2 = cam.get_state_json();
        REQUIRE_EQ(st2.value("alarm_latched", false), true);
        REQUIRE_EQ(st2.value("bare_hand_violation", false), true);
        REQUIRE_EQ(st2.value("result_reason", ""), "BARE_HAND_VIOLATION");
    }

    // Feed 52 more frames so all 10 post-trigger 5-FPS frames finish and flush cleanly
    for (int i = 0; i < 52; ++i) {
        ts += 0.04;
        REQUIRE(cam.process_injected_frame(f_bare, ts));
    }
    cam.stop_async_writer();
    {
        const json stats = cam.async_evidence_stats();
        const json writer = stats.value("writer", json::object());
        REQUIRE(writer.value("written_bursts", uint64_t{0}) >= 1);
        REQUIRE(writer.value("written_frames", uint64_t{0}) >= 10);
    }

    // 7D. Stopping engine mid-burst still flushes pending post-trigger frames safely
    LiveCameraEngine cam_early(0);
    const auto early_ev_dir = tmp.path() / "cam_ev_early";
    cam_early.set_evidence_root(early_ev_dir);
    double ets = 10.0;
    for (int i = 0; i < 4; ++i) {
        ets += 0.04;
        REQUIRE(cam_early.process_injected_frame(empty_bg, ets));
    }
    const cv::Mat f_tube_clean = make_tube_scene(false, false);
    for (int i = 0; i < 2; ++i) {
        ets += 0.04;
        REQUIRE(cam_early.process_injected_frame(f_tube_clean, ets));
    }
    for (int i = 0; i < 3; ++i) {
        ets += 0.04;
        REQUIRE(cam_early.process_injected_frame(f_bare, ets));
    }
    REQUIRE_EQ(cam_early.get_state_json().value("alarm_latched", false), true);
    cam_early.stop();
    cam_early.stop_async_writer();
    REQUIRE(cam_early.async_evidence_stats().value("writer", json::object()).value("written_bursts", uint64_t{0}) >= 1);
}

// ----------------------------------------------------------------------------
// Test 8: Camera Optics Persistence, Closed-Loop Anti-Glare Stepper & Specular Resistance
// ----------------------------------------------------------------------------
void test_camera_optics_and_anti_glare() {
    ScopedTempDir tmp;

    // 8A. SQLite camera_optics defaults, clamping & validation
    {
        StationDatabase db(tmp.path());
        const json def_opt = db.get_camera_optics();
        REQUIRE_EQ(def_opt.value("lock_3a", false), true);
        REQUIRE_EQ(def_opt.value("calibrated", true), false);
        REQUIRE(std::abs(def_opt.value("exposure", 0.0) - (-7.0)) < 1e-6);

        bool rejected_extra = false;
        try {
            db.save_camera_optics(json{{"unknown_knob", 1}}, "tester");
        } catch (const DatabaseHttpError& err) {
            REQUIRE_EQ(err.status_code(), 422);
            rejected_extra = true;
        }
        REQUIRE(rejected_extra);

        const json saved = db.save_camera_optics(
            json{{"lock_3a", true}, {"exposure", -20.0}, {"gain", 999.0}, {"focus", 42}, {"backlight_comp", 0}},
            "eng-op"
        );
        REQUIRE(std::abs(saved.value("exposure", 0.0) - (-13.0)) < 1e-6);
        REQUIRE(std::abs(saved.value("gain", 0.0) - 255.0) < 1e-6);
        REQUIRE_EQ(saved.value("focus", -1), 42);
        REQUIRE_EQ(saved.value("calibrated", false), true);
        REQUIRE_EQ(saved.value("changed_by", ""), "eng-op");
    }

    // Reopen DB and verify optics persisted across restarts
    StationDatabase db(tmp.path());
    {
        const json reloaded = db.get_camera_optics();
        REQUIRE(std::abs(reloaded.value("exposure", 0.0) - (-13.0)) < 1e-6);
        REQUIRE_EQ(reloaded.value("focus", -1), 42);
        REQUIRE_EQ(reloaded.value("calibrated", false), true);
    }

    // 8B. HTTP /api/camera/optics & /api/camera/optics/auto_calibrate & /api/diagnostics/runtime
    const auto web_dir = tmp.path() / "web_dist";
    std::filesystem::create_directories(web_dir);
    const std::string html = "<!doctype html><html><body>optics OK</body></html>";
    REQUIRE(write_file_bytes_atomic(web_dir / "index.html", html.data(), html.size()));

    LiveCameraEngine cam_srv(0);
    TrainingManager training(tmp.path());
    StationServerOptions opts;
    opts.host = "127.0.0.1";
    opts.port = 0;
    opts.data_dir = tmp.path();
    opts.web_dist_dir = web_dir;
    opts.enable_live_camera = false;

    StationHttpServer server(db, cam_srv, training, opts);
    server.auth().set_env_override("IGBT_ADMIN_TOKEN", "optics-admin-secret");
    const int port = server.start_background();
    REQUIRE(port > 0);

    httplib::Client cli("127.0.0.1", port);
    {
        auto g = cli.Get("/api/camera/optics");
        REQUIRE(g != nullptr);
        REQUIRE_EQ(g->status, 200);
        const json gj = json::parse(g->body);
        REQUIRE_EQ(gj.value("focus", -1), 42);
        REQUIRE_EQ(gj.value("calibrated", false), true);
    }
    {
        auto unauth = cli.Post("/api/camera/optics", R"({"exposure":-8})", "application/json");
        REQUIRE(unauth != nullptr);
        REQUIRE_EQ(unauth->status, 401);
    }
    {
        httplib::Headers hdr{{"X-Admin-Token", "optics-admin-secret"}};
        auto bad = cli.Post("/api/camera/optics", hdr, R"({"bogus":1})", "application/json");
        REQUIRE(bad != nullptr);
        REQUIRE_EQ(bad->status, 422);

        auto ok = cli.Post(
            "/api/camera/optics",
            hdr,
            R"({"lock_3a":true,"exposure":-8,"gain":16,"focus":38,"backlight_comp":0})",
            "application/json"
        );
        REQUIRE(ok != nullptr);
        REQUIRE_EQ(ok->status, 200);
        const json okj = json::parse(ok->body);
        REQUIRE(std::abs(okj.value("exposure", 0.0) - (-8.0)) < 1e-6);
        REQUIRE(std::abs(okj.value("gain", 0.0) - 16.0) < 1e-6);
        REQUIRE_EQ(okj.value("focus", -1), 38);

        auto cal = cli.Post("/api/camera/optics/auto_calibrate", hdr, "", "application/json");
        REQUIRE(cal != nullptr);
        REQUIRE_EQ(cal->status, 200);
        const json calj = json::parse(cal->body);
        REQUIRE_EQ(calj.value("triggered", false), true);
        REQUIRE_EQ(calj["optics"].value("auto_calibrating", false), true);

        auto diag = cli.Get("/api/diagnostics/runtime");
        REQUIRE(diag != nullptr);
        REQUIRE_EQ(diag->status, 200);
        const json dj = json::parse(diag->body);
        REQUIRE(dj.contains("camera_optics") && dj["camera_optics"].is_object());
        REQUIRE_EQ(dj["camera_optics"].value("auto_calibrating", false), true);
    }
    server.stop();

    // 8C. Closed-loop anti-glare stepper determinism (highlight-protected)
    {
        const auto blown = LiveCameraEngine::next_anti_glare_step(-6.0, 0.0, 245.0, 0.85, 255.0);
        REQUIRE(!blown.converged);
        REQUIRE(blown.next_exposure < -6.0);

        const auto clipped_mid = LiveCameraEngine::next_anti_glare_step(-5.0, 0.0, 110.0, 0.04, 248.0);
        REQUIRE(!clipped_mid.converged);
        REQUIRE(clipped_mid.next_exposure < -5.0);

        const auto dim_cap = LiveCameraEngine::next_anti_glare_step(-5.0, 0.0, 50.0, 0.0, 195.0);
        REQUIRE(dim_cap.converged);
        REQUIRE(std::abs(dim_cap.next_exposure - (-5.0)) < 1e-6);
        REQUIRE(std::abs(dim_cap.next_gain - 0.0) < 1e-6);

        const auto dim = LiveCameraEngine::next_anti_glare_step(-8.0, 0.0, 45.0, 0.0, 90.0);
        REQUIRE(!dim.converged);
        REQUIRE(dim.next_exposure > -8.0);

        const auto good = LiveCameraEngine::next_anti_glare_step(-6.0, 0.0, 115.0, 0.005, 195.0);
        REQUIRE(good.converged);
    }

    // 8D. Specular glare resistance: pure white glare strips must NOT spoof ESD glove or complete label,
    //     while measure_label_quality() continues flagging OVEREXPOSED accurately.
    LiveCameraEngine cam_spec(0);
    cam_spec.set_work_mode("debug");
    cv::Mat dark_bg(kDefaultCameraHeight, kDefaultCameraWidth, CV_8UC3, cv::Scalar(42, 46, 50));
    double ts = 20.0;
    for (int i = 0; i < 4; ++i) {
        ts += 0.04;
        REQUIRE(cam_spec.process_injected_frame(dark_bg, ts));
    }

    cv::Mat glare_scene = dark_bg.clone();
    // Dark empty tube with NO label + clipped pure-white BGR(255,255,255) specular glare strips
    cv::rectangle(
        glare_scene,
        cv::Rect(
            static_cast<int>(0.18 * kDefaultCameraWidth),
            static_cast<int>(0.44 * kDefaultCameraHeight),
            static_cast<int>(0.64 * kDefaultCameraWidth),
            static_cast<int>(0.12 * kDefaultCameraHeight)
        ),
        cv::Scalar(30, 120, 145),
        cv::FILLED
    );
    cv::rectangle(
        glare_scene,
        cv::Rect(
            static_cast<int>(0.43 * kDefaultCameraWidth),
            static_cast<int>(0.46 * kDefaultCameraHeight),
            static_cast<int>(0.14 * kDefaultCameraWidth),
            static_cast<int>(0.07 * kDefaultCameraHeight)
        ),
        cv::Scalar(255, 255, 255),
        cv::FILLED
    );
    cv::rectangle(
        glare_scene,
        cv::Rect(
            static_cast<int>(0.21 * kDefaultCameraWidth),
            static_cast<int>(0.42 * kDefaultCameraHeight),
            static_cast<int>(0.09 * kDefaultCameraWidth),
            static_cast<int>(0.24 * kDefaultCameraHeight)
        ),
        cv::Scalar(255, 255, 255),
        cv::FILLED
    );

    for (int i = 0; i < 3; ++i) {
        ts += 0.04;
        REQUIRE(cam_spec.process_injected_frame(glare_scene, ts));
    }
    const json st_spec = cam_spec.get_state_json();
    REQUIRE_EQ(st_spec.value("glove_compliant", true), false);
    REQUIRE(st_spec.value("label_status", "") != "complete");

    const auto q = measure_label_quality(glare_scene, Box{0.42, 0.44, 0.58, 0.56});
    REQUIRE(q.has_value());
    REQUIRE(q->overexposed_fraction > 0.08);
}

// ----------------------------------------------------------------------------
// Test 9: v4.3 Red Rabbit Mandatory Challenge, Anti-Cheat & SHA-256 Audit Chain
// ----------------------------------------------------------------------------
void test_red_rabbit_challenge_and_hash_chain_audit() {
    ScopedTempDir tmp;
    const auto web_dir = tmp.path() / "web_dist";
    std::filesystem::create_directories(web_dir);
    const std::string html = "<!doctype html><html><body>Red Rabbit OK</body></html>";
    REQUIRE(write_file_bytes_atomic(web_dir / "index.html", html.data(), html.size()));

    StationDatabase db(tmp.path());
    REQUIRE_EQ(db.rr_list_samples(true).size(), 4u);

    LiveCameraEngine cam(0);
    TrainingManager training(tmp.path());
    StationServerOptions opts;
    opts.host = "127.0.0.1";
    opts.port = 0;
    opts.data_dir = tmp.path();
    opts.web_dist_dir = web_dir;
    opts.enable_live_camera = false;

    std::tm base_tm{};
    base_tm.tm_year = 126;  // 2026
    base_tm.tm_mon = 10;    // Nov
    base_tm.tm_mday = 26;
    base_tm.tm_hour = 7;
    base_tm.tm_min = 50;
    base_tm.tm_sec = 0;
    base_tm.tm_isdst = -1;
    const int64_t t_0750_ms = static_cast<int64_t>(std::mktime(&base_tm)) * 1000;
    REQUIRE(t_0750_ms > 0);
    std::atomic<int64_t> fake_clock_ms{t_0750_ms};

    StationHttpServer server(db, cam, training, opts);
    server.red_rabbit().set_wall_clock_for_test([&]() { return fake_clock_ms.load(); });
    server.auth().set_env_override("IGBT_ENGINEER_PIN", "271828");
    const int port = server.start_background();
    REQUIRE(port > 0);

    httplib::Client cli("127.0.0.1", port);

    // 9A. Cold boot enters CHALLENGE_REQUIRED and marks /api/state as blocking
    auto st_cold = cli.Get("/api/state");
    REQUIRE(st_cold != nullptr && st_cold->status == 200);
    json st_cold_j = json::parse(st_cold->body);
    REQUIRE_EQ(st_cold_j.value("red_rabbit_blocking", false), true);
    REQUIRE(st_cold_j.contains("red_rabbit") && st_cold_j["red_rabbit"].is_object());
    REQUIRE_EQ(st_cold_j["red_rabbit"].value("state", ""), "CHALLENGE_REQUIRED");
    REQUIRE_EQ(st_cold_j["red_rabbit"].value("trigger_reason", ""), "COLD_BOOT");

    // Authenticate admin token & add an expired golden sample (must be excluded from plans)
    auto pin_res = cli.Post("/api/auth/verify", R"({"passcode":"271828"})", "application/json");
    REQUIRE(pin_res != nullptr && pin_res->status == 200);
    const std::string admin_tok = json::parse(pin_res->body).value("admin_token", "");
    httplib::Headers admin_hdrs = {{"X-Admin-Token", admin_tok}};

    json exp_payload = {
        {"sample_uid", "RR-NG-EXPIRED-999"},
        {"sample_type", "NG_OOD"},
        {"expected_verdict", "ALARM"},
        {"product_family", "IGBT_TO247"},
        {"is_active", true},
        {"expires_at_ms", t_0750_ms - 1000},
    };
    auto exp_res = cli.Post("/api/red_rabbit/samples", admin_hdrs, exp_payload.dump(), "application/json");
    REQUIRE(exp_res != nullptr && exp_res->status == 200);

    // 9B. Start randomized challenge (1 OK + 2 NG) & verify anti-cheat guards
    auto start_res = cli.Post(
        "/api/red_rabbit/sessions/start",
        R"({"operator_id":"OP_TEST","lot_id":"LOT_RR_01"})",
        "application/json"
    );
    REQUIRE(start_res != nullptr && start_res->status == 200);
    json rr = json::parse(start_res->body);
    REQUIRE(rr.contains("session") && rr["session"].is_object());
    const json& sess = rr["session"];
    const std::string sid = sess.value("session_id", "");
    REQUIRE(!sid.empty());
    const json& plan = sess["required_plan"];
    REQUIRE_EQ(plan.size(), 3u);

    int ok_cnt = 0;
    int ng_cnt = 0;
    for (const auto& step : plan) {
        REQUIRE(step.value("sample_uid", "") != "RR-NG-EXPIRED-999");
        if (step.value("sample_type", "") == "OK_GOLDEN") ++ok_cnt;
        else ++ng_cnt;
    }
    REQUIRE_EQ(ok_cnt, 1);
    REQUIRE_EQ(ng_cnt, 2);

    const std::string uid0 = plan[0].value("sample_uid", "");
    const std::string exp0 = plan[0].value("expected_verdict", "");
    const std::string uid1 = plan[1].value("sample_uid", "");
    const std::string exp1 = plan[1].value("expected_verdict", "");
    const std::string uid2 = plan[2].value("sample_uid", "");
    const std::string exp2 = plan[2].value("expected_verdict", "");

    // Wrong UID for step 0 rejected with 422
    auto bad_arm = cli.Post(
        "/api/red_rabbit/sessions/" + sid + "/arm",
        json{{"sample_uid", uid1}}.dump(),
        "application/json"
    );
    REQUIRE(bad_arm != nullptr && bad_arm->status == 422);

    // Arm & pass step 0
    auto arm0 = cli.Post(
        "/api/red_rabbit/sessions/" + sid + "/arm",
        json{{"sample_uid", uid0}}.dump(),
        "application/json"
    );
    REQUIRE(arm0 != nullptr && arm0->status == 200);
    auto att0 = cli.Post(
        "/api/red_rabbit/sessions/" + sid + "/attempt",
        json{{"sample_uid", uid0}, {"actual_verdict", exp0}}.dump(),
        "application/json"
    );
    REQUIRE(att0 != nullptr && att0->status == 200);
    REQUIRE_EQ(json::parse(att0->body)["session"].value("current_step_index", 0), 1);

    // Duplicate reuse of step 0 UID rejected with 409
    auto dup_arm = cli.Post(
        "/api/red_rabbit/sessions/" + sid + "/arm",
        json{{"sample_uid", uid0}}.dump(),
        "application/json"
    );
    REQUIRE(dup_arm != nullptr && dup_arm->status == 409);

    // 9C. Force CRITICAL_NG_MISSED_AS_PASS -> CHALLENGE_HARD_LOCK
    std::string fail_uid = uid1;
    if (exp1 == "PASS") {
        // Step 1 is OK_GOLDEN; pass it first so step 2 is guaranteed NG_ALARM
        REQUIRE(cli.Post("/api/red_rabbit/sessions/" + sid + "/arm", json{{"sample_uid", uid1}}.dump(), "application/json")->status == 200);
        REQUIRE(cli.Post("/api/red_rabbit/sessions/" + sid + "/attempt", json{{"sample_uid", uid1}, {"actual_verdict", "PASS"}}.dump(), "application/json")->status == 200);
        REQUIRE_EQ(exp2, "ALARM");
        fail_uid = uid2;
    }
    auto arm_ng = cli.Post(
        "/api/red_rabbit/sessions/" + sid + "/arm",
        json{{"sample_uid", fail_uid}}.dump(),
        "application/json"
    );
    REQUIRE(arm_ng != nullptr && arm_ng->status == 200);
    auto fail_ng = cli.Post(
        "/api/red_rabbit/sessions/" + sid + "/attempt",
        json{{"sample_uid", fail_uid}, {"actual_verdict", "PASS"}}.dump(),
        "application/json"
    );
    REQUIRE(fail_ng != nullptr && fail_ng->status == 200);
    json hard_j = json::parse(fail_ng->body);
    REQUIRE_EQ(hard_j.value("state", ""), "CHALLENGE_HARD_LOCK");
    REQUIRE_EQ(hard_j.value("hard_locked", false), true);
    REQUIRE_EQ(hard_j["last_lock_event"].value("reason_code", ""), "CRITICAL_NG_MISSED_AS_PASS");

    // Cannot start new challenge while hard-locked
    REQUIRE_EQ(cli.Post("/api/red_rabbit/sessions/start", "{}", "application/json")->status, 409);

    // Unlock requires admin token + >= 6 char comment, and returns to CHALLENGE_REQUIRED (never PROD_READY)
    REQUIRE_EQ(
        cli.Post(
            "/api/red_rabbit/unlock",
            R"({"actor_id":"QE_01","actor_role":"QE","reason_code":"OPTICS_RECHECKED","comment":"已复检偏振镜与金样，重新执行点检"})",
            "application/json"
        )->status,
        401
    );
    REQUIRE_EQ(
        cli.Post(
            "/api/red_rabbit/unlock",
            admin_hdrs,
            R"({"actor_id":"QE_01","actor_role":"QE","reason_code":"OPTICS_RECHECKED","comment":"ok"})",
            "application/json"
        )->status,
        422
    );
    auto unlock_res = cli.Post(
        "/api/red_rabbit/unlock",
        admin_hdrs,
        R"({"actor_id":"QE_01","actor_role":"QE","reason_code":"OPTICS_RECHECKED","comment":"已复检光源偏振与金样表面，退回重测"})",
        "application/json"
    );
    REQUIRE(unlock_res != nullptr && unlock_res->status == 200);
    REQUIRE_EQ(json::parse(unlock_res->body).value("state", ""), "CHALLENGE_REQUIRED");

    // Helper to execute a full passing 3-step challenge
    auto run_full_pass_challenge = [&]() {
        auto s_res = cli.Post(
            "/api/red_rabbit/sessions/start",
            R"({"operator_id":"OP_TEST","lot_id":"LOT_RR_01"})",
            "application/json"
        );
        REQUIRE(s_res != nullptr && s_res->status == 200);
        json sj = json::parse(s_res->body);
        const std::string cur_sid = sj["session"].value("session_id", "");
        const json& cur_plan = sj["session"]["required_plan"];
        REQUIRE_EQ(cur_plan.size(), 3u);
        json last_resp;
        for (const auto& step : cur_plan) {
            const std::string suid = step.value("sample_uid", "");
            const std::string sexp = step.value("expected_verdict", "");
            REQUIRE_EQ(
                cli.Post("/api/red_rabbit/sessions/" + cur_sid + "/arm", json{{"sample_uid", suid}}.dump(), "application/json")->status,
                200
            );
            auto a_res = cli.Post(
                "/api/red_rabbit/sessions/" + cur_sid + "/attempt",
                json{{"sample_uid", suid}, {"actual_verdict", sexp}}.dump(),
                "application/json"
            );
            REQUIRE(a_res != nullptr && a_res->status == 200);
            last_resp = json::parse(a_res->body);
        }
        REQUIRE_EQ(last_resp.value("state", ""), "PROD_READY");
        REQUIRE_EQ(last_resp.value("blocking", true), false);
    };

    // 9D. Complete challenge cleanly at 07:50 -> PROD_READY
    run_full_pass_challenge();
    auto st_ready = cli.Get("/api/state");
    REQUIRE(st_ready != nullptr && st_ready->status == 200);
    REQUIRE_EQ(json::parse(st_ready->body).value("red_rabbit_blocking", true), false);

    // 9E. Shift change trigger (08:06 local > 08:00 + 5min grace)
    fake_clock_ms.store(t_0750_ms + 16LL * 60 * 1000);
    auto st_shift = cli.Get("/api/red_rabbit/status");
    REQUIRE(st_shift != nullptr && st_shift->status == 200);
    json shift_j = json::parse(st_shift->body);
    REQUIRE_EQ(shift_j.value("state", ""), "CHALLENGE_REQUIRED");
    REQUIRE_EQ(shift_j.value("trigger_reason", ""), "SHIFT_CHANGE");
    run_full_pass_challenge();

    // 9F. 4h interval expiration trigger
    fake_clock_ms.fetch_add(4LL * 3600 * 1000 + 1000);
    auto st_int = cli.Get("/api/red_rabbit/status");
    REQUIRE(st_int != nullptr && st_int->status == 200);
    json int_j = json::parse(st_int->body);
    REQUIRE_EQ(int_j.value("state", ""), "CHALLENGE_REQUIRED");
    REQUIRE_EQ(int_j.value("trigger_reason", ""), "INTERVAL_EXPIRED");
    run_full_pass_challenge();

    // 9G. Optics 3A update triggers MODEL_OR_OPTICS_CHANGE
    auto opt_res = cli.Post(
        "/api/camera/optics",
        admin_hdrs,
        R"({"lock_3a":true,"exposure":-6.0,"gain":0.0,"focus":35,"wb_temperature":4600,"backlight_comp":0})",
        "application/json"
    );
    REQUIRE(opt_res != nullptr && opt_res->status == 200);
    auto st_opt = cli.Get("/api/red_rabbit/status");
    REQUIRE(st_opt != nullptr && st_opt->status == 200);
    REQUIRE_EQ(json::parse(st_opt->body).value("trigger_reason", ""), "MODEL_OR_OPTICS_CHANGE");

    // 9H. Session timeout (> 10 min) transitions to CHALLENGE_HARD_LOCK
    auto start_to = cli.Post("/api/red_rabbit/sessions/start", "{}", "application/json");
    REQUIRE(start_to != nullptr && start_to->status == 200);
    fake_clock_ms.fetch_add(11LL * 60 * 1000);
    auto st_to = cli.Get("/api/red_rabbit/status");
    REQUIRE(st_to != nullptr && st_to->status == 200);
    json to_j = json::parse(st_to->body);
    REQUIRE_EQ(to_j.value("state", ""), "CHALLENGE_HARD_LOCK");
    REQUIRE_EQ(to_j.value("trigger_reason", ""), "SESSION_TIMEOUT");

    // 9I. SHA-256 hash-chain verification & SQLite tampering detection
    auto aud_res = cli.Get("/api/red_rabbit/audit");
    REQUIRE(aud_res != nullptr && aud_res->status == 200);
    json aud_j = json::parse(aud_res->body);
    REQUIRE_EQ(aud_j["chain_verification"].value("valid", false), true);
    REQUIRE(aud_j["chain_verification"].value("checked_rows", 0) >= 3);

    server.stop();

    sqlite3* raw_db = nullptr;
    REQUIRE_EQ(sqlite3_open_v2(path_to_utf8(db.db_path()).c_str(), &raw_db, SQLITE_OPEN_READWRITE, nullptr), SQLITE_OK);
    char* errmsg = nullptr;
    int rc = sqlite3_exec(raw_db, "UPDATE station_lock_audit SET comment='TAMPERED' WHERE audit_seq=1;", nullptr, nullptr, &errmsg);
    if (errmsg) sqlite3_free(errmsg);
    sqlite3_close_v2(raw_db);
    REQUIRE_EQ(rc, SQLITE_OK);

    const json broken = db.rr_verify_lock_audit_chain();
    REQUIRE_EQ(broken.value("valid", true), false);
    REQUIRE_EQ(broken.value("broken_seq", int64_t{-1}), int64_t{1});

    // 9J. Crash dump presence triggers CRASH_RECOVERY on server initialization
    ScopedTempDir tmp_crash;
    const auto dumps_dir = tmp_crash.path() / "dumps";
    std::filesystem::create_directories(dumps_dir);
    const std::string fake_dmp = "MDMP_TEST";
    REQUIRE(write_file_bytes_atomic(dumps_dir / "crash_999.dmp", fake_dmp.data(), fake_dmp.size()));
    StationDatabase db_crash(tmp_crash.path());
    LiveCameraEngine cam_crash(0);
    TrainingManager tr_crash(tmp_crash.path());
    StationServerOptions opts_crash;
    opts_crash.data_dir = tmp_crash.path();
    opts_crash.web_dist_dir = web_dir;
    StationHttpServer srv_crash(db_crash, cam_crash, tr_crash, opts_crash);
    REQUIRE_EQ(srv_crash.red_rabbit().status_json().value("trigger_reason", ""), "CRASH_RECOVERY");
}

// ----------------------------------------------------------------------------
// Test 10: v4.3 ADAM-6050/6060 Native Winsock2 Modbus TCP Watchdog & Fail-Safe
// ----------------------------------------------------------------------------
class MockModbusTcpServer {
public:
    MockModbusTcpServer() {
        WSADATA wsa{};
        (void)WSAStartup(MAKEWORD(2, 2), &wsa);

        listen_sock_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        REQUIRE(listen_sock_ != INVALID_SOCKET);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        REQUIRE_EQ(::bind(listen_sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
        REQUIRE_EQ(::listen(listen_sock_, 4), 0);

        sockaddr_in bound{};
        int bound_len = sizeof(bound);
        REQUIRE_EQ(::getsockname(listen_sock_, reinterpret_cast<sockaddr*>(&bound), &bound_len), 0);
        port_ = ntohs(bound.sin_port);

        u_long nonblock = 1;
        ioctlsocket(listen_sock_, FIONBIO, &nonblock);
        worker_ = std::thread([this]() { serve_loop(); });
    }

    ~MockModbusTcpServer() {
        stop();
    }

    uint16_t port() const noexcept { return port_; }

    void stop() {
        stop_requested_.store(true);
        if (listen_sock_ != INVALID_SOCKET) {
            closesocket(listen_sock_);
            listen_sock_ = INVALID_SOCKET;
        }
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::atomic<uint16_t> di_mask{0x0009};  // bit 0 (active-low E-Stop OK) + bit 3 (ESD ground OK)
    std::atomic<bool> drop_responses{false};
    std::atomic<bool> corrupt_tx_id{false};
    std::atomic<uint8_t> latest_do_mask{0};
    std::atomic<int> fc15_count{0};
    std::atomic<int> fc02_count{0};
    std::atomic<int> seen_toggle_0{0};
    std::atomic<int> seen_toggle_1{0};

private:
    static bool recv_n(SOCKET s, uint8_t* dst, std::size_t n, const std::atomic<bool>& stop_flag) {
        std::size_t got = 0;
        while (got < n && !stop_flag.load()) {
            WSAPOLLFD pfd{};
            pfd.fd = s;
            pfd.events = POLLRDNORM;
            const int prc = WSAPoll(&pfd, 1, 20);
            if (prc < 0) return false;
            if (prc == 0) continue;
            if (pfd.revents & (POLLERR | POLLNVAL)) return false;
            const int r = ::recv(s, reinterpret_cast<char*>(dst + got), static_cast<int>(n - got), 0);
            if (r <= 0) return false;
            got += static_cast<std::size_t>(r);
        }
        return got == n;
    }

    void handle_client(SOCKET cli) {
        u_long nb = 1;
        ioctlsocket(cli, FIONBIO, &nb);
        while (!stop_requested_.load()) {
            uint8_t mbap[7]{};
            if (!recv_n(cli, mbap, 7, stop_requested_)) break;
            const uint16_t rx_tx = static_cast<uint16_t>((mbap[0] << 8) | mbap[1]);
            const uint16_t pdu_len = static_cast<uint16_t>((mbap[4] << 8) | mbap[5]);
            const uint8_t unit_id = mbap[6];
            if (pdu_len < 2 || pdu_len > 64) break;
            std::vector<uint8_t> pdu(pdu_len - 1);
            if (!recv_n(cli, pdu.data(), pdu.size(), stop_requested_)) break;

            if (drop_responses.load()) {
                continue;
            }

            const uint16_t tx_out = corrupt_tx_id.load() ? static_cast<uint16_t>(rx_tx ^ 0x8000U) : rx_tx;
            if (pdu[0] == 0x0F && pdu.size() >= 7) {
                const uint8_t mask = pdu[6];
                latest_do_mask.store(mask);
                ++fc15_count;
                if ((mask & (1U << 2)) != 0) ++seen_toggle_1;
                else ++seen_toggle_0;

                uint8_t resp[12]{
                    static_cast<uint8_t>(tx_out >> 8), static_cast<uint8_t>(tx_out & 0xFF),
                    0x00, 0x00, 0x00, 0x06, unit_id, 0x0F,
                    pdu[1], pdu[2], pdu[3], pdu[4]
                };
                (void)::send(cli, reinterpret_cast<const char*>(resp), sizeof(resp), 0);
            } else if (pdu[0] == 0x02 && pdu.size() >= 5) {
                ++fc02_count;
                const uint16_t cnt = static_cast<uint16_t>((pdu[3] << 8) | pdu[4]);
                const uint8_t byte_cnt = static_cast<uint8_t>((cnt + 7) / 8);
                const uint16_t cur_di = di_mask.load();
                std::vector<uint8_t> resp(9 + byte_cnt, 0);
                resp[0] = static_cast<uint8_t>(tx_out >> 8);
                resp[1] = static_cast<uint8_t>(tx_out & 0xFF);
                resp[2] = 0x00;
                resp[3] = 0x00;
                const uint16_t rlen = static_cast<uint16_t>(3 + byte_cnt);
                resp[4] = static_cast<uint8_t>(rlen >> 8);
                resp[5] = static_cast<uint8_t>(rlen & 0xFF);
                resp[6] = unit_id;
                resp[7] = 0x02;
                resp[8] = byte_cnt;
                for (uint8_t b = 0; b < byte_cnt; ++b) {
                    resp[9 + b] = static_cast<uint8_t>((cur_di >> (b * 8)) & 0xFF);
                }
                (void)::send(cli, reinterpret_cast<const char*>(resp.data()), static_cast<int>(resp.size()), 0);
            }
        }
        closesocket(cli);
    }

    void serve_loop() {
        while (!stop_requested_.load()) {
            WSAPOLLFD pfd{};
            pfd.fd = listen_sock_;
            pfd.events = POLLRDNORM;
            const int prc = WSAPoll(&pfd, 1, 20);
            if (prc <= 0) continue;
            SOCKET cli = ::accept(listen_sock_, nullptr, nullptr);
            if (cli != INVALID_SOCKET) {
                handle_client(cli);
            }
        }
    }

    SOCKET listen_sock_{INVALID_SOCKET};
    uint16_t port_{0};
    std::atomic<bool> stop_requested_{false};
    std::thread worker_;
};

template <typename Pred>
bool wait_until_true(Pred pred, int timeout_ms = 2500) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    return pred();
}

void test_adam_modbus_dio_watchdog_and_failsafe() {
    MockModbusTcpServer mock;

    auto steady_ms = []() -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    };

    AdamModbusConfig cfg;
    cfg.enabled = true;
    cfg.host = "127.0.0.1";
    cfg.port = mock.port();
    cfg.unit_id = 1;
    cfg.heartbeat_period_ms = 35;
    cfg.connect_timeout_ms = 200;
    cfg.io_timeout_ms = 100;
    cfg.max_consecutive_timeouts = 2;
    cfg.vision_stall_timeout_ms = 220;
    cfg.enable_estop_di_check = true;
    cfg.enable_relay_aux_check = true;
    cfg.enable_esd_di_check = true;

    std::atomic<bool> fake_vision_running{true};
    std::atomic<bool> freeze_vision_tick{false};
    std::atomic<int64_t> frozen_tick_ms{0};

    AdamModbusDioDriver drv(
        cfg,
        [&]() { return fake_vision_running.load(); },
        [&]() { return freeze_vision_tick.load() ? frozen_tick_ms.load() : steady_ms(); }
    );
    drv.start();

    // 10A. Normal FC15+FC02 heartbeat loop, toggle bit flip, PASS_ALLOW & LOCKED
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return st.connected && st.hardware_ready && !st.fault_locked &&
               st.total_heartbeats >= 4 &&
               mock.seen_toggle_0.load() > 0 && mock.seen_toggle_1.load() > 0;
    }));
    REQUIRE_EQ(drv.is_esd_ground_ok(), true);
    REQUIRE((mock.latest_do_mask.load() & (1U << 0)) != 0);  // system healthy bit
    REQUIRE((mock.latest_do_mask.load() & (1U << 1)) == 0);  // pass allow initially 0

    REQUIRE(drv.set_interlock(InterlockSignal::PASS_ALLOW));
    REQUIRE(wait_until_true([&]() { return (mock.latest_do_mask.load() & (1U << 1)) != 0; }));
    REQUIRE(drv.set_interlock(InterlockSignal::LOCKED));
    REQUIRE(wait_until_true([&]() { return (mock.latest_do_mask.load() & (1U << 1)) == 0; }));

    // 10B. Active-low E-Stop fault drops DO to 0x00 and recovers on restore
    mock.di_mask.store(0x0008);  // clear bit 0 (E-Stop active)
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return st.fault_locked && st.fault_reason == "HARDWARE_ESTOP_ACTIVE" &&
               mock.latest_do_mask.load() == 0x00;
    }));
    REQUIRE(!drv.is_hardware_ready());
    mock.di_mask.store(0x0009);  // restore bit 0
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return !st.fault_locked && drv.is_hardware_ready();
    }));

    // 10C. Safety relay auxiliary contact weld latches until engineer acknowledge
    mock.di_mask.store(0x000D);  // bit 0 + bit 2 (relay_aux closed while pass_allow=0) + bit 3
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return st.relay_weld_fault && st.fault_locked &&
               st.fault_reason == "RELAY_CONTACT_WELD_FAULT";
    }));
    mock.di_mask.store(0x0009);  // clear aux bit -> fault MUST remain latched
    std::this_thread::sleep_for(std::chrono::milliseconds(90));
    REQUIRE_EQ(drv.health_status().fault_locked, true);
    REQUIRE(drv.acknowledge_relay_fault());
    REQUIRE(wait_until_true([&]() { return !drv.health_status().fault_locked && drv.is_hardware_ready(); }));

    // 10D. Vision loop soft deadlock -> immediate safe-zero DO and stops feeding watchdog
    frozen_tick_ms.store(steady_ms() - 1000);
    freeze_vision_tick.store(true);
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return st.vision_stalled && st.fault_locked &&
               st.fault_reason == "VISION_LOOP_STALL_WATCHDOG_STOP" &&
               mock.latest_do_mask.load() == 0x00;
    }));
    const int fc15_at_stall = mock.fc15_count.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(160));
    REQUIRE_EQ(mock.fc15_count.load(), fc15_at_stall);  // must NOT keep feeding ADAM WDT!

    freeze_vision_tick.store(false);
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return !st.vision_stalled && !st.fault_locked && drv.is_hardware_ready();
    }));

    // 10E. Corrupt MBAP transaction ID & comm timeout -> MODBUS_WATCHDOG_TIMEOUT -> recovery -> clean stop
    mock.corrupt_tx_id.store(true);
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return st.total_comm_errors >= 2 && st.fault_locked &&
               st.fault_reason == "MODBUS_WATCHDOG_TIMEOUT";
    }));
    mock.corrupt_tx_id.store(false);
    REQUIRE(wait_until_true([&]() { return !drv.health_status().fault_locked && drv.is_hardware_ready(); }));

    mock.drop_responses.store(true);
    REQUIRE(wait_until_true([&]() {
        const auto st = drv.health_status();
        return st.fault_locked && st.fault_reason == "MODBUS_WATCHDOG_TIMEOUT";
    }));
    mock.drop_responses.store(false);
    REQUIRE(wait_until_true([&]() { return !drv.health_status().fault_locked && drv.is_hardware_ready(); }));

    drv.stop();
    REQUIRE_EQ(mock.latest_do_mask.load(), uint8_t{0x00});

    // 10F. StationHttpServer + HAL + Audit Chain + LiveCameraEngine tick integration
    ScopedTempDir tmp;
    const auto web_dir = tmp.path() / "web_dist";
    std::filesystem::create_directories(web_dir);
    const std::string html = "<!doctype html><html><body>HAL OK</body></html>";
    REQUIRE(write_file_bytes_atomic(web_dir / "index.html", html.data(), html.size()));

    StationDatabase db(tmp.path());
    LiveCameraEngine cam(0);
    cv::Mat synth = make_synthetic_station_frame(1280, 720, "complete", false, false);
    REQUIRE(cam.process_injected_frame(synth, 100.0));
    REQUIRE(cam.last_vision_tick_ms() > 0);

    TrainingManager tr(tmp.path());
    StationServerOptions opts;
    opts.host = "127.0.0.1";
    opts.port = 0;
    opts.data_dir = tmp.path();
    opts.web_dist_dir = web_dir;
    opts.enable_live_camera = false;
    opts.adam_config = cfg;

    StationHttpServer server(db, cam, tr, opts);
    server.auth().set_env_override("IGBT_ENGINEER_PIN", "112233");
    const int http_port = server.start_background();
    REQUIRE(http_port > 0);

    httplib::Client cli("127.0.0.1", http_port);
    REQUIRE(wait_until_true([&]() {
        auto h = cli.Get("/api/health");
        if (!h || h->status != 200) return false;
        json hj = json::parse(h->body);
        return hj.value("interlock_backend", "") == "adam_modbus_tcp" &&
               hj.value("interlock_hardware_ready", false) &&
               hj.value("mode", "") == "hardware_interlock";
    }));

    mock.drop_responses.store(true);
    REQUIRE(wait_until_true([&]() {
        auto s = cli.Get("/api/state");
        if (!s || s->status != 200) return false;
        json sj = json::parse(s->body);
        return sj.value("hardware_fault_locked", false) &&
               sj.value("outcome", "") == "FAULT" &&
               sj.value("fsm_state", "") == "FAULT" &&
               sj.value("result_reason", "") == "MODBUS_WATCHDOG_TIMEOUT";
    }));

    auto aud_res = cli.Get("/api/red_rabbit/audit");
    REQUIRE(aud_res != nullptr && aud_res->status == 200);
    json aud_j = json::parse(aud_res->body);
    REQUIRE_EQ(aud_j["chain_verification"].value("valid", false), true);
    bool found_wdt_audit = false;
    for (const auto& row : aud_j["lock_audit"]) {
        if (row.value("event_type", "") == "WATCHDOG_FAULT_LOCK" &&
            row.value("reason_code", "") == "MODBUS_WATCHDOG_TIMEOUT") {
            found_wdt_audit = true;
            break;
        }
    }
    REQUIRE(found_wdt_audit);

    mock.drop_responses.store(false);
    REQUIRE(wait_until_true([&]() {
        auto s = cli.Get("/api/state");
        return s && s->status == 200 && !json::parse(s->body).value("hardware_fault_locked", true);
    }));

    // Relay weld fault reset route requires admin token & writes audit trail
    mock.di_mask.store(0x000D);
    REQUIRE(wait_until_true([&]() {
        auto d = cli.Get("/api/hal/dio");
        return d && d->status == 200 && json::parse(d->body).value("relay_weld_fault", false);
    }));
    REQUIRE_EQ(cli.Post("/api/hal/dio/reset_weld_fault", "", "application/json")->status, 401);

    auto pin_res = cli.Post("/api/auth/verify", R"({"passcode":"112233"})", "application/json");
    REQUIRE(pin_res != nullptr && pin_res->status == 200);
    httplib::Headers admin_hdrs = {{"X-Admin-Token", json::parse(pin_res->body).value("admin_token", "")}};
    mock.di_mask.store(0x0009);
    auto rst = cli.Post("/api/hal/dio/reset_weld_fault", admin_hdrs, "", "application/json");
    REQUIRE(rst != nullptr && rst->status == 200);
    REQUIRE_EQ(json::parse(rst->body).value("reset_applied", false), true);
    REQUIRE(wait_until_true([&]() {
        auto d = cli.Get("/api/hal/dio");
        return d && d->status == 200 && !json::parse(d->body).value("fault_locked", true);
    }));

    server.stop();
    mock.stop();
}

// ----------------------------------------------------------------------------
// Test 11: v4.4 P0-3 RANSAC Tube Alignment, TUBE_ALIGN_FAIL, NanoDet GFL & 2/10 Hysteresis
// ----------------------------------------------------------------------------
void test_v44_ransac_alignment_nanodet_and_hysteresis() {
    ScopedTempDir tmp;

    auto make_tilted_tube_scene = [](double angle_deg, double tube_h_frac = 0.125, bool glare_midband = false) {
        const int W = kDefaultCameraWidth;
        const int H = kDefaultCameraHeight;
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(46, 50, 54));
        const double rad = angle_deg * 3.14159265358979323846 / 180.0;
        const cv::Point2d center(W * 0.50, H * 0.50);
        const cv::Point2d u(std::cos(rad), std::sin(rad));
        const cv::Point2d n(-u.y, u.x);
        const double half_l = W * 0.34;
        const double half_w = H * tube_h_frac * 0.5;

        auto to_pt = [&](double s, double t) {
            return cv::Point(
                static_cast<int>(std::llround(center.x + s * u.x + t * n.x)),
                static_cast<int>(std::llround(center.y + s * u.y + t * n.y))
            );
        };
        const std::vector<cv::Point> tube_poly = {
            to_pt(-half_l, -half_w), to_pt(half_l, -half_w),
            to_pt(half_l, half_w), to_pt(-half_l, half_w)
        };
        cv::fillConvexPoly(img, tube_poly, cv::Scalar(35, 195, 235));
        cv::polylines(img, tube_poly, true, cv::Scalar(20, 125, 170), 3, cv::LINE_AA);

        const double lbl_hl = W * 0.07;
        const double lbl_hw = half_w * 0.72;
        const std::vector<cv::Point> lbl_poly = {
            to_pt(-lbl_hl, -lbl_hw), to_pt(lbl_hl, -lbl_hw),
            to_pt(lbl_hl, lbl_hw), to_pt(-lbl_hl, lbl_hw)
        };
        cv::fillConvexPoly(img, lbl_poly, cv::Scalar(212, 215, 218));
        cv::polylines(img, lbl_poly, true, cv::Scalar(60, 62, 65), 2, cv::LINE_AA);
        for (double s = -lbl_hl + 12.0; s <= lbl_hl - 12.0; s += 10.0) {
            cv::line(img, to_pt(s, -lbl_hw + 8.0), to_pt(s, lbl_hw - 8.0), cv::Scalar(55, 58, 60), 2, cv::LINE_AA);
        }

        // Occluding white ESD gloves gripping both tube ends
        cv::rectangle(img, cv::Rect(static_cast<int>(0.15 * W), static_cast<int>(0.34 * H),
                                    static_cast<int>(0.13 * W), static_cast<int>(0.36 * H)),
                      cv::Scalar(235, 238, 240), cv::FILLED);
        cv::rectangle(img, cv::Rect(static_cast<int>(0.72 * W), static_cast<int>(0.34 * H),
                                    static_cast<int>(0.13 * W), static_cast<int>(0.36 * H)),
                      cv::Scalar(235, 238, 240), cv::FILLED);

        if (glare_midband) {
            cv::rectangle(img, cv::Rect(static_cast<int>(0.29 * W), static_cast<int>(0.38 * H),
                                        static_cast<int>(0.42 * W), static_cast<int>(0.24 * H)),
                          cv::Scalar(255, 255, 255), cv::FILLED);
        }
        return img;
    };

    // 11A. Horizontal (0 deg, affine path) & 10 deg tilted (perspective path) with end-glove occlusion
    {
        vision::TubeGeometryAligner aligner;
        const Box tb{0.16, 0.42, 0.84, 0.58};
        const Box lb{0.43, 0.44, 0.57, 0.56};
        const std::vector<Box> hands{{0.15, 0.34, 0.28, 0.70}, {0.72, 0.34, 0.85, 0.70}};

        const cv::Mat f0 = make_tilted_tube_scene(0.0, 0.125, false);
        const auto r0 = aligner.align(f0, tb, hands, lb, 0.125);
        REQUIRE_EQ(vision::tube_align_status_to_string(r0.status), "ALIGNED");
        REQUIRE(r0.geometry_ok);
        REQUIRE(!r0.used_perspective);
        REQUIRE(r0.condition_number <= 3.5);
        REQUIRE(r0.det_2x2 >= 0.25 && r0.det_2x2 <= 4.0);
        REQUIRE(r0.parallel_diff_deg <= 3.0);
        REQUIRE_EQ(r0.rectified_tube_bgr.size(), cv::Size(256, 64));
        REQUIRE_EQ(r0.rectified_label_bgr.size(), cv::Size(160, 96));

        vision::TubeGeometryAligner aligner_tilt;
        const Box tb_tilt{0.15, 0.32, 0.85, 0.68};
        const cv::Mat f10 = make_tilted_tube_scene(10.0, 0.125, false);
        const auto r10 = aligner_tilt.align(f10, tb_tilt, hands, lb, 0.125);
        REQUIRE_EQ(vision::tube_align_status_to_string(r10.status), "ALIGNED");
        REQUIRE(r10.geometry_ok);
        REQUIRE(r10.used_perspective);
        REQUIRE(std::abs(r10.angle_deg - 10.0) < 2.5);
        REQUIRE(r10.condition_number <= 3.5);
        REQUIRE(r10.parallel_diff_deg <= 3.0);
        REQUIRE_EQ(r10.rectified_tube_bgr.size(), cv::Size(256, 64));
        REQUIRE_EQ(r10.rectified_label_bgr.size(), cv::Size(160, 96));
    }

    // 11B. Homography gate SVD / det boundary assertions
    {
        double cond = 0.0, det = 0.0;
        cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);
        REQUIRE(vision::TubeGeometryAligner::check_homography_gate(eye, 3.5, 0.25, 4.0, cond, det));
        REQUIRE(cond < 1.05 && std::abs(det - 1.0) < 1e-3);

        cv::Mat singular = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        REQUIRE(!vision::TubeGeometryAligner::check_homography_gate(singular, 3.5, 0.25, 4.0, cond, det));

        cv::Mat high_shear = (cv::Mat_<double>(3, 3) << 1.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        REQUIRE(!vision::TubeGeometryAligner::check_homography_gate(high_shear, 3.5, 0.25, 4.0, cond, det));
        REQUIRE(cond > 3.5);

        cv::Mat tiny_det = (cv::Mat_<double>(3, 3) << 0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0);
        REQUIRE(!vision::TubeGeometryAligner::check_homography_gate(tiny_det, 3.5, 0.25, 4.0, cond, det));
    }

    // 11C. REUSED_LOW_CONF (<=2 failures) -> TUBE_ALIGN_FAIL (>=3 failures) & NO unrectified fallback
    {
        json cfg = StationDatabase::default_config();
        cfg["roi_calibrated"] = true;
        cfg["roi_config_version"] = "ROI_V001";
        cfg["config_version"] = 1;
        cfg["calibrated_at_utc"] = utc_now_iso();
        cfg["rois"] = json{
            {"tube_roi", {0.16, 0.42, 0.84, 0.58}},
            {"label_roi", {0.43, 0.45, 0.57, 0.55}},
            {"label_contact_roi", {0.40, 0.38, 0.60, 0.62}},
            {"left_grip_zone", {0.16, 0.34, 0.31, 0.82}},
            {"right_grip_zone", {0.69, 0.34, 0.84, 0.82}}
        };
        cfg["camera_width"] = 1280;
        cfg["camera_height"] = 720;
        cfg["thresholds"]["expected_tube_width_norm"] = 0.14;

        LiveCameraEngine cam(0, [cfg]() { return cfg; });
        cam.update_config(cfg);
        const cv::Mat empty_bg(kDefaultCameraHeight, kDefaultCameraWidth, CV_8UC3, cv::Scalar(48, 52, 56));
        double ts = monotonic_seconds();
        for (int i = 0; i < 3; ++i) {
            ts += 0.04;
            REQUIRE(cam.process_injected_frame(empty_bg, ts));
        }

        const cv::Mat good_frame = make_synthetic_station_frame(1280, 720, "complete", false, false);
        for (int i = 0; i < 3; ++i) {
            ts += 0.04;
            REQUIRE(cam.process_injected_frame(good_frame, ts));
        }
        {
            const json st_ok = cam.get_state_json();
            REQUIRE_EQ(st_ok.value("tube_align_failed", true), false);
            REQUIRE_EQ(st_ok.at("tube_alignment").value("status", ""), "ALIGNED");
            REQUIRE_EQ(st_ok.value("label_status", ""), "complete");
        }

        // Blind the midband with clipped specular glare while keeping the outer tube visible
        cv::Mat blinded = good_frame.clone();
        cv::rectangle(
            blinded,
            cv::Rect(static_cast<int>(0.28 * 1280), static_cast<int>(0.38 * 720),
                     static_cast<int>(0.44 * 1280), static_cast<int>(0.24 * 720)),
            cv::Scalar(255, 255, 255),
            cv::FILLED
        );

        ts += 0.04;
        REQUIRE(cam.process_injected_frame(blinded, ts));
        {
            const json st1 = cam.get_state_json();
            REQUIRE_EQ(st1.at("tube_alignment").value("status", ""), "REUSED_LOW_CONF");
            REQUIRE_EQ(st1.value("tube_align_failed", true), false);
            REQUIRE_EQ(st1.value("view_valid", true), false);
            REQUIRE_EQ(st1.value("c_timer_ms", -1), 0);
        }

        ts += 0.04;
        REQUIRE(cam.process_injected_frame(blinded, ts));
        REQUIRE_EQ(cam.get_state_json().at("tube_alignment").value("status", ""), "REUSED_LOW_CONF");
        REQUIRE_EQ(cam.get_state_json().value("view_valid", true), false);

        ts += 0.04;
        REQUIRE(cam.process_injected_frame(blinded, ts));
        {
            const json st3 = cam.get_state_json();
            REQUIRE_EQ(st3.at("tube_alignment").value("status", ""), "TUBE_ALIGN_FAIL");
            REQUIRE_EQ(st3.value("tube_align_failed", false), true);
            REQUIRE_EQ(st3.value("view_valid", true), false);
            REQUIRE_EQ(st3.value("label_status", ""), "unknown");
            REQUIRE(st3.at("label_confidence").is_null());
            REQUIRE_EQ(st3.value("result_reason", ""), "TUBE_ALIGN_FAIL");
        }
    }

    // 11D. Width deviation > 12% & interframe angle jump > 15 deg rejection
    {
        vision::TubeGeometryAligner aligner;
        const Box tb{0.16, 0.40, 0.84, 0.60};
        const std::vector<Box> hands{{0.15, 0.34, 0.28, 0.70}, {0.72, 0.34, 0.85, 0.70}};

        const cv::Mat wide_tube = make_tilted_tube_scene(0.0, 0.20, false);  // 0.20 vs 0.125 prior (+60%)
        const auto rw = aligner.align(wide_tube, tb, hands, std::nullopt, 0.125);
        REQUIRE(!rw.geometry_ok);
        REQUIRE_EQ(rw.fail_reason, "WIDTH_DEV_EXCEEDED");

        aligner.reset();
        const cv::Mat f0 = make_tilted_tube_scene(0.0, 0.125, false);
        REQUIRE(aligner.align(f0, Box{0.16, 0.42, 0.84, 0.58}, hands, std::nullopt, 0.125).geometry_ok);
        const cv::Mat f20 = make_tilted_tube_scene(20.0, 0.125, false);
        const auto rj = aligner.align(f20, Box{0.14, 0.26, 0.86, 0.74}, hands, std::nullopt, 0.125);
        REQUIRE(!rj.geometry_ok);
        REQUIRE_EQ(rj.fail_reason, "ANGLE_JUMP_EXCEEDED");
    }

    // 11E. NanoDet-Plus-m GFL output decoding ([1, 2125, 36] & [1, 2100, 36]), Letterbox, NMS & thresholds
    {
        vision::HandGloveConfig hcfg;
        const int orig_w = 1280, orig_h = 720, sz = 320;
        const float lb_scale = std::min(static_cast<float>(sz) / orig_w, static_cast<float>(sz) / orig_h);
        const int pad_x = (sz - static_cast<int>(std::round(orig_w * lb_scale))) / 2;
        const int pad_y = (sz - static_cast<int>(std::round(orig_h * lb_scale))) / 2;

        auto set_gfl_cell = [&](cv::Mat& out, int row_idx, int cls, float score,
                                float dl_bins, float dt_bins, float dr_bins, float db_bins) {
            out.at<float>(0, row_idx, cls) = score;
            const std::array<float, 4> d_bins{dl_bins, dt_bins, dr_bins, db_bins};
            for (int e = 0; e < 4; ++e) {
                const int best_k = std::clamp(static_cast<int>(std::llround(d_bins[e])), 0, 7);
                for (int k = 0; k < 8; ++k) {
                    out.at<float>(0, row_idx, 4 + e * 8 + k) = (k == best_k) ? 10.0f : -10.0f;
                }
            }
        };

        // 4-stride [1, 2125, 36]
        const int dims4[3] = {1, 2125, 36};
        cv::Mat raw4 = cv::Mat::zeros(3, dims4, CV_32F);
        // Stride-32 starts at 1600 + 400 = 2000 (grid 10x10, cell (5, 2) -> cx=64, cy=160)
        set_gfl_cell(raw4, 2000 + 5 * 10 + 2, 1, 0.90f, 1.0f, 1.0f, 1.0f, 1.0f);  // esd_white left
        // Duplicate overlapping esd_white on stride-16 cell (10, 4) -> cx=64, cy=160 (grid 20x20 starts at 1600)
        set_gfl_cell(raw4, 1600 + 10 * 20 + 4, 1, 0.72f, 2.0f, 2.0f, 2.0f, 2.0f);
        // Valid bare_hand on stride-32 cell (5, 5) -> cx=160, cy=160 (in work zone, score 0.85 >= 0.60)
        set_gfl_cell(raw4, 2000 + 5 * 10 + 5, 0, 0.85f, 1.0f, 1.0f, 1.0f, 1.0f);
        // Below-threshold bare_hand (0.55 < 0.60) MUST be filtered out
        set_gfl_cell(raw4, 2000 + 6 * 10 + 8, 0, 0.55f, 1.0f, 1.0f, 1.0f, 1.0f);
        // Out-of-work-zone bare_hand (cy near top pad_y + 10 -> cy_norm < 0.22) MUST be filtered out
        set_gfl_cell(raw4, 2000 + 2 * 10 + 5, 0, 0.92f, 0.5f, 0.5f, 0.5f, 0.5f);
        // esd_gray & esd_black
        set_gfl_cell(raw4, 2000 + 5 * 10 + 8, 2, 0.82f, 1.0f, 1.0f, 1.0f, 1.0f);
        set_gfl_cell(raw4, 2000 + 6 * 10 + 7, 3, 0.78f, 1.0f, 1.0f, 1.0f, 1.0f);

        const auto dets4 = vision::HandGloveDetector::decode_nanodet_plus_output(
            raw4, orig_w, orig_h, lb_scale, pad_x, pad_y, hcfg);
        int white_cnt = 0, bare_cnt = 0, gray_cnt = 0, black_cnt = 0;
        for (const auto& d : dets4) {
            if (d.class_name == "esd_white") {
                ++white_cnt;
                const double cx = 0.5 * (d.box[0] + d.box[2]);
                REQUIRE(std::abs(cx - 0.20) < 0.02);
            } else if (d.class_name == "bare_hand") {
                ++bare_cnt;
                const double cy = 0.5 * (d.box[1] + d.box[3]);
                REQUIRE(cy >= 0.22 && cy <= 0.88);
                REQUIRE(d.score >= 0.60);
            } else if (d.class_name == "esd_gray") ++gray_cnt;
            else if (d.class_name == "esd_black") ++black_cnt;
        }
        REQUIRE_EQ(white_cnt, 1);  // NMS deduplicated stride-16 duplicate
        REQUIRE_EQ(bare_cnt, 1);
        REQUIRE_EQ(gray_cnt, 1);
        REQUIRE_EQ(black_cnt, 1);

        // 3-stride [1, 2100, 36]
        const int dims3[3] = {1, 2100, 36};
        cv::Mat raw3 = cv::Mat::zeros(3, dims3, CV_32F);
        set_gfl_cell(raw3, 2000 + 5 * 10 + 2, 1, 0.88f, 1.0f, 1.0f, 1.0f, 1.0f);
        const auto dets3 = vision::HandGloveDetector::decode_nanodet_plus_output(
            raw3, orig_w, orig_h, lb_scale, pad_x, pad_y, hcfg);
        REQUIRE_EQ(dets3.size(), 1u);
        REQUIRE_EQ(dets3[0].class_name, "esd_white");
    }

    // 11F. 3 Physical Guardrails (white overexposure, label isolation, black carbon-fiber texture)
    {
        vision::HandGloveConfig hcfg;
        cv::Mat scene(360, 640, CV_8UC3, cv::Scalar(48, 52, 56));
        // Overexposed white glove (V=247, S=22 -> glare_warning=true, not pure specular)
        cv::rectangle(scene, cv::Rect(32, 120, 96, 140), cv::Scalar(225, 232, 247), cv::FILLED);
        // White false box inside label window (must be dropped by label isolation)
        cv::rectangle(scene, cv::Rect(275, 160, 90, 40), cv::Scalar(220, 222, 224), cv::FILLED);
        // Flat dark shadow (V=22, zero texture -> must be dropped)
        cv::rectangle(scene, cv::Rect(380, 235, 90, 100), cv::Scalar(20, 22, 22), cv::FILLED);
        // Textured carbon-fiber black ESD glove (checker weave V~30 -> must pass)
        for (int y = 110; y < 220; ++y) {
            for (int x = 500; x < 590; ++x) {
                const uint8_t v = (((x / 3) + (y / 3)) % 2 == 0) ? 14 : 42;
                scene.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
            }
        }

        std::vector<vision::HandGloveDetection> dets;
        dets.push_back({vision::HandGloveClass::ESD_WHITE, "esd_white", 0.90, {0.05, 0.33, 0.20, 0.72}});
        dets.push_back({vision::HandGloveClass::ESD_WHITE, "esd_white", 0.85, {0.43, 0.44, 0.57, 0.56}});
        dets.push_back({vision::HandGloveClass::ESD_BLACK, "esd_black", 0.82, {0.60, 0.66, 0.73, 0.92}});
        dets.push_back({vision::HandGloveClass::ESD_BLACK, "esd_black", 0.84, {0.78, 0.31, 0.92, 0.61}});

        bool glare_warn = false;
        vision::HandGloveDetector::apply_physical_guardrails(
            scene, dets, Box{0.16, 0.42, 0.84, 0.58}, Box{0.42, 0.43, 0.58, 0.57}, hcfg, glare_warn);
        REQUIRE(glare_warn);
        REQUIRE_EQ(dets.size(), 2u);
        REQUIRE_EQ(dets[0].class_name, "esd_white");
        REQUIRE(dets[0].glare_warning);
        REQUIRE_EQ(dets[1].class_name, "esd_black");
        REQUIRE(dets[1].texture_gradient >= hcfg.black_glove_min_texture_grad);
    }

    // 11G. Asymmetric Bare-Hand Hysteresis (2-frame fast enter, 10-frame slow clear) + LiveCameraEngine E2E
    {
        vision::AsymmetricBareHandHysteresis hyst(2, 10);
        REQUIRE(!hyst.update(true, false).bare_hand_active);   // frame 1 -> false
        REQUIRE(hyst.update(true, false).bare_hand_active);    // frame 2 -> true
        for (int i = 0; i < 9; ++i) {
            REQUIRE(hyst.update(false, false).bare_hand_active);  // 9 clean frames -> still true
        }
        REQUIRE(!hyst.update(false, false).bare_hand_active);  // 10th clean frame -> cleared

        LiveCameraEngine cam(0);
        cv::Mat empty_bg(kDefaultCameraHeight, kDefaultCameraWidth, CV_8UC3, cv::Scalar(48, 52, 56));
        for (int y = 0; y < empty_bg.rows; y += 8) {
            cv::line(empty_bg, cv::Point(0, y), cv::Point(empty_bg.cols, y), cv::Scalar(58, 62, 66), 1);
        }
        cv::Mat with_bare = empty_bg.clone();
        cv::rectangle(with_bare,
                      cv::Rect(static_cast<int>(0.08 * 1280), static_cast<int>(0.50 * 720),
                               static_cast<int>(0.10 * 1280), static_cast<int>(0.24 * 720)),
                      cv::Scalar(115, 150, 215), cv::FILLED);

        double ts = 50.0;
        for (int i = 0; i < 4; ++i) {
            ts += 0.04;
            REQUIRE(cam.process_injected_frame(empty_bg, ts));
        }
        ts += 0.04;
        REQUIRE(cam.process_injected_frame(with_bare, ts));
        REQUIRE_EQ(cam.get_state_json().value("bare_hand_detected", true), false);

        ts += 0.04;
        REQUIRE(cam.process_injected_frame(with_bare, ts));
        REQUIRE_EQ(cam.get_state_json().value("bare_hand_detected", false), true);
        REQUIRE_EQ(cam.get_state_json().value("alarm_latched", true), false);

        for (int i = 0; i < 9; ++i) {
            ts += 0.04;
            REQUIRE(cam.process_injected_frame(empty_bg, ts));
            REQUIRE_EQ(cam.get_state_json().value("bare_hand_detected", false), true);
        }
        ts += 0.04;
        REQUIRE(cam.process_injected_frame(empty_bg, ts));
        REQUIRE_EQ(cam.get_state_json().value("bare_hand_detected", true), false);
    }

    // 11H. ONNX Hot-Load Error Guards & HTTP Pipeline Endpoints
    {
        const auto data_dir = tmp.path() / "srv_data";
        StationDatabase db(data_dir);
        LiveCameraEngine camera(0, [&db]() { return db.get_config(); });
        TrainingManager training(data_dir);
        StationServerOptions opts{};
        opts.port = 0;
        opts.data_dir = data_dir;
        opts.web_dist_dir = tmp.path() / "web_dist";
        opts.assets_dir = StationHttpServer::detect_default_assets_dir();
        std::filesystem::create_directories(opts.web_dist_dir);
        write_file_bytes_atomic(opts.web_dist_dir / "index.html", "<html></html>", 13);

        StationHttpServer server(db, camera, training, opts);
        server.auth().set_env_override("IGBT_ENGINEER_PIN", "445566");
        const int port = server.start_background();
        REQUIRE(port > 0);
        httplib::Client cli("127.0.0.1", port);

        auto ps = cli.Get("/api/vision/pipeline_status");
        REQUIRE(ps && ps->status == 200);
        REQUIRE_EQ(json::parse(ps->body).at("hand_glove_detector").value("backend", ""), "heuristic_fallback");

        REQUIRE_EQ(cli.Post("/api/models/hand_glove/load", R"({"onnx_path":"missing.onnx"})", "application/json")->status, 401);

        auto auth_res = cli.Post("/api/auth/verify", R"({"passcode":"445566"})", "application/json");
        REQUIRE(auth_res && auth_res->status == 200);
        httplib::Headers hdrs{{"X-Admin-Token", json::parse(auth_res->body).value("admin_token", "")}};

        REQUIRE_EQ(cli.Post("/api/models/hand_glove/load", hdrs, R"({"onnx_path":"missing.onnx"})", "application/json")->status, 404);

        const auto bad_onnx = data_dir / "models" / "bad_hand.onnx";
        const std::string bad_payload = "NOT_A_VALID_ONNX_BINARY";
        REQUIRE(write_file_bytes_atomic(bad_onnx, bad_payload.data(), bad_payload.size()));
        const std::string bad_sha = sha256_hex(bad_payload);

        json req_wrong_sha{{"onnx_path", path_to_utf8(bad_onnx)}, {"expected_sha256", std::string(64, '0')}};
        REQUIRE_EQ(cli.Post("/api/models/hand_glove/load", hdrs, req_wrong_sha.dump(), "application/json")->status, 409);

        json req_corrupt{{"onnx_path", path_to_utf8(bad_onnx)}, {"expected_sha256", bad_sha}};
        REQUIRE_EQ(cli.Post("/api/models/hand_glove/load", hdrs, req_corrupt.dump(), "application/json")->status, 422);

        auto ps_after = cli.Get("/api/vision/pipeline_status");
        REQUIRE(ps_after && ps_after->status == 200);
        REQUIRE_EQ(json::parse(ps_after->body).at("hand_glove_detector").value("backend", ""), "heuristic_fallback");
        server.stop();
    }

    // 11I. 720P CPU Latency Benchmark (P50 / P95 vision pipeline budget check <= 15 ms)
    {
        LiveCameraEngine bench_cam(0);
        const cv::Mat bench_frame = make_synthetic_station_frame(1280, 720, "complete", false, false);
        double ts = 100.0;
        for (int i = 0; i < 4; ++i) {
            ts += 0.04;
            REQUIRE(bench_cam.process_injected_frame(bench_frame, ts));
        }
        std::vector<double> vision_ms;
        vision_ms.reserve(25);
        for (int i = 0; i < 25; ++i) {
            ts += 0.04;
            REQUIRE(bench_cam.process_injected_frame(bench_frame, ts));
            const json st = bench_cam.get_state_json();
            const double align_ms = st.at("tube_alignment").value("align_ms", 0.0);
            const double hand_ms = st.at("hand_glove_detector").value("last_infer_ms", 0.0);
            const double cls_ms = st.value("inference_ms", 0.0);
            vision_ms.push_back(align_ms + hand_ms + cls_ms);
        }
        std::sort(vision_ms.begin(), vision_ms.end());
        const double p50 = vision_ms[vision_ms.size() / 2];
        const double p95 = vision_ms[(vision_ms.size() * 95) / 100];
        std::cout << "       [BENCH] 720P vision pipeline (hand+RANSAC+cls): P50="
                  << p50 << " ms, P95=" << p95 << " ms" << std::endl;
        REQUIRE(p95 <= 15.0);
    }
}

}  // namespace

int main() {
    SetConsoleOutputCP(CP_UTF8);
    struct TestCase {
        const char* name;
        std::function<void()> fn;
    };

    const std::vector<TestCase> tests = {
        {"1. InspectionFSM Poka-Yoke Rules (A/B/C 1200ms + Reset Gate)", test_fsm_rules},
        {"2. Vision 720p@25fps Pacer, Occlusion/Glare & Bare-Hand Gate", test_vision_720p_25fps_and_pokayoke},
        {"3. SQLite3 WAL Storage, Legacy Migration, Purge & 409 Model Guard", test_sqlite_storage_and_migration},
        {"4. HTTP Loopback Server, Zero-Default Auth, Skip-Frame Auto-Capture, ANN_MLP/ONNX & MJPEG", test_http_server_auth_autocapture_training_and_mjpeg},
        {"5. Wave 1 & Wave 2 Hardening (Strict Default, Auth Rate Limit, SQLite FULL, HAL)", test_wave1_wave2_hardening},
        {"6. Skip-Frame Ring Buffer, Async Burst Writer, Diff-Gated Capture & Evidence Quota", test_skip_frame_storage_and_diff_gate},
        {"7. v4.2.0 Hardening (bursts/ Quota, 429 Retry-After, /api/diagnostics/runtime, Debounce & Post-Burst)", test_v42_hardening_and_diagnostics},
        {"8. Hardware Camera Optics Persistence, Anti-Glare Stepper & Specular Resistance", test_camera_optics_and_anti_glare},
        {"9. v4.3 Red Rabbit Mandatory Challenge, Anti-Cheat & SHA-256 Audit Chain", test_red_rabbit_challenge_and_hash_chain_audit},
        {"10. v4.3 ADAM-6050/6060 Native Winsock2 Modbus TCP Watchdog & Fail-Safe", test_adam_modbus_dio_watchdog_and_failsafe},
        {"11. v4.4 P0-3 RANSAC Tube Alignment, TUBE_ALIGN_FAIL Gate, NanoDet GFL & 2/10 Hysteresis", test_v44_ransac_alignment_nanodet_and_hysteresis},
    };

    int passed = 0;
    for (const auto& tc : tests) {
        std::cout << "[RUN ] " << tc.name << std::endl;
        try {
            tc.fn();
            std::cout << "[PASS] " << tc.name << std::endl;
            ++passed;
        } catch (const std::exception& exc) {
            std::cerr << "[FAIL] " << tc.name << " -> " << exc.what() << std::endl;
            return 1;
        }
    }

    std::cout << "\nAll " << passed << "/" << tests.size() << " C++20 station test suites PASSED!" << std::endl;
    return 0;
}
