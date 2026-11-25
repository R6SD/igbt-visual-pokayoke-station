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

    // In-work-zone bare skin touching tube requires 3 consecutive frames before latching ALARM
    const cv::Mat f_bare = make_tube_scene(false, true);
    ts += 0.04;
    REQUIRE(cam.process_injected_frame(f_bare, ts));
    REQUIRE_EQ(cam.get_state_json().value("alarm_latched", true), false);

    ts += 0.04;
    REQUIRE(cam.process_injected_frame(f_bare, ts));
    REQUIRE_EQ(cam.get_state_json().value("alarm_latched", true), false);

    ts += 0.04;
    REQUIRE(cam.process_injected_frame(f_bare, ts));
    {
        const json st3 = cam.get_state_json();
        REQUIRE_EQ(st3.value("alarm_latched", false), true);
        REQUIRE_EQ(st3.value("bare_hand_violation", false), true);
        REQUIRE_EQ(st3.value("result_reason", ""), "BARE_HAND_VIOLATION");
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
