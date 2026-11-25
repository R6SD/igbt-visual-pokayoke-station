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
