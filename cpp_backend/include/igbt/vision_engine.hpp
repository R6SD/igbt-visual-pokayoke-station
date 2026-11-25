#pragma once

#include <igbt/async_evidence_writer.hpp>
#include <igbt/common.hpp>
#include <igbt/fsm.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/videoio.hpp>

namespace igbt {

inline constexpr const char* kHeuristicModelVersion = "dynamic-tracker-cpp-v4.0";
inline constexpr const char* kHeuristicBackendLabel = "C++20 动态多目标视觉追踪引擎";

// ============================================================================
// Geometry & Optical Quality Gate (ViewValidEvaluator)
// ============================================================================

bool valid_box(const Box& box);
bool valid_optional_box(const std::optional<Box>& box);
double box_intersection_fraction(const Box& box, const Box& region);
double overlap_of_box_fraction(const Box& box, const Box& region);

struct QualityMetrics {
    double brightness_mean{0.0};
    double overexposed_fraction{0.0};
    double underexposed_fraction{0.0};
    double laplacian_variance{0.0};
};

std::optional<QualityMetrics> measure_label_quality(const cv::Mat& frame, const Box& label_roi);

struct ViewConfig {
    std::string camera_device_id{"CAM_STATION_01"};
    int calibrated_width{kDefaultCameraWidth};    // 1280
    int calibrated_height{kDefaultCameraHeight};  // 720
    std::string roi_config_version{"ROI_V001"};
    ROISet rois{};
    bool has_left_grip_zone{true};
    bool has_right_grip_zone{true};
    bool quality_thresholds_approved{true};
    bool orientation_detector_ready{true};
    std::optional<double> min_brightness{55.0};
    std::optional<double> max_brightness{225.0};
    std::optional<double> max_overexposed_fraction{0.08};
    std::optional<double> max_underexposed_fraction{0.15};
    std::optional<double> min_laplacian_variance{120.0};
    double max_frame_gap_ms{200.0};
    double stable_window_ms{400.0};
    int min_stable_frames{5};
    double max_center_drift_norm{0.008};
    double max_label_roi_offset_norm{0.04};

    void validate() const;
};

ViewConfig default_view_config(
    int width = kDefaultCameraWidth,
    int height = kDefaultCameraHeight,
    const std::string& camera_device_id = "CAM_STATION_01",
    const std::string& roi_config_version = "ROI_V001",
    const ROISet& rois = ROISet{}
);

struct FrameObservation {
    int64_t frame_id{0};
    double captured_monotonic_s{0.0};
    std::string camera_device_id{"CAM_STATION_01"};
    std::optional<std::string> roi_config_version{"ROI_V001"};
    std::optional<bool> camera_healthy{true};
    std::optional<bool> tube_present{true};
    std::optional<bool> label_window_present{true};
    std::optional<bool> label_occluded{false};
    std::optional<bool> front_face_confirmed{true};
    std::optional<bool> orientation_reference_visible{true};
    std::optional<std::string> orientation_evidence_source{"machine_reference"};
    std::optional<std::pair<double, double>> tube_center_norm;
    std::optional<std::pair<double, double>> label_center_norm;
    std::vector<Box> occluder_boxes_norm;
};

struct ViewResult {
    bool view_valid{false};
    std::vector<std::string> reasons;
    std::optional<QualityMetrics> quality;
    int stable_frames{0};
    std::string roi_config_version;
    std::string event{"VIEW_VALIDITY_OBSERVATION"};
};

class ViewValidEvaluator {
public:
    explicit ViewValidEvaluator(const ViewConfig& config = default_view_config());

    void reset();
    ViewResult evaluate(const cv::Mat& frame, const FrameObservation& observation);
    const ViewConfig& config() const { return config_; }

private:
    struct HistoryItem {
        double timestamp_s{0.0};
        int64_t frame_id{0};
        std::pair<double, double> tube_center{0.5, 0.5};
        std::pair<double, double> label_center{0.5, 0.5};
    };

    std::pair<std::vector<std::string>, std::optional<QualityMetrics>> base_checks(
        const cv::Mat& frame,
        const FrameObservation& obs
    ) const;

    ViewConfig config_;
    std::deque<HistoryItem> history_;
    std::optional<int64_t> last_frame_id_;
    std::optional<double> last_timestamp_;
};

// ============================================================================
// Hand / ESD Glove Contact Candidate Tracker
// ============================================================================

struct DetectionItem {
    std::string class_name;
    double score{0.0};
    Box box{0.0, 0.0, 0.0, 0.0};
};

struct ContactConfig {
    double hand_score_threshold{0.70};
    double contact_overlap_ratio{0.20};
    double grip_overlap_ratio_exclusion{0.10};
};

struct ContactCandidate {
    bool hand_contact_candidate{false};
    bool label_contact_event{false};
    std::string event{"none"};
    std::string reason{"clear"};
    std::string evidence_type{"2d_visual_proximity_proxy"};
    bool proves_physical_contact{false};
};

ContactCandidate detect_contact_candidate(
    const std::vector<DetectionItem>& detections,
    const Box& label_contact_roi,
    const std::optional<Box>& left_grip_zone,
    const std::optional<Box>& right_grip_zone,
    const ContactConfig& config = ContactConfig{}
);

class ContactCandidateTracker {
public:
    explicit ContactCandidateTracker(int enter_frames = 2, int exit_frames = 2);

    ContactCandidate update(const ContactCandidate& candidate, bool view_available = true);
    void reset();
    bool is_inside() const { return inside_; }

private:
    int enter_frames_{2};
    int exit_frames_{2};
    bool inside_{false};
    int enter_count_{0};
    int exit_count_{0};
};

// ============================================================================
// Deterministic Station Simulation Engine
// ============================================================================

struct SimulationRunResult {
    std::string scenario;
    json snapshot;
    json transitions;  // JSON array of FSM transition records
    std::string reference_image_path;
    std::vector<uint8_t> evidence_image_bytes;
    std::string evidence_kind;
    std::string pipe_color;

    json to_json() const;
};

// Supports: "a_pass", "b_pass", "c_pass", "initial_alarm" (alias "alarm"),
//           "bare_hand_alarm", "glare_invalid", "reset"
SimulationRunResult run_simulation_scenario(
    const std::string& scenario,
    const std::filesystem::path& assets_dir = {}
);

// ============================================================================
// Hardware Camera Optics & Closed-Loop Anti-Glare Configuration
// ============================================================================

struct CameraOpticsConfig {
    bool lock_3a{true};
    double exposure{-7.0};
    double gain{0.0};
    int focus{-1};
    int wb_temperature{-1};
    int backlight_comp{0};
    bool calibrated{false};
};

json camera_optics_to_json(const CameraOpticsConfig& cfg);
CameraOpticsConfig camera_optics_from_json(
    const json& j,
    const CameraOpticsConfig& base = CameraOpticsConfig{}
);

// ============================================================================
// Real-Time 720P @ 25.0 FPS Live Camera, Dual Model Inference & Auto-Capture
// ============================================================================

struct AtomicCaptureSnapshot {
    cv::Mat frame_bgr;
    cv::Mat crop_bgr;
    std::vector<uint8_t> frame_jpeg;
    std::optional<Box> crop_box;
    json state;

    bool has_frame() const { return !frame_bgr.empty(); }
};

struct WaitJpegResult {
    std::vector<uint8_t> jpeg;
    uint64_t seq{0};

    bool has_frame() const { return !jpeg.empty(); }
};

class LiveCameraEngine {
public:
    // Callback invoked on cycle/state transitions:
    // (snapshot_json, prev_fsm_state, frozen_alarm_evidence_jpeg) -> bool (true if persisted)
    using CycleCallback = std::function<bool(
        const json& snapshot,
        const std::optional<std::string>& prev_fsm_state,
        const std::vector<uint8_t>& evidence_jpeg
    )>;

    // Callback invoked during skip-frame auto-capture:
    // (frame_bgr, crop_bgr, label, pipe_color, physical_tube_id, batch_id, frame_box)
    using AutoSampleCallback = std::function<void(
        const cv::Mat& frame_bgr,
        const cv::Mat& crop_bgr,
        const std::string& label,
        const std::string& pipe_color,
        const std::string& physical_tube_id,
        const std::string& batch_id,
        std::optional<Box> frame_box
    )>;

    using ConfigLoader = std::function<json()>;

    explicit LiveCameraEngine(
        int device_index = 0,
        ConfigLoader config_loader = nullptr,
        CycleCallback cycle_callback = nullptr
    );
    ~LiveCameraEngine();

    LiveCameraEngine(const LiveCameraEngine&) = delete;
    LiveCameraEngine& operator=(const LiveCameraEngine&) = delete;

    // Lifecycle & 720p@25fps camera loop
    bool start();
    bool stop(int timeout_ms = 2500);
    bool is_running() const { return running_.load(); }
    bool is_online() const;
    bool camera_online() const { return is_online(); }

    // Callbacks & configuration
    void set_cycle_callback(CycleCallback cb);
    void set_config_loader(ConfigLoader loader);
    void update_config(const json& cfg);

    // Operator / Engineer runtime controls
    bool set_mirror_horizontal(bool mirror);
    bool get_mirror_horizontal() const;
    std::string set_work_mode(const std::string& mode);
    std::string get_work_mode() const;
    bool set_show_auxiliary_rois(bool show);
    bool get_show_auxiliary_rois() const;

    // Hardware Camera Optics & Closed-Loop Anti-Glare Controls
    struct AntiGlareStepResult {
        double next_exposure{-7.0};
        double next_gain{0.0};
        bool converged{false};
        double penalty{0.0};
    };

    using OpticsPersistCallback = std::function<void(const CameraOpticsConfig&)>;

    static AntiGlareStepResult next_anti_glare_step(
        double cur_exposure,
        double cur_gain,
        double mean_lum,
        double over_frac,
        double p95_lum
    );

    void set_optics_persist_callback(OpticsPersistCallback cb);
    json set_camera_optics(const CameraOpticsConfig& cfg, bool mark_calibrated = true);
    json get_camera_optics_json() const;
    json trigger_anti_glare_calibration();

    // Cycle & simulation controls
    static std::string generate_live_cycle_id();
    json trigger_new_cycle(const std::string& pipe_color = "yellow");
    void pause_for_simulation(double seconds = 6.0);
    json resume_live();
    SimulationRunResult run_simulation_scenario(
        const std::string& scenario,
        const std::filesystem::path& assets_dir = {}
    );
    std::size_t discard_pending_transitions(const std::vector<std::string>& cycle_ids = {});

    // Pure C++ Dual Model Hot-Load (OpenCV ANN_MLP .xml & OpenCV DNN .onnx with SHA-256 check)
    json load_mlp_model(
        const std::filesystem::path& xml_path,
        const std::string& model_version,
        const std::string& expected_sha256 = ""
    );
    json load_onnx_model(
        const std::filesystem::path& onnx_path,
        const std::string& model_version,
        const std::string& expected_sha256 = "",
        int input_size = kDefaultInputSize
    );
    void clear_loaded_model();

    // Skip-Frame Auto-Capture (A3) & Async Pre-Trigger Ring Buffer
    AutoCaptureConfig set_auto_capture(const AutoCaptureConfig& cfg);
    AutoCaptureConfig get_auto_capture() const;
    void set_auto_sample_callback(AutoSampleCallback cb);
    void set_evidence_root(const std::filesystem::path& root);
    json async_evidence_stats() const;
    void stop_async_writer();

    // Thread-safe state, atomic capture & MJPEG streaming
    json get_state_json() const;
    AtomicCaptureSnapshot get_atomic_capture(
        std::optional<Box> custom_box = std::nullopt,
        bool encode_jpeg = true
    ) const;
    std::vector<uint8_t> get_raw_frame_jpeg(int quality = 92) const;
    std::vector<uint8_t> get_hud_jpeg() const;
    std::vector<uint8_t> get_crop_jpeg() const;
    std::pair<std::vector<uint8_t>, uint64_t> get_hud_jpeg_with_seq() const;
    std::pair<std::vector<uint8_t>, uint64_t> get_crop_jpeg_with_seq() const;
    cv::Mat get_tracked_crop_bgr() const;
    WaitJpegResult wait_for_jpeg(uint64_t last_seq, int timeout_ms = 500, bool crop = false) const;

    // Deterministic synthetic frame injection hook (used by unit/integration tests without hardware camera)
    bool process_injected_frame(const cv::Mat& raw_bgr, std::optional<double> capture_ts = std::nullopt);

    // Pacer helper exposed for unit verification of 40ms (25.0 FPS) drift-compensated cadence
    static std::chrono::steady_clock::time_point next_pacer_deadline(
        std::chrono::steady_clock::time_point prev_deadline,
        std::chrono::steady_clock::time_point now,
        std::chrono::microseconds interval = std::chrono::microseconds(40000)
    );

private:
    enum class ActiveModelKind {
        NONE,
        MLP_XML,
        ONNX_DNN
    };

    struct PendingTransition {
        std::string event_id;
        json snapshot;
        std::optional<std::string> prev_fsm;
        bool needs_evidence{false};
        cv::Mat frame_bgr;
        std::vector<uint8_t> evidence_jpeg;
    };

    struct TrackResult {
        std::optional<Box> tube_box;
        std::optional<Box> label_box;
        std::optional<Box> left_glove;
        std::optional<Box> right_glove;
        std::optional<Box> bare_hand;
        bool contact_active{false};
        bool bare_hand_touching_part{false};
        bool target_locked{false};
        json tracked_objects{json::array()};
    };

    struct ClassificationResult {
        std::string label_status{"unknown"};
        std::optional<double> confidence;
        std::string pipe_color{"unknown"};
        double infer_ms{0.0};
        std::string model_version{kHeuristicModelVersion};
        std::string backend_label{kHeuristicBackendLabel};
    };

    void capture_loop();
    bool process_frame_internal(const cv::Mat& input_bgr, double capture_ts);
    void reset_tracking_state();
    void mark_camera_unavailable(const std::string& reason);
    static json read_camera_optics_hardware(
        cv::VideoCapture& cap,
        const CameraOpticsConfig& optics,
        bool auto_calibrating,
        const std::string& status_label
    );
    static json apply_camera_optics(
        cv::VideoCapture& cap,
        CameraOpticsConfig& optics,
        bool allow_converge_readback,
        const std::string& status_label
    );

    std::tuple<ROISet, json, std::string, json> active_rois_and_thresholds() const;
    std::tuple<bool, bool, std::vector<std::string>> evaluate_live_calibration(
        const json& cfg,
        int frame_w,
        int frame_h,
        const std::string& work_mode
    ) const;

    std::pair<std::optional<Box>, double> detect_tube_body(
        const cv::Mat& hsv,
        const cv::Mat& edges,
        const cv::Mat& edges_dilated,
        const cv::Mat& fg_diff,
        const cv::Mat& person_exclusion,
        int sw,
        int sh,
        bool already_locked
    ) const;

    Box project_label_window(const Box& tube_box) const;
    TrackResult track_dynamic_targets(const cv::Mat& frame, const std::string& work_mode);
    ClassificationResult classify_live_label_roi(
        const cv::Mat& crop,
        bool target_locked,
        bool contact_seen,
        bool label_occluded
    );

    std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, cv::Mat> draw_hud(
        const cv::Mat& frame,
        const ROISet& rois,
        const std::optional<Box>& tube_box,
        const std::optional<Box>& label_box,
        const std::optional<Box>& left_glove,
        const std::optional<Box>& right_glove,
        const std::optional<Box>& bare_hand,
        bool contact_active,
        bool bare_hand_violation,
        bool alarm_latched,
        bool target_locked,
        bool view_valid,
        const std::string& label_status,
        std::optional<double> label_conf,
        const std::string& work_mode,
        bool show_aux
    );

    void draw_unicode_badge(
        cv::Mat& img_bgr,
        int x,
        int y,
        const std::string& utf8_text,
        const cv::Scalar& bg_bgr
    );
    cv::Mat render_gdi_badge(const std::string& utf8_text, const cv::Scalar& bg_bgr);

    std::size_t flush_pending_transitions();
    static cv::Mat crop_norm(const cv::Mat& frame, const Box& box);

    int device_index_{0};
    mutable std::mutex mutex_;
    mutable std::condition_variable frame_cv_;
    std::mutex lifecycle_mutex_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::thread worker_thread_;

    ConfigLoader config_loader_;
    json cached_config_;
    CycleCallback cycle_callback_;
    AutoSampleCallback auto_sample_callback_;
    AutoCaptureConfig auto_capture_;

    std::unordered_map<std::string, cv::Mat> badge_cache_;

    std::string camera_name_{"HD Pro Webcam C920"};
    std::string camera_device_id_{"CAM_STATION_01"};
    std::string backend_{"DSHOW"};
    int width_{kDefaultCameraWidth};
    int height_{kDefaultCameraHeight};
    double fps_{0.0};
    bool camera_online_{false};
    double simulation_override_until_{0.0};
    bool show_auxiliary_rois_{false};
    bool mirror_horizontal_{true};
    bool applied_mirror_horizontal_{true};
    uint64_t mirror_generation_{0};
    uint64_t applied_mirror_generation_{0};
    bool pending_tracker_reset_{false};
    std::string work_mode_{"strict"};
    uint64_t work_mode_generation_{0};
    uint64_t cycle_generation_{0};
    bool hardware_3a_locked_{true};
    json hardware_3a_status_{{"locked", true}, {"source", "not_opened"}};
    CameraOpticsConfig optics_config_{};
    uint64_t optics_generation_{0};
    uint64_t anti_glare_calibrate_generation_{0};
    bool auto_calibrating_{false};
    std::string calibration_status_{"idle"};
    OpticsPersistCallback optics_persist_callback_;

    cv::Mat raw_frame_;
    std::optional<Box> raw_frame_label_box_;
    json raw_frame_state_;
    cv::Mat last_tracked_crop_;
    std::vector<uint8_t> hud_jpeg_;
    std::vector<uint8_t> crop_jpeg_;
    uint64_t hud_seq_{0};
    uint64_t crop_seq_{0};
    int64_t frame_id_{0};
    std::deque<double> fps_times_;

    // Dynamic tracking buffers (capture thread only)
    cv::Mat bg_small_;
    cv::Mat prev_small_;
    std::optional<Box> tracked_label_box_;
    std::optional<Box> tracked_tube_box_;
    std::array<double, 4> label_rel_offset_{0.0, 0.0, 0.30, 0.72};
    std::optional<Box> tracked_left_glove_;
    std::optional<Box> tracked_right_glove_;
    std::optional<Box> tracked_bare_hand_;
    int left_glove_ttl_{0};
    int right_glove_ttl_{0};
    int bare_hand_ttl_{0};
    int label_ttl_{0};
    int tube_ttl_{0};
    bool target_locked_{false};
    double target_confidence_{0.0};

    // Alarm latch, deduplication & 2000ms per-cycle evidence cooldown
    bool alarm_latched_{false};
    std::optional<std::string> latched_reason_;
    std::optional<std::string> latched_cycle_id_;
    std::optional<double> pass_target_lost_since_;
    std::size_t max_pending_transitions_{256};
    std::deque<PendingTransition> pending_transitions_;
    bool persistence_overflow_fault_{false};
    std::string last_emitted_dedup_key_;
    std::unordered_map<std::string, double> last_evidence_emit_by_cycle_;

    // Pure C++ Dual Model state (OpenCV ANN_MLP + OpenCV DNN ONNX)
    mutable std::mutex model_mutex_;
    ActiveModelKind active_model_kind_{ActiveModelKind::NONE};
    cv::Ptr<cv::ml::ANN_MLP> mlp_model_;
    cv::dnn::Net onnx_net_;
    int onnx_input_size_{kDefaultInputSize};
    std::string active_model_version_{kHeuristicModelVersion};
    std::string active_model_path_;
    std::string active_model_sha256_;
    double inference_ms_{0.0};

    InspectionFSM fsm_;
    ContactCandidateTracker contact_tracker_;
    json state_;

    SkipFrameRingBuffer pre_trigger_ring_{25, 5};
    AsyncEvidenceWriter async_evidence_writer_;
    struct ActiveBurstCapture {
        bool active{false};
        std::string cycle_id;
        std::string reason;
        double trigger_mono{0.0};
        int post_frames_remaining{0};
        std::vector<BufferedFrame> frames;
    };
    ActiveBurstCapture active_burst_;
    cv::Mat last_auto_capture_gray_small_;
    std::optional<std::pair<double, double>> active_cycle_tube_center_;
    std::string active_cycle_pipe_color_;
    std::optional<double> active_cycle_end_asymmetry_;
    int bare_hand_violation_streak_{0};
    int consecutive_detach_count_{0};
};

}  // namespace igbt
