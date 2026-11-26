#pragma once

#include <igbt/common.hpp>

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace igbt::vision {

enum class HandGloveClass : int {
    BARE_HAND = 0,
    ESD_WHITE = 1,
    ESD_GRAY = 2,
    ESD_BLACK = 3
};

std::string hand_glove_class_name(HandGloveClass cls);
bool is_glove_class(HandGloveClass cls);

struct HandGloveConfig {
    int input_size{320};
    int reg_max{7};
    std::vector<int> strides{8, 16, 32, 64};
    double score_threshold{0.45};
    double bare_hand_score_threshold{0.60};
    double nms_iou_threshold{0.45};
    int frame_stride{1};

    double work_zone_cy_min{0.22};
    double work_zone_cy_max{0.88};
    double white_glove_overexp_thresh{0.35};
    double black_glove_min_texture_grad{6.0};

    int bare_hand_enter_frames{2};
    int bare_hand_exit_frames{10};
};

struct HandGloveDetection {
    HandGloveClass cls{HandGloveClass::BARE_HAND};
    std::string class_name{"bare_hand"};
    double score{0.0};
    Box box{0.0, 0.0, 0.0, 0.0};
    bool glare_warning{false};
    bool wrist_skin_flag{false};
    double texture_gradient{0.0};
    std::string source{"heuristic_fallback"};

    json to_json() const;
};

struct HysteresisState {
    bool bare_hand_active{false};
    bool bare_hand_violation{false};
    int bare_enter_streak{0};
    int bare_clear_streak{0};
    int touch_enter_streak{0};
};

// Asymmetric bare-hand debounce: fast trigger (2 frames) and slow clear (10 frames).
class AsymmetricBareHandHysteresis {
public:
    explicit AsymmetricBareHandHysteresis(int enter_frames = 2, int exit_frames = 10);

    void reset();
    void configure(int enter_frames, int exit_frames);
    HysteresisState update(bool raw_bare_detected, bool raw_bare_touching_tube);
    const HysteresisState& state() const noexcept { return state_; }

private:
    int enter_frames_{2};
    int exit_frames_{10};
    HysteresisState state_{};
};

struct HandGloveInferResult {
    std::vector<HandGloveDetection> detections;
    std::optional<Box> left_glove;
    std::optional<Box> right_glove;
    std::optional<Box> bare_hand;
    std::string left_glove_class;
    std::string right_glove_class;
    bool white_glove_glare_warning{false};
    bool reused_stride_cache{false};
    std::string backend{"heuristic_fallback"};
    std::string model_version;
    double infer_ms{0.0};
};

class HandGloveDetector {
public:
    explicit HandGloveDetector(const HandGloveConfig& config = HandGloveConfig{});

    void update_config(const HandGloveConfig& config);
    const HandGloveConfig& config() const;
    void set_frame_stride(int stride);
    void reset_state();

    bool load_onnx(
        const std::filesystem::path& onnx_path,
        const std::string& model_version,
        const std::string& expected_sha256,
        std::string& out_error
    );
    void clear_onnx();
    bool is_onnx_loaded() const;
    json status_json() const;

    // Runs ONNX NanoDet-Plus-m (when loaded) or enhanced heuristic fallback,
    // without applying label-isolation guardrail #2 (call finalize_after_tube_lock once label_box is known).
    HandGloveInferResult detect_candidates(
        const cv::Mat& frame_bgr,
        const cv::Mat& hsv_small,
        const cv::Mat& skin_mask_small,
        const cv::Mat& fg_diff,
        const cv::Mat& motion_diff,
        const cv::Mat& person_exclusion,
        const cv::Mat& specular_mask,
        const cv::Mat& clipped_mask
    );

    // Applies all 3 physical guardrails (white overexposure/wrist skin, label isolation, black texture gradient).
    static void apply_physical_guardrails(
        const cv::Mat& frame_bgr,
        std::vector<HandGloveDetection>& dets,
        const std::optional<Box>& tube_box,
        const std::optional<Box>& label_box,
        const HandGloveConfig& cfg,
        bool& out_white_glare_warning
    );

    // Removes white-glove false positives confined inside the white label window.
    static void filter_label_isolation(
        std::vector<HandGloveDetection>& dets,
        const std::optional<Box>& label_box
    );

    // Populates left_glove, right_glove, bare_hand and classes from filtered detections.
    static void summarize_detections(HandGloveInferResult& result);

    // Decodes raw NanoDet-Plus-m GFL output tensor [1, N, 4 + 4*(reg_max+1)] or [N, C].
    static std::vector<HandGloveDetection> decode_nanodet_plus_output(
        const cv::Mat& raw_output,
        int orig_w,
        int orig_h,
        float letterbox_scale,
        int pad_x,
        int pad_y,
        const HandGloveConfig& cfg
    );

    HysteresisState update_hysteresis(bool raw_bare_detected, bool raw_bare_touching_tube);
    const HysteresisState& hysteresis_state() const noexcept { return hysteresis_.state(); }

private:
    HandGloveConfig config_;
    mutable std::mutex mutex_;
    bool onnx_loaded_{false};
    cv::dnn::Net net_;
    std::string model_version_;
    std::string model_path_;
    std::string model_sha256_;
    std::string last_error_;

    int frame_counter_{0};
    std::vector<HandGloveDetection> cached_onnx_dets_;
    double last_infer_ms_{0.0};
    AsymmetricBareHandHysteresis hysteresis_{2, 10};
};

}  // namespace igbt::vision
