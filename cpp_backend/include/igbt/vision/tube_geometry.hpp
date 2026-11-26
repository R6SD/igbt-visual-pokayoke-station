#pragma once

#include <igbt/common.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace igbt::vision {

struct TubeAlignConfig {
    int work_width{640};
    int work_height{360};
    int out_width{256};
    int out_height{64};
    int label_out_width{160};
    int label_out_height{96};

    double midband_start_frac{0.22};
    double midband_end_frac{0.78};
    double max_angle_deg{25.0};
    double max_parallel_diff_deg{3.0};
    double max_cond_number{3.5};
    double min_det_2x2{0.25};
    double max_det_2x2{4.0};
    double max_width_rel_dev{0.12};
    double max_angle_jump_deg{15.0};
    double ema_alpha{0.6};
    double perspective_angle_thresh_deg{5.0};

    int max_reuse_frames{2};
    int hard_fail_frames{3};
};

enum class TubeAlignStatus {
    NO_TUBE,
    ALIGNED,
    REUSED_LOW_CONF,
    TUBE_ALIGN_FAIL
};

std::string tube_align_status_to_string(TubeAlignStatus status);

struct TubeAlignResult {
    TubeAlignStatus status{TubeAlignStatus::NO_TUBE};
    bool geometry_ok{false};
    bool reused_prev_matrix{false};
    bool hard_failed{false};
    int consecutive_failures{0};

    double angle_deg{0.0};
    double parallel_diff_deg{0.0};
    double condition_number{1.0};
    double det_2x2{1.0};
    double measured_width_px{0.0};
    double expected_width_px{0.0};
    double width_rel_error{0.0};
    double inlier_ratio{0.0};
    bool used_perspective{false};
    std::string fail_reason;

    cv::Mat homography_3x3;  // CV_64F 3x3 mapping full-res frame -> out_width x out_height
    std::array<cv::Point2f, 4> quad_norm{};
    bool has_quad{false};
    cv::Mat rectified_tube_bgr;
    cv::Mat rectified_label_bgr;
    double align_ms{0.0};

    json to_json() const;
};

class TubeGeometryAligner {
public:
    explicit TubeGeometryAligner(const TubeAlignConfig& config = TubeAlignConfig{});

    void reset();
    void update_config(const TubeAlignConfig& config);
    const TubeAlignConfig& config() const noexcept { return config_; }

    TubeAlignResult align(
        const cv::Mat& frame_bgr,
        const std::optional<Box>& tube_box_norm,
        const std::vector<Box>& hand_boxes_norm,
        const std::optional<Box>& label_box_norm = std::nullopt,
        const std::optional<double>& calibrated_tube_height_norm = std::nullopt
    );

    json last_status_json() const;

    // Validates upper-left 2x2 singular value ratio (kappa <= max_cond) and determinant range.
    static bool check_homography_gate(
        const cv::Mat& H_norm,
        double max_cond,
        double min_det,
        double max_det,
        double& out_cond,
        double& out_det
    );

private:
    TubeAlignResult handle_failure(
        const cv::Mat& frame_bgr,
        const std::optional<Box>& label_box_norm,
        const std::string& reason,
        double t0_s,
        bool force_hard_if_no_history,
        const TubeAlignResult* diagnostics = nullptr
    );

    void warp_outputs(
        const cv::Mat& frame_bgr,
        const cv::Mat& H_full,
        const std::array<cv::Point2f, 4>& quad_norm,
        const std::optional<Box>& label_box_norm,
        cv::Mat& out_tube_bgr,
        cv::Mat& out_label_bgr
    ) const;

    TubeAlignConfig config_;
    bool has_prev_valid_{false};
    int consecutive_failures_{0};
    double ema_cx_{0.0};
    double ema_cy_{0.0};
    double ema_angle_deg_{0.0};
    double ema_width_px_{0.0};
    double ema_length_px_{0.0};
    cv::Mat prev_homography_;
    std::array<cv::Point2f, 4> prev_quad_norm_{};
    json last_status_{{"status", "NO_TUBE"}, {"geometry_ok", false}, {"consecutive_failures", 0}};
};

}  // namespace igbt::vision
