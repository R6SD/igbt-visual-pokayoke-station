#pragma once

#include <igbt/common.hpp>

#include <array>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>

namespace igbt::vision {

struct OodCalibration {
    int schema_version{1};
    std::string model_version;
    std::string model_sha256;
    std::string calibrated_at_utc;
    int input_width{kDefaultInputSize};
    int input_height{kDefaultInputSize};
    std::string logits_output_name{"logits"};
    std::string feature_output_name{"penultimate_feat"};
    bool dual_output_available{false};

    double temperature{1.0};
    double prob_threshold{0.85};
    double energy_threshold{-1.0};
    double mahalanobis_threshold{12.0};
    double regularization_lambda{1e-3};

    int feature_dim{0};
    std::array<std::vector<double>, 4> class_means;
    std::vector<double> shared_precision_flat;

    bool enabled{true};
    double validation_auroc{-1.0};
    double validation_fpr95{-1.0};
    int id_sample_count{0};
    int ood_sample_count{0};

    bool has_mahalanobis_params() const noexcept;
    json to_json(bool include_matrices = true) const;
    static std::optional<OodCalibration> from_json(const json& j, std::string& out_error);
};

struct OodGateConfig {
    bool gate_enabled{true};
    int alarm_streak_threshold{5};
    bool require_calibration_for_onnx{false};
    std::optional<double> prob_threshold_override;
    std::optional<double> energy_threshold_override;
    std::optional<double> mahalanobis_threshold_override;

    json to_json() const;
};

struct LabelOodEvalResult {
    bool valid{false};
    bool ood_rejected{false};
    std::string reject_reason;
    std::string raw_predicted_class{"unknown"};
    std::string final_label_class{"unknown"};

    std::array<double, 4> logits{};
    std::array<double, 4> calibrated_probs{};
    double max_prob{0.0};
    double energy_score{0.0};
    std::optional<double> mahalanobis_distance;
    int nearest_class_idx{-1};
    std::vector<float> feature_vec;

    double infer_ms{0.0};
    std::string backend_mode{"heuristic_fallback"};
    std::string model_version;
    std::string model_sha256;

    json to_json() const;
};

struct CalibrationFeatureSample {
    int class_idx{-1};  // 0..3 for ID classes, -1 for OOD negative samples
    std::array<double, 4> logits{};
    std::vector<double> feature;
};

struct OnnxProbeInfo {
    bool valid{false};
    bool dual_output{false};
    std::string logits_output_name;
    std::string feature_output_name;
    int logits_dim{0};
    int feature_dim{0};
    std::string error;

    json to_json() const;
};

bool compute_temperature_softmax_and_energy(
    const std::array<double, 4>& logits,
    double temperature,
    std::array<double, 4>& out_probs,
    double& out_energy
);

bool compute_mahalanobis_distance(
    const std::vector<double>& feat,
    const OodCalibration& cal,
    int& out_nearest_cls,
    double& out_distance
);

OodCalibration calibrate_ood_from_samples(
    const std::vector<CalibrationFeatureSample>& samples,
    const std::string& model_version,
    const std::string& model_sha256,
    int input_w,
    int input_h,
    bool dual_output_available,
    const std::string& logits_name,
    const std::string& feat_name,
    double quantile = 0.98,
    double prob_threshold = 0.85,
    double reg_lambda = 1e-3,
    bool fit_temperature = true
);

class LabelOodClassifier {
public:
    enum class ModelKind {
        NONE,
        MLP_XML,
        ONNX_DNN
    };

    LabelOodClassifier() = default;

    static OnnxProbeInfo probe_onnx_bytes(
        const std::vector<uint8_t>& bytes,
        int input_width,
        int input_height,
        cv::dnn::Net* out_net = nullptr
    );

    bool load_onnx(
        const std::filesystem::path& onnx_path,
        const std::string& model_version,
        const std::string& expected_sha256,
        int input_width,
        int input_height,
        const std::filesystem::path& calibration_path,
        std::string& out_error
    );

    bool load_mlp(
        const std::filesystem::path& xml_path,
        const std::string& model_version,
        const std::string& expected_sha256,
        const std::filesystem::path& calibration_path,
        std::string& out_error
    );

    void clear_model();

    bool load_calibration(const OodCalibration& cal, std::string& out_error);
    bool load_calibration_file(const std::filesystem::path& path, std::string& out_error);
    void clear_calibration();

    void set_gate_config(const OodGateConfig& cfg);
    OodGateConfig gate_config() const;

    bool has_neural_model() const;
    ModelKind model_kind() const;
    std::string active_model_version() const;
    std::string active_model_sha256() const;
    std::filesystem::path active_model_path() const;
    int input_width() const;
    int input_height() const;
    bool dual_output_available() const;
    std::optional<OodCalibration> active_calibration() const;

    LabelOodEvalResult evaluate(const cv::Mat& crop_bgr, bool capture_feature = false);

    OodCalibration calibrate_from_crops(
        const std::vector<std::pair<cv::Mat, int>>& labeled_crops,
        double quantile = 0.98,
        double prob_threshold = 0.85,
        double reg_lambda = 1e-3,
        bool fit_temperature = true
    );

    json status_json() const;

private:
    static cv::Mat build_onnx_blob(const cv::Mat& crop_bgr, int input_width, int input_height);
    bool forward_unlocked(
        const cv::Mat& crop_bgr,
        std::array<double, 4>& out_logits,
        std::vector<double>& out_feature,
        std::string& out_error
    );
    bool try_autoload_sidecar_calibration_unlocked(
        const std::filesystem::path& explicit_cal_path,
        const std::filesystem::path& model_path,
        std::string& out_error
    );

    mutable std::mutex mutex_;
    ModelKind model_kind_{ModelKind::NONE};
    cv::Ptr<cv::ml::ANN_MLP> mlp_model_;
    cv::dnn::Net onnx_net_;

    std::string model_version_;
    std::filesystem::path model_path_;
    std::string model_sha256_;
    int input_width_{kDefaultInputSize};
    int input_height_{kDefaultInputSize};
    bool dual_output_available_{false};
    std::string logits_output_name_{"logits"};
    std::string feature_output_name_{"penultimate_feat"};
    int feature_dim_{0};

    std::optional<OodCalibration> calibration_;
    std::filesystem::path calibration_path_;
    std::string calibration_error_;
    OodGateConfig gate_config_{};

    std::optional<LabelOodEvalResult> last_eval_;
};

}  // namespace igbt::vision
