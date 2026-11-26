#pragma once

#include <igbt/common.hpp>
#include <igbt/vision/label_ood_classifier.hpp>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace igbt {

struct VerifiedModelRun {
    std::string model_version;
    std::string model_name;
    std::string artifact_format;  // "opencv_mlp" or "onnx"
    std::filesystem::path model_path;
    std::string sha256;
    int input_size{kDefaultInputSize};
    int input_width{kDefaultInputSize};
    int input_height{kDefaultInputSize};
    bool dual_output_available{false};
    bool has_ood_calibration{false};
    std::filesystem::path ood_calibration_path;
    json metrics;

    json to_json() const {
        const std::string path_str = path_to_utf8(model_path);
        const std::string cal_str = path_to_utf8(ood_calibration_path);
        return json{
            {"model_version", model_version},
            {"model_name", model_name},
            {"artifact_format", artifact_format},
            {"model_path", path_str},
            {"onnx_path", path_str},
            {"sha256", sha256},
            {"onnx_sha256", sha256},
            {"input_size", input_size},
            {"input_width", input_width},
            {"input_height", input_height},
            {"dual_output_available", dual_output_available},
            {"has_ood_calibration", has_ood_calibration},
            {"ood_calibration_path", cal_str},
            {"metrics", metrics},
        };
    }
};

namespace training {

class OnnxModelRegistry {
public:
    using AllocateRunDirFn = std::function<std::pair<std::string, std::filesystem::path>(const std::string& prefix)>;

    static bool is_strict_version_name(const std::string& version);

    static json import_onnx_bytes(
        const std::filesystem::path& runs_dir,
        const std::vector<uint8_t>& onnx_bytes,
        const std::string& version_name,
        int input_width,
        int input_height,
        const AllocateRunDirFn& allocate_run_dir_fn
    );

    static std::vector<VerifiedModelRun> find_verified_runs(const std::filesystem::path& runs_dir);

    static std::optional<VerifiedModelRun> resolve_verified_run(
        const std::filesystem::path& runs_dir,
        const std::string& model_version,
        const std::string& expected_sha256 = ""
    );

    static std::filesystem::path save_ood_calibration(
        const std::filesystem::path& runs_dir,
        const std::string& model_version,
        const vision::OodCalibration& cal
    );

    static std::optional<vision::OodCalibration> load_ood_calibration(
        const std::filesystem::path& runs_dir,
        const std::string& model_version,
        const std::string& expected_sha256,
        std::string& out_error
    );
};

}  // namespace training
}  // namespace igbt
