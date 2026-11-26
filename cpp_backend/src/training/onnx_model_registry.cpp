#include <igbt/training/onnx_model_registry.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace igbt::training {

namespace {

constexpr std::size_t kMaxOnnxImportBytes = 64 * 1024 * 1024;

std::string trim_copy(const std::string& s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    return s.substr(first, last - first);
}

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool write_json_atomic(const std::filesystem::path& p, const json& doc) {
    const std::string dumped = doc.dump(2);
    return write_file_bytes_atomic(p, dumped.data(), dumped.size());
}

std::optional<json> read_json_file(const std::filesystem::path& p) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(p, bytes) || bytes.empty()) return std::nullopt;
    try {
        return json::parse(bytes.begin(), bytes.end());
    } catch (...) {
        return std::nullopt;
    }
}

std::string sha256_file_hex(const std::filesystem::path& p) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(p, bytes)) {
        return {};
    }
    return sha256_hex(bytes);
}

}  // namespace

bool OnnxModelRegistry::is_strict_version_name(const std::string& s) {
    if (s.empty() || s.size() > 64) return false;
    if (s == "." || s == ".." || s.find("..") != std::string::npos) return false;
    for (char c : s) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) {
            return false;
        }
    }
    return true;
}

json OnnxModelRegistry::import_onnx_bytes(
    const std::filesystem::path& runs_dir,
    const std::vector<uint8_t>& onnx_bytes,
    const std::string& version_name,
    int input_width,
    int input_height,
    const AllocateRunDirFn& allocate_run_dir_fn
) {
    if (onnx_bytes.empty()) {
        throw std::invalid_argument("EMPTY_ONNX_MODEL_BYTES");
    }
    if (onnx_bytes.size() > kMaxOnnxImportBytes) {
        throw std::invalid_argument("ONNX_MODEL_EXCEEDS_MAX_SIZE");
    }
    const int eff_w = input_width;
    const int eff_h = (input_height > 0) ? input_height : input_width;
    if (eff_w < 32 || eff_w > 1024 || eff_h < 32 || eff_h > 1024) {
        throw std::invalid_argument("INVALID_ONNX_INPUT_SIZE");
    }

    const std::string clean_version = trim_copy(version_name);
    if (!clean_version.empty() && !is_strict_version_name(clean_version)) {
        throw std::invalid_argument("INVALID_MODEL_VERSION_NAME");
    }

    const vision::OnnxProbeInfo probe = vision::LabelOodClassifier::probe_onnx_bytes(
        onnx_bytes,
        eff_w,
        eff_h,
        nullptr
    );
    if (!probe.valid) {
        throw std::invalid_argument(probe.error.empty() ? "INVALID_ONNX_MODEL" : probe.error);
    }

    const std::string onnx_sha = sha256_hex(onnx_bytes);
    std::filesystem::path run_dir;
    std::string allocated_version;

    if (clean_version.empty()) {
        if (!allocate_run_dir_fn) {
            throw std::runtime_error("ALLOCATE_RUN_DIR_NOT_PROVIDED");
        }
        std::tie(allocated_version, run_dir) = allocate_run_dir_fn("onnx_import");
    } else {
        allocated_version = clean_version;
        std::error_code ec;
        std::filesystem::create_directories(runs_dir, ec);
        run_dir = runs_dir / path_from_utf8(allocated_version);
        if (!is_path_within(runs_dir, run_dir)) {
            throw std::invalid_argument("ONNX_VERSION_OUTSIDE_RUNS_DIR");
        }
        if (std::filesystem::exists(run_dir, ec)) {
            allocated_version = clean_version + "_" + random_token_hex(2);
            run_dir = runs_dir / path_from_utf8(allocated_version);
        }
        std::filesystem::create_directories(run_dir, ec);
        if (ec) {
            throw std::runtime_error("Failed to create ONNX run directory: " + ec.message());
        }
    }

    const auto onnx_path = run_dir / L"label_classifier.onnx";
    if (!write_file_bytes_atomic(onnx_path, onnx_bytes.data(), onnx_bytes.size())) {
        throw std::runtime_error("Failed to persist label_classifier.onnx");
    }

    const std::string path_str = path_to_utf8(onnx_path);
    const std::string now_iso = utc_now_iso();
    const json metrics = json{
        {"model_name", "onnx-imported-igbt"},
        {"model_version", allocated_version},
        {"architecture", "opencv_dnn_onnx"},
        {"artifact_format", "onnx"},
        {"input_size", eff_w},
        {"input_width", eff_w},
        {"input_height", eff_h},
        {"output_dim", probe.logits_dim},
        {"dual_output_available", probe.dual_output},
        {"logits_output_name", probe.logits_output_name},
        {"feature_output_name", probe.feature_output_name},
        {"feature_dim", probe.feature_dim},
        {"model_path", path_str},
        {"onnx_path", path_str},
        {"sha256", onnx_sha},
        {"onnx_sha256", onnx_sha},
        {"onnx_parity_ok", true},
        {"onnx_parity_passed", true},
        {"artifact_verified", true},
        {"exported", true},
        {"completed_at_utc", now_iso},
    };

    if (!write_json_atomic(run_dir / L"metrics.json", metrics)) {
        throw std::runtime_error("Failed to persist ONNX metrics.json");
    }

    return metrics;
}

std::vector<VerifiedModelRun> OnnxModelRegistry::find_verified_runs(const std::filesystem::path& runs_dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(runs_dir, ec)) {
        return {};
    }

    struct CandidateItem {
        std::filesystem::file_time_type mtime;
        VerifiedModelRun run;
    };
    std::vector<CandidateItem> items;

    for (const auto& entry : std::filesystem::directory_iterator(runs_dir, ec)) {
        std::error_code d_ec;
        if (!entry.is_directory(d_ec)) continue;
        const auto sub_dir = entry.path();
        if (!is_path_within(runs_dir, sub_dir)) continue;

        const auto metrics_path = sub_dir / L"metrics.json";
        if (!std::filesystem::is_regular_file(metrics_path, d_ec)) continue;
        const auto meta_opt = read_json_file(metrics_path);
        if (!meta_opt.has_value() || !meta_opt->is_object()) continue;
        const json& meta = *meta_opt;

        bool verified_flag = false;
        for (const char* k : {"artifact_verified", "onnx_parity_ok", "onnx_parity_passed"}) {
            if (meta.contains(k)) {
                if (!meta[k].is_boolean() || !meta[k].get<bool>()) {
                    verified_flag = false;
                    break;
                }
                verified_flag = true;
            }
        }
        if (!verified_flag) continue;

        const auto xml_cand = sub_dir / L"label_classifier.xml";
        const auto onnx_cand = sub_dir / L"label_classifier.onnx";
        std::filesystem::path artifact_path;
        std::string fmt;
        const std::string declared_fmt = meta.value("artifact_format", "");

        if (declared_fmt == "opencv_mlp" && std::filesystem::is_regular_file(xml_cand, d_ec)) {
            artifact_path = xml_cand;
            fmt = "opencv_mlp";
        } else if (declared_fmt == "onnx" && std::filesystem::is_regular_file(onnx_cand, d_ec)) {
            artifact_path = onnx_cand;
            fmt = "onnx";
        } else if (std::filesystem::is_regular_file(xml_cand, d_ec)) {
            artifact_path = xml_cand;
            fmt = "opencv_mlp";
        } else if (std::filesystem::is_regular_file(onnx_cand, d_ec)) {
            artifact_path = onnx_cand;
            fmt = "onnx";
        } else {
            continue;
        }

        if (!is_path_within(runs_dir, artifact_path)) continue;
        const std::string actual_sha = to_lower_copy(sha256_file_hex(artifact_path));
        if (actual_sha.size() != 64) continue;

        std::string recorded_sha = meta.value("sha256", "");
        if (recorded_sha.empty()) recorded_sha = meta.value("onnx_sha256", "");
        recorded_sha = to_lower_copy(trim_copy(recorded_sha));
        if (recorded_sha.empty() || !constant_time_equals(actual_sha, recorded_sha)) {
            continue;
        }

        int in_sz = kDefaultInputSize;
        if (meta.contains("input_size") && meta["input_size"].is_number_integer()) {
            const int v = meta["input_size"].get<int>();
            if (v >= 32 && v <= 1024) in_sz = v;
        }
        int in_w = in_sz;
        if (meta.contains("input_width") && meta["input_width"].is_number_integer()) {
            const int v = meta["input_width"].get<int>();
            if (v >= 32 && v <= 1024) in_w = v;
        }
        int in_h = in_sz;
        if (meta.contains("input_height") && meta["input_height"].is_number_integer()) {
            const int v = meta["input_height"].get<int>();
            if (v >= 32 && v <= 1024) in_h = v;
        }

        const auto mt = std::filesystem::last_write_time(artifact_path, d_ec);
        if (d_ec) continue;

        VerifiedModelRun run;
        run.model_version = path_to_utf8(sub_dir.filename());
        run.model_name = meta.value("model_name", fmt == "opencv_mlp" ? "opencv-mlp-cpp-igbt" : "mobilenetv3-small-igbt-roi");
        run.artifact_format = fmt;
        run.model_path = artifact_path;
        run.sha256 = actual_sha;
        run.input_size = in_w;
        run.input_width = in_w;
        run.input_height = in_h;
        run.dual_output_available = meta.value("dual_output_available", false);
        run.metrics = meta;

        const auto cal_cand = sub_dir / L"ood_calibration.json";
        if (std::filesystem::is_regular_file(cal_cand, d_ec)) {
            std::string cal_err;
            const auto cal_json = read_json_file(cal_cand);
            if (cal_json.has_value()) {
                auto parsed_cal = vision::OodCalibration::from_json(*cal_json, cal_err);
                if (parsed_cal.has_value() && constant_time_equals(parsed_cal->model_sha256, actual_sha)) {
                    run.has_ood_calibration = true;
                    run.ood_calibration_path = cal_cand;
                }
            }
        }

        items.push_back(CandidateItem{mt, std::move(run)});
    }

    std::sort(items.begin(), items.end(), [](const CandidateItem& a, const CandidateItem& b) {
        if (a.mtime != b.mtime) return a.mtime > b.mtime;
        return a.run.model_version > b.run.model_version;
    });

    std::vector<VerifiedModelRun> result;
    result.reserve(items.size());
    for (auto& it : items) {
        result.push_back(std::move(it.run));
    }
    return result;
}

std::optional<VerifiedModelRun> OnnxModelRegistry::resolve_verified_run(
    const std::filesystem::path& runs_dir,
    const std::string& model_version,
    const std::string& expected_sha256
) {
    const std::string clean_ver = trim_copy(model_version);
    if (!is_strict_version_name(clean_ver)) return std::nullopt;
    const std::string clean_sha = to_lower_copy(trim_copy(expected_sha256));

    const auto runs = find_verified_runs(runs_dir);
    for (const auto& r : runs) {
        if (r.model_version == clean_ver) {
            if (!clean_sha.empty() && !constant_time_equals(r.sha256, clean_sha)) {
                return std::nullopt;
            }
            return r;
        }
    }
    return std::nullopt;
}

std::filesystem::path OnnxModelRegistry::save_ood_calibration(
    const std::filesystem::path& runs_dir,
    const std::string& model_version,
    const vision::OodCalibration& cal
) {
    const std::string clean_ver = trim_copy(model_version);
    if (!is_strict_version_name(clean_ver)) {
        throw std::invalid_argument("INVALID_MODEL_VERSION_NAME");
    }
    const auto run_dir = runs_dir / path_from_utf8(clean_ver);
    if (!is_path_within(runs_dir, run_dir)) {
        throw std::invalid_argument("OOD_CALIBRATION_OUTSIDE_RUNS_DIR");
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(run_dir, ec)) {
        throw std::runtime_error("MODEL_RUN_DIR_NOT_FOUND");
    }
    const auto cal_path = run_dir / L"ood_calibration.json";
    if (!write_json_atomic(cal_path, cal.to_json(true))) {
        throw std::runtime_error("FAILED_TO_PERSIST_OOD_CALIBRATION");
    }
    return cal_path;
}

std::optional<vision::OodCalibration> OnnxModelRegistry::load_ood_calibration(
    const std::filesystem::path& runs_dir,
    const std::string& model_version,
    const std::string& expected_sha256,
    std::string& out_error
) {
    out_error.clear();
    const std::string clean_ver = trim_copy(model_version);
    if (!is_strict_version_name(clean_ver)) {
        out_error = "INVALID_MODEL_VERSION_NAME";
        return std::nullopt;
    }
    const auto run_dir = runs_dir / path_from_utf8(clean_ver);
    if (!is_path_within(runs_dir, run_dir)) {
        out_error = "OOD_CALIBRATION_OUTSIDE_RUNS_DIR";
        return std::nullopt;
    }
    const auto cal_path = run_dir / L"ood_calibration.json";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(cal_path, ec)) {
        out_error = "OOD_CALIBRATION_NOT_FOUND";
        return std::nullopt;
    }
    const auto cal_json = read_json_file(cal_path);
    if (!cal_json.has_value()) {
        out_error = "OOD_CALIBRATION_INVALID_JSON";
        return std::nullopt;
    }
    auto parsed = vision::OodCalibration::from_json(*cal_json, out_error);
    if (!parsed.has_value()) {
        return std::nullopt;
    }
    const std::string clean_sha = to_lower_copy(trim_copy(expected_sha256));
    if (!clean_sha.empty() && !constant_time_equals(parsed->model_sha256, clean_sha)) {
        out_error = "OOD_CALIBRATION_SHA256_MISMATCH";
        return std::nullopt;
    }
    return parsed;
}

}  // namespace igbt::training
