#include <igbt/vision/label_ood_classifier.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace igbt::vision {

namespace {

constexpr double kMlpLogitScale = 2.2;
constexpr int kMinFeatureDim = 2;
constexpr int kMaxFeatureDim = 1024;
constexpr double kDefaultProbThresh = 0.85;

std::string to_lower_trim(std::string s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    s = s.substr(first, last - first);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool is_hex64(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

double round4(double v) {
    return std::round(v * 10000.0) / 10000.0;
}

bool extract_flat_f32(const cv::Mat& m, std::vector<double>& out) {
    if (m.empty() || m.total() == 0) return false;
    cv::Mat flat = m.reshape(1, 1);
    cv::Mat f32;
    if (flat.type() != CV_32F) {
        flat.convertTo(f32, CV_32F);
    } else {
        f32 = flat.isContinuous() ? flat : flat.clone();
    }
    const int n = f32.cols;
    out.resize(static_cast<std::size_t>(n));
    const float* ptr = f32.ptr<float>(0);
    for (int i = 0; i < n; ++i) {
        const double val = static_cast<double>(ptr[i]);
        if (!std::isfinite(val)) return false;
        out[static_cast<std::size_t>(i)] = val;
    }
    return true;
}

double linear_quantile(std::vector<double> sorted_vals, double q) {
    if (sorted_vals.empty()) return 0.0;
    std::sort(sorted_vals.begin(), sorted_vals.end());
    if (sorted_vals.size() == 1) return sorted_vals.front();
    const double clamped_q = std::clamp(q, 0.0, 1.0);
    const double pos = clamped_q * static_cast<double>(sorted_vals.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) return sorted_vals[lo];
    const double frac = pos - static_cast<double>(lo);
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

double mean_id_nll(const std::vector<CalibrationFeatureSample>& id_samples, double temperature) {
    double total_nll = 0.0;
    int count = 0;
    std::array<double, 4> probs{};
    double energy = 0.0;
    for (const auto& s : id_samples) {
        if (s.class_idx < 0 || s.class_idx >= 4) continue;
        if (!compute_temperature_softmax_and_energy(s.logits, temperature, probs, energy)) {
            return 1e9;
        }
        const double p = std::max(1e-12, probs[static_cast<std::size_t>(s.class_idx)]);
        total_nll -= std::log(p);
        ++count;
    }
    return count > 0 ? (total_nll / static_cast<double>(count)) : 1e9;
}

std::pair<double, double> compute_auroc_and_fpr95(
    const std::vector<double>& id_scores,
    const std::vector<double>& ood_scores
) {
    if (id_scores.empty() || ood_scores.empty()) {
        return {-1.0, -1.0};
    }

    // AUROC via exact pairwise Wilcoxon-Mann-Whitney statistic (handles ties cleanly)
    double concordant = 0.0;
    for (double o : ood_scores) {
        for (double i : id_scores) {
            if (o > i) {
                concordant += 1.0;
            } else if (o == i) {
                concordant += 0.5;
            }
        }
    }
    const double auroc = concordant /
        (static_cast<double>(id_scores.size()) * static_cast<double>(ood_scores.size()));

    // FPR @ 95% TPR: pick highest threshold achieving TPR >= 0.95 (smallest FPR among valid recall >= 0.95)
    std::vector<double> sorted_ood = ood_scores;
    std::sort(sorted_ood.begin(), sorted_ood.end());
    const std::size_t min_tp = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(sorted_ood.size()))
    );
    const std::size_t idx = (sorted_ood.size() >= min_tp && min_tp > 0)
                              ? (sorted_ood.size() - min_tp)
                              : 0;
    const double thr = sorted_ood[idx];
    std::size_t fp = 0;
    for (double i : id_scores) {
        if (i >= thr) ++fp;
    }
    const double fpr95 = static_cast<double>(fp) / static_cast<double>(id_scores.size());
    return {std::clamp(auroc, 0.0, 1.0), std::clamp(fpr95, 0.0, 1.0)};
}

}  // namespace

bool OodCalibration::has_mahalanobis_params() const noexcept {
    if (feature_dim < kMinFeatureDim || feature_dim > kMaxFeatureDim) {
        return false;
    }
    const std::size_t d = static_cast<std::size_t>(feature_dim);
    if (shared_precision_flat.size() != d * d) {
        return false;
    }
    for (const auto& mean_vec : class_means) {
        if (mean_vec.size() != d) return false;
    }
    return true;
}

json OodCalibration::to_json(bool include_matrices) const {
    json j{
        {"schema_version", schema_version},
        {"model_version", model_version},
        {"model_sha256", model_sha256},
        {"calibrated_at_utc", calibrated_at_utc},
        {"input_width", input_width},
        {"input_height", input_height},
        {"logits_output_name", logits_output_name},
        {"feature_output_name", feature_output_name},
        {"dual_output_available", dual_output_available},
        {"temperature", round4(temperature)},
        {"prob_threshold", round4(prob_threshold)},
        {"energy_threshold", round4(energy_threshold)},
        {"mahalanobis_threshold", round4(mahalanobis_threshold)},
        {"regularization_lambda", regularization_lambda},
        {"feature_dim", feature_dim},
        {"has_mahalanobis_params", has_mahalanobis_params()},
        {"enabled", enabled},
        {"validation_auroc", validation_auroc >= 0.0 ? json(round4(validation_auroc)) : json(nullptr)},
        {"validation_fpr95", validation_fpr95 >= 0.0 ? json(round4(validation_fpr95)) : json(nullptr)},
        {"id_sample_count", id_sample_count},
        {"ood_sample_count", ood_sample_count},
    };
    if (include_matrices) {
        json means_arr = json::array();
        for (const auto& m : class_means) {
            means_arr.push_back(m);
        }
        j["class_means"] = std::move(means_arr);
        j["shared_precision_flat"] = shared_precision_flat;
    }
    return j;
}

std::optional<OodCalibration> OodCalibration::from_json(const json& j, std::string& out_error) {
    out_error.clear();
    if (!j.is_object()) {
        out_error = "OOD_CALIBRATION_NOT_OBJECT";
        return std::nullopt;
    }

    auto read_finite = [&](const char* key, double& target, bool required) -> bool {
        if (!j.contains(key) || j[key].is_null()) {
            if (required) {
                out_error = std::string("OOD_CALIBRATION_MISSING_") + key;
                return false;
            }
            return true;
        }
        if (!j[key].is_number()) {
            out_error = std::string("OOD_CALIBRATION_INVALID_") + key;
            return false;
        }
        const double val = j[key].get<double>();
        if (!std::isfinite(val)) {
            out_error = std::string("OOD_CALIBRATION_NON_FINITE_") + key;
            return false;
        }
        target = val;
        return true;
    };

    OodCalibration cal;
    if (j.contains("schema_version")) {
        if (!j["schema_version"].is_number_integer()) {
            out_error = "OOD_CALIBRATION_INVALID_SCHEMA_VERSION";
            return std::nullopt;
        }
        cal.schema_version = j["schema_version"].get<int>();
    }
    if (j.contains("model_version") && j["model_version"].is_string()) {
        cal.model_version = j["model_version"].get<std::string>();
    }
    if (!j.contains("model_sha256") || !j["model_sha256"].is_string()) {
        out_error = "OOD_CALIBRATION_MISSING_MODEL_SHA256";
        return std::nullopt;
    }
    cal.model_sha256 = to_lower_trim(j["model_sha256"].get<std::string>());
    if (!is_hex64(cal.model_sha256)) {
        out_error = "OOD_CALIBRATION_INVALID_MODEL_SHA256";
        return std::nullopt;
    }

    if (j.contains("calibrated_at_utc") && j["calibrated_at_utc"].is_string()) {
        cal.calibrated_at_utc = j["calibrated_at_utc"].get<std::string>();
    }
    if (j.contains("input_width") && j["input_width"].is_number_integer()) {
        cal.input_width = std::clamp(j["input_width"].get<int>(), 32, 1024);
    }
    if (j.contains("input_height") && j["input_height"].is_number_integer()) {
        cal.input_height = std::clamp(j["input_height"].get<int>(), 32, 1024);
    }
    if (j.contains("logits_output_name") && j["logits_output_name"].is_string()) {
        cal.logits_output_name = j["logits_output_name"].get<std::string>();
    }
    if (j.contains("feature_output_name") && j["feature_output_name"].is_string()) {
        cal.feature_output_name = j["feature_output_name"].get<std::string>();
    }
    if (j.contains("dual_output_available") && j["dual_output_available"].is_boolean()) {
        cal.dual_output_available = j["dual_output_available"].get<bool>();
    }
    if (j.contains("enabled") && j["enabled"].is_boolean()) {
        cal.enabled = j["enabled"].get<bool>();
    }

    if (!read_finite("temperature", cal.temperature, false) ||
        !read_finite("prob_threshold", cal.prob_threshold, false) ||
        !read_finite("energy_threshold", cal.energy_threshold, true) ||
        !read_finite("mahalanobis_threshold", cal.mahalanobis_threshold, false) ||
        !read_finite("regularization_lambda", cal.regularization_lambda, false) ||
        !read_finite("validation_auroc", cal.validation_auroc, false) ||
        !read_finite("validation_fpr95", cal.validation_fpr95, false)) {
        return std::nullopt;
    }

    if (cal.temperature < 0.05 || cal.temperature > 20.0) {
        out_error = "OOD_CALIBRATION_INVALID_TEMPERATURE";
        return std::nullopt;
    }
    if (cal.prob_threshold < 0.0 || cal.prob_threshold > 1.0) {
        out_error = "OOD_CALIBRATION_INVALID_PROB_THRESHOLD";
        return std::nullopt;
    }
    if (cal.mahalanobis_threshold < 0.0) {
        out_error = "OOD_CALIBRATION_INVALID_MAHALANOBIS_THRESHOLD";
        return std::nullopt;
    }

    if (j.contains("id_sample_count") && j["id_sample_count"].is_number_integer()) {
        cal.id_sample_count = std::max(0, j["id_sample_count"].get<int>());
    }
    if (j.contains("ood_sample_count") && j["ood_sample_count"].is_number_integer()) {
        cal.ood_sample_count = std::max(0, j["ood_sample_count"].get<int>());
    }

    if (j.contains("feature_dim")) {
        if (!j["feature_dim"].is_number_integer()) {
            out_error = "OOD_CALIBRATION_INVALID_FEATURE_DIM";
            return std::nullopt;
        }
        cal.feature_dim = j["feature_dim"].get<int>();
    }
    if (cal.feature_dim < 0 || cal.feature_dim > kMaxFeatureDim) {
        out_error = "OOD_CALIBRATION_INVALID_FEATURE_DIM";
        return std::nullopt;
    }

    if (cal.feature_dim > 0) {
        if (cal.feature_dim < kMinFeatureDim) {
            out_error = "OOD_CALIBRATION_INVALID_FEATURE_DIM";
            return std::nullopt;
        }
        const std::size_t d = static_cast<std::size_t>(cal.feature_dim);
        if (!j.contains("class_means") || !j["class_means"].is_array() || j["class_means"].size() != 4) {
            out_error = "OOD_CALIBRATION_INVALID_CLASS_MEANS";
            return std::nullopt;
        }
        for (std::size_t c = 0; c < 4; ++c) {
            const auto& row = j["class_means"][c];
            if (!row.is_array() || row.size() != d) {
                out_error = "OOD_CALIBRATION_INVALID_CLASS_MEANS";
                return std::nullopt;
            }
            cal.class_means[c].resize(d);
            for (std::size_t k = 0; k < d; ++k) {
                if (!row[k].is_number()) {
                    out_error = "OOD_CALIBRATION_NON_FINITE_MATRIX";
                    return std::nullopt;
                }
                const double v = row[k].get<double>();
                if (!std::isfinite(v)) {
                    out_error = "OOD_CALIBRATION_NON_FINITE_MATRIX";
                    return std::nullopt;
                }
                cal.class_means[c][k] = v;
            }
        }
        if (!j.contains("shared_precision_flat") || !j["shared_precision_flat"].is_array() ||
            j["shared_precision_flat"].size() != d * d) {
            out_error = "OOD_CALIBRATION_INVALID_PRECISION_MATRIX";
            return std::nullopt;
        }
        cal.shared_precision_flat.resize(d * d);
        for (std::size_t i = 0; i < d * d; ++i) {
            const auto& elem = j["shared_precision_flat"][i];
            if (!elem.is_number()) {
                out_error = "OOD_CALIBRATION_NON_FINITE_MATRIX";
                return std::nullopt;
            }
            const double v = elem.get<double>();
            if (!std::isfinite(v)) {
                out_error = "OOD_CALIBRATION_NON_FINITE_MATRIX";
                return std::nullopt;
            }
            cal.shared_precision_flat[i] = v;
        }
    }

    return cal;
}

json OodGateConfig::to_json() const {
    return json{
        {"gate_enabled", gate_enabled},
        {"alarm_streak_threshold", std::clamp(alarm_streak_threshold, 1, 30)},
        {"require_calibration_for_onnx", require_calibration_for_onnx},
        {"prob_threshold_override", prob_threshold_override.has_value() ? json(*prob_threshold_override) : json(nullptr)},
        {"energy_threshold_override", energy_threshold_override.has_value() ? json(*energy_threshold_override) : json(nullptr)},
        {"mahalanobis_threshold_override", mahalanobis_threshold_override.has_value() ? json(*mahalanobis_threshold_override) : json(nullptr)},
    };
}

json LabelOodEvalResult::to_json() const {
    json probs_json = json::object();
    json logits_json = json::object();
    for (std::size_t i = 0; i < 4; ++i) {
        probs_json[kLabelClasses[i]] = round4(calibrated_probs[i]);
        logits_json[kLabelClasses[i]] = round4(logits[i]);
    }
    return json{
        {"valid", valid},
        {"ood_rejected", ood_rejected},
        {"reject_reason", reject_reason},
        {"raw_predicted_class", raw_predicted_class},
        {"final_label_class", final_label_class},
        {"logits", std::move(logits_json)},
        {"calibrated_probs", std::move(probs_json)},
        {"max_prob", round4(max_prob)},
        {"energy_score", round4(energy_score)},
        {"mahalanobis_distance", mahalanobis_distance.has_value() ? json(round4(*mahalanobis_distance)) : json(nullptr)},
        {"nearest_class_idx", nearest_class_idx},
        {"infer_ms", round4(infer_ms)},
        {"backend_mode", backend_mode},
        {"model_version", model_version},
        {"model_sha256", model_sha256},
    };
}

json OnnxProbeInfo::to_json() const {
    return json{
        {"valid", valid},
        {"dual_output", dual_output},
        {"logits_output_name", logits_output_name},
        {"feature_output_name", feature_output_name},
        {"logits_dim", logits_dim},
        {"feature_dim", feature_dim},
        {"error", error},
    };
}

bool compute_temperature_softmax_and_energy(
    const std::array<double, 4>& logits,
    double temperature,
    std::array<double, 4>& out_probs,
    double& out_energy
) {
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        return false;
    }
    for (double z : logits) {
        if (!std::isfinite(z)) return false;
    }

    std::array<double, 4> scaled{};
    double max_scaled = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < 4; ++i) {
        scaled[i] = logits[i] / temperature;
        if (!std::isfinite(scaled[i])) return false;
        if (scaled[i] > max_scaled) max_scaled = scaled[i];
    }

    std::array<double, 4> exps{};
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        exps[i] = std::exp(scaled[i] - max_scaled);
        if (!std::isfinite(exps[i])) return false;
        sum_exp += exps[i];
    }
    if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
        return false;
    }

    for (std::size_t i = 0; i < 4; ++i) {
        out_probs[i] = exps[i] / sum_exp;
    }
    const double lse = max_scaled + std::log(sum_exp);
    const double energy = -temperature * lse;
    if (!std::isfinite(energy)) {
        return false;
    }
    out_energy = energy;
    return true;
}

bool compute_mahalanobis_distance(
    const std::vector<double>& feat,
    const OodCalibration& cal,
    int& out_nearest_cls,
    double& out_distance
) {
    if (!cal.has_mahalanobis_params()) {
        return false;
    }
    const std::size_t d = static_cast<std::size_t>(cal.feature_dim);
    if (feat.size() != d) {
        return false;
    }
    for (double v : feat) {
        if (!std::isfinite(v)) return false;
    }

    int best_cls = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    std::vector<double> diff(d);

    for (int c = 0; c < 4; ++c) {
        const auto& mu = cal.class_means[static_cast<std::size_t>(c)];
        for (std::size_t i = 0; i < d; ++i) {
            diff[i] = feat[i] - mu[i];
        }
        double quad = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            const double* row = cal.shared_precision_flat.data() + i * d;
            double row_dot = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                row_dot += row[j] * diff[j];
            }
            quad += diff[i] * row_dot;
        }
        if (!std::isfinite(quad)) {
            return false;
        }
        if (quad < best_d2) {
            best_d2 = quad;
            best_cls = c;
        }
    }

    if (best_cls < 0 || !std::isfinite(best_d2)) {
        return false;
    }
    out_nearest_cls = best_cls;
    out_distance = std::sqrt(std::max(0.0, best_d2));
    return std::isfinite(out_distance);
}

OodCalibration calibrate_ood_from_samples(
    const std::vector<CalibrationFeatureSample>& samples,
    const std::string& model_version,
    const std::string& model_sha256,
    int input_w,
    int input_h,
    bool dual_output_available,
    const std::string& logits_name,
    const std::string& feat_name,
    double quantile,
    double prob_threshold,
    double reg_lambda,
    bool fit_temperature
) {
    std::vector<CalibrationFeatureSample> id_samples;
    std::vector<CalibrationFeatureSample> ood_samples;
    for (const auto& s : samples) {
        bool finite_logits = true;
        for (double z : s.logits) {
            if (!std::isfinite(z)) {
                finite_logits = false;
                break;
            }
        }
        if (!finite_logits) continue;
        if (s.class_idx >= 0 && s.class_idx < 4) {
            id_samples.push_back(s);
        } else if (s.class_idx < 0) {
            ood_samples.push_back(s);
        }
    }

    if (id_samples.size() < 4) {
        throw std::invalid_argument("INSUFFICIENT_CALIBRATION_SAMPLES");
    }

    const double clean_q = std::clamp(std::isfinite(quantile) ? quantile : 0.98, 0.50, 0.9999);
    const double clean_prob = std::clamp(std::isfinite(prob_threshold) ? prob_threshold : kDefaultProbThresh, 0.0, 1.0);
    const double clean_lambda = std::clamp(std::isfinite(reg_lambda) ? reg_lambda : 1e-3, 1e-6, 1.0);

    double best_temp = 1.0;
    if (fit_temperature) {
        // Golden-section search over T in [0.2, 5.0] minimizing ID NLL
        double lo = 0.2;
        double hi = 5.0;
        const double inv_phi = (std::sqrt(5.0) - 1.0) * 0.5;
        double c = hi - inv_phi * (hi - lo);
        double d = lo + inv_phi * (hi - lo);
        double fc = mean_id_nll(id_samples, c);
        double fd = mean_id_nll(id_samples, d);
        for (int iter = 0; iter < 40; ++iter) {
            if (fc < fd) {
                hi = d;
                d = c;
                fd = fc;
                c = hi - inv_phi * (hi - lo);
                fc = mean_id_nll(id_samples, c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + inv_phi * (hi - lo);
                fd = mean_id_nll(id_samples, d);
            }
        }
        const double cand = (lo + hi) * 0.5;
        if (mean_id_nll(id_samples, cand) <= mean_id_nll(id_samples, 1.0)) {
            best_temp = cand;
        }
    }

    OodCalibration cal;
    cal.schema_version = 1;
    cal.model_version = model_version;
    cal.model_sha256 = to_lower_trim(model_sha256);
    cal.calibrated_at_utc = utc_now_iso();
    cal.input_width = std::clamp(input_w, 32, 1024);
    cal.input_height = std::clamp(input_h, 32, 1024);
    cal.logits_output_name = logits_name.empty() ? "logits" : logits_name;
    cal.feature_output_name = feat_name.empty() ? "penultimate_feat" : feat_name;
    cal.dual_output_available = dual_output_available;
    cal.temperature = best_temp;
    cal.prob_threshold = clean_prob;
    cal.regularization_lambda = clean_lambda;
    cal.enabled = true;
    cal.id_sample_count = static_cast<int>(id_samples.size());
    cal.ood_sample_count = static_cast<int>(ood_samples.size());

    std::vector<double> id_energies;
    id_energies.reserve(id_samples.size());
    std::array<double, 4> tmp_probs{};
    for (const auto& s : id_samples) {
        double e = 0.0;
        if (compute_temperature_softmax_and_energy(s.logits, best_temp, tmp_probs, e)) {
            id_energies.push_back(e);
        }
    }
    if (id_energies.empty()) {
        throw std::invalid_argument("CALIBRATION_ENERGY_COMPUTE_FAILED");
    }
    cal.energy_threshold = linear_quantile(id_energies, clean_q);

    // Check consistent feature dimensions for Mahalanobis calibration
    std::size_t feat_dim = 0;
    bool can_maha = dual_output_available && !id_samples.front().feature.empty();
    if (can_maha) {
        feat_dim = id_samples.front().feature.size();
        if (feat_dim < static_cast<std::size_t>(kMinFeatureDim) ||
            feat_dim > static_cast<std::size_t>(kMaxFeatureDim)) {
            can_maha = false;
        }
    }
    if (can_maha) {
        for (const auto& s : id_samples) {
            if (s.feature.size() != feat_dim) {
                can_maha = false;
                break;
            }
            for (double v : s.feature) {
                if (!std::isfinite(v)) {
                    can_maha = false;
                    break;
                }
            }
            if (!can_maha) break;
        }
    }

    std::vector<double> id_mahas;
    if (can_maha) {
        cal.feature_dim = static_cast<int>(feat_dim);
        std::vector<double> global_mean(feat_dim, 0.0);
        std::array<int, 4> cls_counts{0, 0, 0, 0};
        for (int c = 0; c < 4; ++c) {
            cal.class_means[static_cast<std::size_t>(c)].assign(feat_dim, 0.0);
        }

        for (const auto& s : id_samples) {
            const int c = s.class_idx;
            ++cls_counts[static_cast<std::size_t>(c)];
            for (std::size_t i = 0; i < feat_dim; ++i) {
                cal.class_means[static_cast<std::size_t>(c)][i] += s.feature[i];
                global_mean[i] += s.feature[i];
            }
        }
        const double inv_n = 1.0 / static_cast<double>(id_samples.size());
        for (std::size_t i = 0; i < feat_dim; ++i) {
            global_mean[i] *= inv_n;
        }
        for (int c = 0; c < 4; ++c) {
            const int cnt = cls_counts[static_cast<std::size_t>(c)];
            if (cnt > 0) {
                const double inv_c = 1.0 / static_cast<double>(cnt);
                for (std::size_t i = 0; i < feat_dim; ++i) {
                    cal.class_means[static_cast<std::size_t>(c)][i] *= inv_c;
                }
            } else {
                cal.class_means[static_cast<std::size_t>(c)] = global_mean;
            }
        }

        const int d = static_cast<int>(feat_dim);
        cv::Mat cov = cv::Mat::zeros(d, d, CV_64F);
        std::vector<double> diff(feat_dim);
        for (const auto& s : id_samples) {
            const auto& mu = cal.class_means[static_cast<std::size_t>(s.class_idx)];
            for (std::size_t i = 0; i < feat_dim; ++i) {
                diff[i] = s.feature[i] - mu[i];
            }
            for (int r = 0; r < d; ++r) {
                double* crow = cov.ptr<double>(r);
                const double dr = diff[static_cast<std::size_t>(r)];
                for (int c = 0; c < d; ++c) {
                    crow[c] += dr * diff[static_cast<std::size_t>(c)];
                }
            }
        }
        cov *= inv_n;
        for (int i = 0; i < d; ++i) {
            cov.at<double>(i, i) += clean_lambda;
        }

        cv::Mat prec;
        const double inv_ret = cv::invert(cov, prec, cv::DECOMP_CHOLESKY);
        if (inv_ret == 0.0 || prec.empty()) {
            cv::invert(cov, prec, cv::DECOMP_SVD);
        }
        if (!prec.empty()) {
            prec = 0.5 * (prec + prec.t());
            cal.shared_precision_flat.resize(feat_dim * feat_dim);
            bool all_finite = true;
            for (int r = 0; r < d; ++r) {
                const double* prow = prec.ptr<double>(r);
                for (int c = 0; c < d; ++c) {
                    const double v = prow[c];
                    if (!std::isfinite(v)) {
                        all_finite = false;
                        break;
                    }
                    cal.shared_precision_flat[static_cast<std::size_t>(r) * feat_dim + static_cast<std::size_t>(c)] = v;
                }
                if (!all_finite) break;
            }
            if (all_finite) {
                id_mahas.reserve(id_samples.size());
                for (const auto& s : id_samples) {
                    int nearest = -1;
                    double dist = 0.0;
                    if (compute_mahalanobis_distance(s.feature, cal, nearest, dist)) {
                        id_mahas.push_back(dist);
                    }
                }
                if (!id_mahas.empty()) {
                    cal.mahalanobis_threshold = linear_quantile(id_mahas, clean_q);
                } else {
                    cal.feature_dim = 0;
                    for (auto& m : cal.class_means) m.clear();
                    cal.shared_precision_flat.clear();
                }
            } else {
                cal.feature_dim = 0;
                for (auto& m : cal.class_means) m.clear();
                cal.shared_precision_flat.clear();
            }
        } else {
            cal.feature_dim = 0;
            for (auto& m : cal.class_means) m.clear();
            cal.shared_precision_flat.clear();
        }
    }

    if (!ood_samples.empty()) {
        auto compute_std = [](const std::vector<double>& vals) -> double {
            if (vals.size() < 2) return 1.0;
            const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
            const double mean = sum / static_cast<double>(vals.size());
            double sq = 0.0;
            for (double v : vals) sq += (v - mean) * (v - mean);
            const double sd = std::sqrt(sq / static_cast<double>(vals.size()));
            return std::max(1e-4, sd);
        };
        const double e_std = compute_std(id_energies);
        const double m_std = compute_std(id_mahas);
        const bool use_maha = cal.has_mahalanobis_params();

        auto anomaly_score = [&](const CalibrationFeatureSample& s) -> double {
            std::array<double, 4> p{};
            double e = 0.0;
            if (!compute_temperature_softmax_and_energy(s.logits, cal.temperature, p, e)) {
                return 1e6;
            }
            double score = (e - cal.energy_threshold) / e_std;
            if (use_maha && s.feature.size() == static_cast<std::size_t>(cal.feature_dim)) {
                int nearest = -1;
                double md = 0.0;
                if (compute_mahalanobis_distance(s.feature, cal, nearest, md)) {
                    score = std::max(score, (md - cal.mahalanobis_threshold) / m_std);
                }
            }
            return score;
        };

        std::vector<double> id_anom;
        id_anom.reserve(id_samples.size());
        for (const auto& s : id_samples) id_anom.push_back(anomaly_score(s));

        std::vector<double> ood_anom;
        ood_anom.reserve(ood_samples.size());
        for (const auto& s : ood_samples) ood_anom.push_back(anomaly_score(s));

        const auto [auroc, fpr95] = compute_auroc_and_fpr95(id_anom, ood_anom);
        cal.validation_auroc = auroc;
        cal.validation_fpr95 = fpr95;
    }

    return cal;
}

cv::Mat LabelOodClassifier::build_onnx_blob(
    const cv::Mat& crop_bgr,
    int input_width,
    int input_height
) {
    const int safe_w = std::clamp(input_width, 32, 1024);
    const int safe_h = std::clamp(input_height, 32, 1024);

    cv::Mat bgr = crop_bgr;
    if (bgr.channels() == 1) {
        cv::cvtColor(bgr, bgr, cv::COLOR_GRAY2BGR);
    } else if (bgr.channels() == 4) {
        cv::cvtColor(bgr, bgr, cv::COLOR_BGRA2BGR);
    }

    cv::Mat rgb, resized, f32;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(safe_w, safe_h), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    const int dims[4] = {1, 3, safe_h, safe_w};
    cv::Mat blob(4, dims, CV_32F);
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    const int plane_sz = safe_h * safe_w;
    float* bp = blob.ptr<float>();
    for (int y = 0; y < safe_h; ++y) {
        const auto* rp = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < safe_w; ++x) {
            const int off = y * safe_w + x;
            bp[0 * plane_sz + off] = (rp[x][0] - kMean[0]) / kStd[0];
            bp[1 * plane_sz + off] = (rp[x][1] - kMean[1]) / kStd[1];
            bp[2 * plane_sz + off] = (rp[x][2] - kMean[2]) / kStd[2];
        }
    }
    return blob;
}

OnnxProbeInfo LabelOodClassifier::probe_onnx_bytes(
    const std::vector<uint8_t>& bytes,
    int input_width,
    int input_height,
    cv::dnn::Net* out_net
) {
    OnnxProbeInfo info;
    if (bytes.empty()) {
        info.error = "EMPTY_ONNX_MODEL_BYTES";
        return info;
    }

    const int safe_w = std::clamp(input_width, 32, 1024);
    const int safe_h = std::clamp(input_height, 32, 1024);

    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(bytes);
        if (net.empty()) {
            info.error = "MODEL_ARTIFACT_LOAD_FAILED: readNetFromONNX returned empty net";
            return info;
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        const int dims[4] = {1, 3, safe_h, safe_w};
        cv::Mat dummy_blob(4, dims, CV_32F, cv::Scalar(0));

        // 1. Prefer canonical dual-output names ("logits", "penultimate_feat")
        if (net.getLayerId("logits") >= 0 && net.getLayerId("penultimate_feat") >= 0) {
            try {
                std::vector<cv::Mat> outs;
                net.setInput(dummy_blob);
                net.forward(outs, std::vector<std::string>{"logits", "penultimate_feat"});
                std::vector<double> logits_v, feat_v;
                if (outs.size() == 2 &&
                    extract_flat_f32(outs[0], logits_v) &&
                    extract_flat_f32(outs[1], feat_v) &&
                    logits_v.size() >= 4 &&
                    feat_v.size() >= static_cast<std::size_t>(kMinFeatureDim) &&
                    feat_v.size() <= static_cast<std::size_t>(kMaxFeatureDim)) {
                    info.valid = true;
                    info.dual_output = true;
                    info.logits_output_name = "logits";
                    info.feature_output_name = "penultimate_feat";
                    info.logits_dim = static_cast<int>(logits_v.size());
                    info.feature_dim = static_cast<int>(feat_v.size());
                    if (out_net) *out_net = std::move(net);
                    return info;
                }
            } catch (...) {
            }
        }

        // 2. Inspect unconnected graph outputs for >=2 branches or penultimate_feat layer
        const std::vector<std::string> unconnected = net.getUnconnectedOutLayersNames();
        if (unconnected.size() >= 2) {
            try {
                std::vector<cv::Mat> outs;
                net.setInput(dummy_blob);
                net.forward(outs, unconnected);
                if (outs.size() == unconnected.size()) {
                    int logits_idx = -1;
                    int feat_idx = -1;
                    for (std::size_t i = 0; i < outs.size(); ++i) {
                        std::vector<double> vec;
                        if (!extract_flat_f32(outs[i], vec)) continue;
                        if (vec.size() == 4 && logits_idx < 0) {
                            logits_idx = static_cast<int>(i);
                        } else if (vec.size() >= static_cast<std::size_t>(kMinFeatureDim) &&
                                   vec.size() <= static_cast<std::size_t>(kMaxFeatureDim) &&
                                   feat_idx < 0) {
                            feat_idx = static_cast<int>(i);
                        }
                    }
                    if (logits_idx >= 0 && feat_idx >= 0) {
                        info.valid = true;
                        info.dual_output = true;
                        info.logits_output_name = unconnected[static_cast<std::size_t>(logits_idx)];
                        info.feature_output_name = unconnected[static_cast<std::size_t>(feat_idx)];
                        info.logits_dim = static_cast<int>(outs[static_cast<std::size_t>(logits_idx)].total());
                        info.feature_dim = static_cast<int>(outs[static_cast<std::size_t>(feat_idx)].total());
                        if (out_net) *out_net = std::move(net);
                        return info;
                    }
                }
            } catch (...) {
            }
        }

        // 3. Fallback search for layer matching penultimate_feat when single terminal node exists
        if (unconnected.size() == 1) {
            const std::vector<std::string> all_layers = net.getLayerNames();
            std::string matched_feat;
            for (const auto& lname : all_layers) {
                if (lname == unconnected[0]) continue;
                if (lname == "penultimate_feat" ||
                    lname.find("penultimate_feat") != std::string::npos) {
                    matched_feat = lname;
                    break;
                }
            }
            if (!matched_feat.empty()) {
                try {
                    std::vector<cv::Mat> outs;
                    net.setInput(dummy_blob);
                    net.forward(outs, std::vector<std::string>{unconnected[0], matched_feat});
                    std::vector<double> logits_v, feat_v;
                    if (outs.size() == 2 &&
                        extract_flat_f32(outs[0], logits_v) &&
                        extract_flat_f32(outs[1], feat_v) &&
                        logits_v.size() >= 4 &&
                        feat_v.size() >= static_cast<std::size_t>(kMinFeatureDim) &&
                        feat_v.size() <= static_cast<std::size_t>(kMaxFeatureDim)) {
                        info.valid = true;
                        info.dual_output = true;
                        info.logits_output_name = unconnected[0];
                        info.feature_output_name = matched_feat;
                        info.logits_dim = static_cast<int>(logits_v.size());
                        info.feature_dim = static_cast<int>(feat_v.size());
                        if (out_net) *out_net = std::move(net);
                        return info;
                    }
                } catch (...) {
                }
            }
        }

        // 4. Single-output ONNX degradation (Energy-only OOD supported)
        net.setInput(dummy_blob);
        cv::Mat single_out = net.forward();
        std::vector<double> single_vec;
        if (single_out.empty() || single_out.total() < 4 || !extract_flat_f32(single_out, single_vec)) {
            info.error = "MODEL_ARTIFACT_LOAD_FAILED: ONNX output dimension must be >= 4 with finite values";
            return info;
        }
        info.valid = true;
        info.dual_output = false;
        info.logits_output_name = unconnected.empty() ? std::string("logits") : unconnected[0];
        info.feature_output_name.clear();
        info.logits_dim = static_cast<int>(single_vec.size());
        info.feature_dim = 0;
        if (out_net) *out_net = std::move(net);
        return info;
    } catch (const std::exception& ex) {
        info.error = std::string("MODEL_ARTIFACT_LOAD_FAILED: ") + ex.what();
        return info;
    }
}

bool LabelOodClassifier::try_autoload_sidecar_calibration_unlocked(
    const std::filesystem::path& explicit_cal_path,
    const std::filesystem::path& model_path,
    std::string& out_error
) {
    calibration_.reset();
    calibration_path_.clear();
    calibration_error_.clear();

    std::filesystem::path cand_path = explicit_cal_path;
    const bool explicit_provided = !cand_path.empty();
    if (!explicit_provided && !model_path.empty()) {
        cand_path = model_path.parent_path() / L"ood_calibration.json";
    }

    std::error_code ec;
    if (cand_path.empty() || !std::filesystem::is_regular_file(cand_path, ec)) {
        if (explicit_provided) {
            out_error = "OOD_CALIBRATION_NOT_FOUND: " + path_to_utf8(cand_path);
            calibration_error_ = out_error;
            return false;
        }
        return true;
    }

    std::vector<uint8_t> cal_bytes;
    if (!read_file_bytes(cand_path, cal_bytes) || cal_bytes.empty()) {
        calibration_error_ = "OOD_CALIBRATION_READ_FAILED";
        if (explicit_provided) {
            out_error = calibration_error_;
            return false;
        }
        return true;
    }

    json j;
    try {
        j = json::parse(cal_bytes.begin(), cal_bytes.end());
    } catch (...) {
        calibration_error_ = "OOD_CALIBRATION_INVALID_JSON";
        if (explicit_provided) {
            out_error = calibration_error_;
            return false;
        }
        return true;
    }

    std::string parse_err;
    auto parsed = OodCalibration::from_json(j, parse_err);
    if (!parsed.has_value()) {
        calibration_error_ = parse_err;
        if (explicit_provided) {
            out_error = calibration_error_;
            return false;
        }
        return true;
    }

    if (!constant_time_equals(parsed->model_sha256, model_sha256_)) {
        calibration_error_ = "OOD_CALIBRATION_SHA256_MISMATCH";
        if (explicit_provided) {
            out_error = calibration_error_;
            return false;
        }
        return true;
    }

    if (parsed->has_mahalanobis_params() && feature_dim_ > 0 && parsed->feature_dim != feature_dim_) {
        calibration_error_ = "OOD_CALIBRATION_FEATURE_DIM_MISMATCH";
        if (explicit_provided) {
            out_error = calibration_error_;
            return false;
        }
        return true;
    }

    calibration_ = std::move(*parsed);
    calibration_path_ = cand_path;
    return true;
}

bool LabelOodClassifier::load_onnx(
    const std::filesystem::path& onnx_path,
    const std::string& model_version,
    const std::string& expected_sha256,
    int input_width,
    int input_height,
    const std::filesystem::path& calibration_path,
    std::string& out_error
) {
    out_error.clear();
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(onnx_path, bytes) || bytes.empty()) {
        out_error = "MODEL_ARTIFACT_UNAVAILABLE: " + path_to_utf8(onnx_path);
        return false;
    }

    const std::string actual_sha = sha256_hex(bytes);
    if (!expected_sha256.empty()) {
        const std::string exp_norm = to_lower_trim(expected_sha256);
        if (!constant_time_equals(actual_sha, exp_norm)) {
            out_error = "MODEL_SHA256_MISMATCH: expected " + exp_norm + ", got " + actual_sha;
            return false;
        }
    }

    const int safe_w = std::clamp(input_width, 32, 1024);
    const int safe_h = std::clamp(input_height > 0 ? input_height : safe_w, 32, 1024);
    cv::dnn::Net net;
    const OnnxProbeInfo probe = probe_onnx_bytes(bytes, safe_w, safe_h, &net);
    if (!probe.valid) {
        out_error = probe.error.empty() ? "MODEL_ARTIFACT_LOAD_FAILED" : probe.error;
        return false;
    }

    std::lock_guard<std::mutex> lk(mutex_);
    model_kind_ = ModelKind::ONNX_DNN;
    onnx_net_ = std::move(net);
    mlp_model_.release();
    model_version_ = model_version;
    model_path_ = onnx_path;
    model_sha256_ = actual_sha;
    input_width_ = safe_w;
    input_height_ = safe_h;
    dual_output_available_ = probe.dual_output;
    logits_output_name_ = probe.logits_output_name;
    feature_output_name_ = probe.feature_output_name;
    feature_dim_ = probe.feature_dim;
    last_eval_.reset();

    if (!try_autoload_sidecar_calibration_unlocked(calibration_path, onnx_path, out_error)) {
        return false;
    }
    return true;
}

bool LabelOodClassifier::load_mlp(
    const std::filesystem::path& xml_path,
    const std::string& model_version,
    const std::string& expected_sha256,
    const std::filesystem::path& calibration_path,
    std::string& out_error
) {
    out_error.clear();
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(xml_path, bytes) || bytes.empty()) {
        out_error = "MODEL_ARTIFACT_UNAVAILABLE: " + path_to_utf8(xml_path);
        return false;
    }

    const std::string actual_sha = sha256_hex(bytes);
    if (!expected_sha256.empty()) {
        const std::string exp_norm = to_lower_trim(expected_sha256);
        if (!constant_time_equals(actual_sha, exp_norm)) {
            out_error = "MODEL_SHA256_MISMATCH: expected " + exp_norm + ", got " + actual_sha;
            return false;
        }
    }

    const std::string xml_str(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    cv::Ptr<cv::ml::ANN_MLP> mlp;
    try {
        mlp = cv::Algorithm::loadFromString<cv::ml::ANN_MLP>(xml_str);
    } catch (const std::exception& ex) {
        out_error = std::string("MODEL_ARTIFACT_LOAD_FAILED: ") + ex.what();
        return false;
    }
    if (mlp.empty() || !mlp->isTrained()) {
        out_error = "MODEL_ARTIFACT_LOAD_FAILED: ANN_MLP is empty or not trained";
        return false;
    }

    cv::Mat dummy_in = cv::Mat::zeros(1, 64, CV_32F);
    cv::Mat dummy_out;
    mlp->predict(dummy_in, dummy_out);
    if (dummy_out.empty() || dummy_out.total() != 4) {
        out_error = "MODEL_ARTIFACT_LOAD_FAILED: ANN_MLP output dimension must be 4";
        return false;
    }

    std::lock_guard<std::mutex> lk(mutex_);
    model_kind_ = ModelKind::MLP_XML;
    mlp_model_ = mlp;
    onnx_net_ = cv::dnn::Net{};
    model_version_ = model_version;
    model_path_ = xml_path;
    model_sha256_ = actual_sha;
    input_width_ = kDefaultInputSize;
    input_height_ = kDefaultInputSize;
    dual_output_available_ = true;
    logits_output_name_ = "mlp_logits";
    feature_output_name_ = "handcrafted_64d";
    feature_dim_ = 64;
    last_eval_.reset();

    if (!try_autoload_sidecar_calibration_unlocked(calibration_path, xml_path, out_error)) {
        return false;
    }
    return true;
}

void LabelOodClassifier::clear_model() {
    std::lock_guard<std::mutex> lk(mutex_);
    model_kind_ = ModelKind::NONE;
    mlp_model_.release();
    onnx_net_ = cv::dnn::Net{};
    model_version_.clear();
    model_path_.clear();
    model_sha256_.clear();
    input_width_ = kDefaultInputSize;
    input_height_ = kDefaultInputSize;
    dual_output_available_ = false;
    logits_output_name_ = "logits";
    feature_output_name_ = "penultimate_feat";
    feature_dim_ = 0;
    calibration_.reset();
    calibration_path_.clear();
    calibration_error_.clear();
    last_eval_.reset();
}

bool LabelOodClassifier::load_calibration(const OodCalibration& cal, std::string& out_error) {
    out_error.clear();
    std::lock_guard<std::mutex> lk(mutex_);
    if (model_kind_ == ModelKind::NONE || model_sha256_.empty()) {
        out_error = "NO_ACTIVE_NEURAL_MODEL";
        return false;
    }
    const std::string cal_sha = to_lower_trim(cal.model_sha256);
    if (!constant_time_equals(cal_sha, model_sha256_)) {
        out_error = "OOD_CALIBRATION_SHA256_MISMATCH";
        calibration_error_ = out_error;
        return false;
    }
    if (cal.has_mahalanobis_params() && feature_dim_ > 0 && cal.feature_dim != feature_dim_) {
        out_error = "OOD_CALIBRATION_FEATURE_DIM_MISMATCH";
        calibration_error_ = out_error;
        return false;
    }
    OodCalibration copy = cal;
    copy.model_sha256 = cal_sha;
    if (copy.model_version.empty()) {
        copy.model_version = model_version_;
    }
    calibration_ = std::move(copy);
    calibration_error_.clear();
    return true;
}

bool LabelOodClassifier::load_calibration_file(const std::filesystem::path& path, std::string& out_error) {
    out_error.clear();
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, bytes) || bytes.empty()) {
        out_error = "OOD_CALIBRATION_NOT_FOUND";
        return false;
    }
    json j;
    try {
        j = json::parse(bytes.begin(), bytes.end());
    } catch (...) {
        out_error = "OOD_CALIBRATION_INVALID_JSON";
        return false;
    }
    auto parsed = OodCalibration::from_json(j, out_error);
    if (!parsed.has_value()) {
        return false;
    }
    if (!load_calibration(*parsed, out_error)) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    calibration_path_ = path;
    return true;
}

void LabelOodClassifier::clear_calibration() {
    std::lock_guard<std::mutex> lk(mutex_);
    calibration_.reset();
    calibration_path_.clear();
    calibration_error_.clear();
}

void LabelOodClassifier::set_gate_config(const OodGateConfig& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    gate_config_ = cfg;
    gate_config_.alarm_streak_threshold = std::clamp(gate_config_.alarm_streak_threshold, 1, 30);
    if (gate_config_.prob_threshold_override.has_value()) {
        *gate_config_.prob_threshold_override = std::clamp(*gate_config_.prob_threshold_override, 0.0, 1.0);
    }
    if (gate_config_.mahalanobis_threshold_override.has_value()) {
        *gate_config_.mahalanobis_threshold_override = std::max(0.0, *gate_config_.mahalanobis_threshold_override);
    }
}

OodGateConfig LabelOodClassifier::gate_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return gate_config_;
}

bool LabelOodClassifier::has_neural_model() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return model_kind_ != ModelKind::NONE;
}

LabelOodClassifier::ModelKind LabelOodClassifier::model_kind() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return model_kind_;
}

std::string LabelOodClassifier::active_model_version() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return model_version_;
}

std::string LabelOodClassifier::active_model_sha256() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return model_sha256_;
}

std::filesystem::path LabelOodClassifier::active_model_path() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return model_path_;
}

int LabelOodClassifier::input_width() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return input_width_;
}

int LabelOodClassifier::input_height() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return input_height_;
}

bool LabelOodClassifier::dual_output_available() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return dual_output_available_;
}

std::optional<OodCalibration> LabelOodClassifier::active_calibration() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return calibration_;
}

bool LabelOodClassifier::forward_unlocked(
    const cv::Mat& crop_bgr,
    std::array<double, 4>& out_logits,
    std::vector<double>& out_feature,
    std::string& out_error
) {
    out_error.clear();
    out_feature.clear();

    if (crop_bgr.empty()) {
        out_error = "EMPTY_CROP";
        return false;
    }

    if (model_kind_ == ModelKind::MLP_XML && !mlp_model_.empty()) {
        try {
            const cv::Mat feat_mat = extract_label_features(crop_bgr);
            if (!extract_flat_f32(feat_mat, out_feature)) {
                out_error = "OOD_NON_FINITE";
                return false;
            }
            cv::Mat raw_out;
            mlp_model_->predict(feat_mat, raw_out);
            std::vector<double> raw_vec;
            if (!extract_flat_f32(raw_out, raw_vec) || raw_vec.size() < 4) {
                out_error = "OOD_NON_FINITE";
                return false;
            }
            for (std::size_t i = 0; i < 4; ++i) {
                out_logits[i] = raw_vec[i] * kMlpLogitScale;
            }
            return true;
        } catch (...) {
            out_error = "MLP_FORWARD_FAILED";
            return false;
        }
    }

    if (model_kind_ == ModelKind::ONNX_DNN && !onnx_net_.empty()) {
        try {
            cv::Mat blob = build_onnx_blob(crop_bgr, input_width_, input_height_);
            onnx_net_.setInput(blob);
            if (dual_output_available_ && !feature_output_name_.empty()) {
                std::vector<cv::Mat> outs;
                onnx_net_.forward(outs, std::vector<std::string>{logits_output_name_, feature_output_name_});
                if (outs.size() != 2) {
                    out_error = "ONNX_FORWARD_FAILED";
                    return false;
                }
                std::vector<double> l_vec;
                if (!extract_flat_f32(outs[0], l_vec) || l_vec.size() < 4) {
                    out_error = "OOD_NON_FINITE";
                    return false;
                }
                if (!extract_flat_f32(outs[1], out_feature)) {
                    out_error = "OOD_NON_FINITE";
                    return false;
                }
                for (std::size_t i = 0; i < 4; ++i) {
                    out_logits[i] = l_vec[i];
                }
                return true;
            }

            cv::Mat logits_mat = onnx_net_.forward();
            std::vector<double> l_vec;
            if (!extract_flat_f32(logits_mat, l_vec) || l_vec.size() < 4) {
                out_error = "OOD_NON_FINITE";
                return false;
            }
            for (std::size_t i = 0; i < 4; ++i) {
                out_logits[i] = l_vec[i];
            }
            return true;
        } catch (...) {
            out_error = "ONNX_FORWARD_FAILED";
            return false;
        }
    }

    out_error = "NO_ACTIVE_NEURAL_MODEL";
    return false;
}

LabelOodEvalResult LabelOodClassifier::evaluate(const cv::Mat& crop_bgr, bool capture_feature) {
    const double t0 = monotonic_seconds();
    std::lock_guard<std::mutex> lk(mutex_);

    LabelOodEvalResult res;
    if (model_kind_ == ModelKind::NONE) {
        res.backend_mode = "heuristic_fallback";
        res.infer_ms = (monotonic_seconds() - t0) * 1000.0;
        return res;
    }

    res.model_version = model_version_;
    res.model_sha256 = model_sha256_;

    const bool has_cal = calibration_.has_value() && calibration_->enabled;
    const bool use_maha = has_cal && dual_output_available_ && calibration_->has_mahalanobis_params();

    if (model_kind_ == ModelKind::ONNX_DNN) {
        if (use_maha) {
            res.backend_mode = "onnx_dual_ood";
        } else if (has_cal) {
            res.backend_mode = "onnx_energy_only";
        } else {
            res.backend_mode = "onnx_uncalibrated";
        }
    } else {
        res.backend_mode = has_cal ? "mlp_ood" : "mlp_uncalibrated";
    }

    std::array<double, 4> logits{};
    std::vector<double> feat;
    std::string fwd_err;
    if (!forward_unlocked(crop_bgr, logits, feat, fwd_err)) {
        res.valid = false;
        res.ood_rejected = (fwd_err == "OOD_NON_FINITE");
        res.reject_reason = (fwd_err == "OOD_NON_FINITE") ? "OOD_NON_FINITE" : "";
        res.final_label_class = res.ood_rejected ? "ood_unknown" : "unknown";
        res.infer_ms = (monotonic_seconds() - t0) * 1000.0;
        last_eval_ = res;
        return res;
    }

    res.logits = logits;
    if (capture_feature && !feat.empty()) {
        res.feature_vec.resize(feat.size());
        for (std::size_t i = 0; i < feat.size(); ++i) {
            res.feature_vec[i] = static_cast<float>(feat[i]);
        }
    }

    const double temp = has_cal ? calibration_->temperature : 1.0;
    std::array<double, 4> probs{};
    double energy = 0.0;
    if (!compute_temperature_softmax_and_energy(logits, temp, probs, energy)) {
        res.valid = false;
        res.ood_rejected = true;
        res.reject_reason = "OOD_NON_FINITE";
        res.final_label_class = "ood_unknown";
        res.infer_ms = (monotonic_seconds() - t0) * 1000.0;
        last_eval_ = res;
        return res;
    }

    res.valid = true;
    res.calibrated_probs = probs;
    res.energy_score = energy;

    int best_idx = 0;
    for (int i = 1; i < 4; ++i) {
        if (probs[static_cast<std::size_t>(i)] > probs[static_cast<std::size_t>(best_idx)]) {
            best_idx = i;
        }
    }
    res.max_prob = probs[static_cast<std::size_t>(best_idx)];
    res.raw_predicted_class = kLabelClasses[static_cast<std::size_t>(best_idx)];
    res.final_label_class = res.raw_predicted_class;

    if (use_maha && feat.size() == static_cast<std::size_t>(calibration_->feature_dim)) {
        int nearest_cls = -1;
        double md = 0.0;
        if (compute_mahalanobis_distance(feat, *calibration_, nearest_cls, md)) {
            res.mahalanobis_distance = md;
            res.nearest_class_idx = nearest_cls;
        }
    }

    if (gate_config_.gate_enabled) {
        if (!has_cal) {
            if (model_kind_ == ModelKind::ONNX_DNN && gate_config_.require_calibration_for_onnx) {
                res.ood_rejected = true;
                res.reject_reason = calibration_error_.empty() ? "OOD_UNCALIBRATED" : calibration_error_;
                res.final_label_class = "ood_unknown";
            }
        } else {
            const double eff_prob = gate_config_.prob_threshold_override.value_or(calibration_->prob_threshold);
            const double eff_energy = gate_config_.energy_threshold_override.value_or(calibration_->energy_threshold);
            const double eff_maha = gate_config_.mahalanobis_threshold_override.value_or(calibration_->mahalanobis_threshold);

            const bool prob_low = res.max_prob < eff_prob;
            const bool energy_high = res.energy_score > eff_energy;
            const bool maha_high = res.mahalanobis_distance.has_value() && (*res.mahalanobis_distance > eff_maha);

            if (energy_high && maha_high) {
                res.ood_rejected = true;
                res.reject_reason = "OOD_DUAL_EXCEEDED";
            } else if (maha_high) {
                res.ood_rejected = true;
                res.reject_reason = "OOD_MAHALANOBIS_HIGH";
            } else if (energy_high) {
                res.ood_rejected = true;
                res.reject_reason = "OOD_ENERGY_HIGH";
            } else if (prob_low) {
                res.ood_rejected = true;
                res.reject_reason = "OOD_PROB_LOW";
            }
            if (res.ood_rejected) {
                res.final_label_class = "ood_unknown";
            }
        }
    }

    res.infer_ms = (monotonic_seconds() - t0) * 1000.0;
    last_eval_ = res;
    return res;
}

OodCalibration LabelOodClassifier::calibrate_from_crops(
    const std::vector<std::pair<cv::Mat, int>>& labeled_crops,
    double quantile,
    double prob_threshold,
    double reg_lambda,
    bool fit_temperature
) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (model_kind_ == ModelKind::NONE) {
        throw std::runtime_error("NO_ACTIVE_NEURAL_MODEL");
    }

    std::vector<CalibrationFeatureSample> samples;
    samples.reserve(labeled_crops.size());
    for (const auto& [crop, cls] : labeled_crops) {
        if (crop.empty()) continue;
        std::array<double, 4> logits{};
        std::vector<double> feat;
        std::string err;
        if (!forward_unlocked(crop, logits, feat, err)) {
            continue;
        }
        CalibrationFeatureSample s;
        s.class_idx = cls;
        s.logits = logits;
        s.feature = std::move(feat);
        samples.push_back(std::move(s));
    }

    OodCalibration cal = calibrate_ood_from_samples(
        samples,
        model_version_,
        model_sha256_,
        input_width_,
        input_height_,
        dual_output_available_,
        logits_output_name_,
        feature_output_name_,
        quantile,
        prob_threshold,
        reg_lambda,
        fit_temperature
    );
    calibration_ = cal;
    calibration_error_.clear();
    return cal;
}

json LabelOodClassifier::status_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::string kind_str = "heuristic";
    if (model_kind_ == ModelKind::ONNX_DNN) kind_str = "onnx";
    else if (model_kind_ == ModelKind::MLP_XML) kind_str = "opencv_mlp";

    const bool has_cal = calibration_.has_value() && calibration_->enabled;
    const bool use_maha = has_cal && dual_output_available_ && calibration_->has_mahalanobis_params();
    std::string mode_str = "heuristic_fallback";
    if (model_kind_ == ModelKind::ONNX_DNN) {
        mode_str = use_maha ? "onnx_dual_ood" : (has_cal ? "onnx_energy_only" : "onnx_uncalibrated");
    } else if (model_kind_ == ModelKind::MLP_XML) {
        mode_str = has_cal ? "mlp_ood" : "mlp_uncalibrated";
    }

    json out{
        {"model_kind", kind_str},
        {"backend_mode", mode_str},
        {"model_version", model_version_},
        {"model_sha256", model_sha256_},
        {"input_width", input_width_},
        {"input_height", input_height_},
        {"dual_output_available", dual_output_available_},
        {"logits_output_name", logits_output_name_},
        {"feature_output_name", feature_output_name_},
        {"feature_dim", feature_dim_},
        {"calibrated", calibration_.has_value()},
        {"calibration", calibration_.has_value() ? calibration_->to_json(false) : json(nullptr)},
        {"calibration_path", path_to_utf8(calibration_path_)},
        {"calibration_error", calibration_error_},
        {"gate_config", gate_config_.to_json()},
        {"last_eval", last_eval_.has_value() ? last_eval_->to_json() : json(nullptr)},
    };
    return out;
}

}  // namespace igbt::vision
