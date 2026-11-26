#include <igbt/vision/hand_glove_detector.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace igbt::vision {

namespace {

double round3(double v) {
    return std::round(v * 1000.0) / 1000.0;
}

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

double box_overlap_fraction(const Box& box, const Box& region) {
    const double area = std::max(0.0, box[2] - box[0]) * std::max(0.0, box[3] - box[1]);
    if (area <= 1e-12) return 0.0;
    const double ix1 = std::max(box[0], region[0]);
    const double iy1 = std::max(box[1], region[1]);
    const double ix2 = std::min(box[2], region[2]);
    const double iy2 = std::min(box[3], region[3]);
    if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
    return ((ix2 - ix1) * (iy2 - iy1)) / area;
}

cv::Rect norm_box_to_rect(const Box& b, int w, int h) {
    const int x1 = std::clamp(static_cast<int>(std::floor(b[0] * w)), 0, std::max(0, w - 1));
    const int y1 = std::clamp(static_cast<int>(std::floor(b[1] * h)), 0, std::max(0, h - 1));
    const int x2 = std::clamp(static_cast<int>(std::ceil(b[2] * w)), x1 + 1, std::max(1, w));
    const int y2 = std::clamp(static_cast<int>(std::ceil(b[3] * h)), y1 + 1, std::max(1, h));
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

std::vector<int> resolve_strides_for_grid(int num_cells, int input_size, const std::vector<int>& cfg_strides) {
    auto sum_cells = [&](const std::vector<int>& st) {
        int total = 0;
        for (int s : st) {
            if (s <= 0 || input_size % s != 0) return -1;
            const int g = input_size / s;
            total += g * g;
        }
        return total;
    };
    if (!cfg_strides.empty() && sum_cells(cfg_strides) == num_cells) return cfg_strides;
    const std::vector<int> s4{8, 16, 32, 64};
    if (sum_cells(s4) == num_cells) return s4;
    const std::vector<int> s3{8, 16, 32};
    if (sum_cells(s3) == num_cells) return s3;
    return {};
}

}  // namespace

std::string hand_glove_class_name(HandGloveClass cls) {
    switch (cls) {
        case HandGloveClass::BARE_HAND: return "bare_hand";
        case HandGloveClass::ESD_WHITE: return "esd_white";
        case HandGloveClass::ESD_GRAY: return "esd_gray";
        case HandGloveClass::ESD_BLACK: return "esd_black";
    }
    return "bare_hand";
}

bool is_glove_class(HandGloveClass cls) {
    return cls == HandGloveClass::ESD_WHITE ||
           cls == HandGloveClass::ESD_GRAY ||
           cls == HandGloveClass::ESD_BLACK;
}

json HandGloveDetection::to_json() const {
    return json{
        {"class_name", class_name},
        {"score", round3(score)},
        {"box", {round3(box[0]), round3(box[1]), round3(box[2]), round3(box[3])}},
        {"glare_warning", glare_warning},
        {"wrist_skin_flag", wrist_skin_flag},
        {"texture_gradient", round3(texture_gradient)},
        {"source", source},
    };
}

AsymmetricBareHandHysteresis::AsymmetricBareHandHysteresis(int enter_frames, int exit_frames)
    : enter_frames_(std::max(1, enter_frames)),
      exit_frames_(std::max(1, exit_frames)) {}

void AsymmetricBareHandHysteresis::reset() {
    state_ = HysteresisState{};
}

void AsymmetricBareHandHysteresis::configure(int enter_frames, int exit_frames) {
    enter_frames_ = std::max(1, enter_frames);
    exit_frames_ = std::max(1, exit_frames);
}

HysteresisState AsymmetricBareHandHysteresis::update(bool raw_bare_detected, bool raw_bare_touching_tube) {
    if (raw_bare_detected) {
        ++state_.bare_enter_streak;
        state_.bare_clear_streak = 0;
        if (state_.bare_enter_streak >= enter_frames_) {
            state_.bare_hand_active = true;
        }
    } else {
        state_.bare_enter_streak = 0;
        if (state_.bare_hand_active) {
            ++state_.bare_clear_streak;
            if (state_.bare_clear_streak >= exit_frames_) {
                state_.bare_hand_active = false;
                state_.bare_clear_streak = 0;
                state_.bare_hand_violation = false;
            }
        }
    }

    if (raw_bare_touching_tube) {
        ++state_.touch_enter_streak;
        if (state_.touch_enter_streak >= enter_frames_) {
            state_.bare_hand_violation = true;
            state_.bare_hand_active = true;
            state_.bare_clear_streak = 0;
        }
    } else {
        state_.touch_enter_streak = 0;
    }

    return state_;
}

HandGloveDetector::HandGloveDetector(const HandGloveConfig& config)
    : config_(config),
      hysteresis_(config.bare_hand_enter_frames, config.bare_hand_exit_frames) {}

void HandGloveDetector::update_config(const HandGloveConfig& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    hysteresis_.configure(config_.bare_hand_enter_frames, config_.bare_hand_exit_frames);
}

const HandGloveConfig& HandGloveDetector::config() const {
    return config_;
}

void HandGloveDetector::set_frame_stride(int stride) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_.frame_stride = std::clamp(stride, 1, 8);
}

void HandGloveDetector::reset_state() {
    std::lock_guard<std::mutex> lk(mutex_);
    frame_counter_ = 0;
    cached_onnx_dets_.clear();
    last_infer_ms_ = 0.0;
    hysteresis_.reset();
}

bool HandGloveDetector::load_onnx(
    const std::filesystem::path& onnx_path,
    const std::string& model_version,
    const std::string& expected_sha256,
    std::string& out_error
) {
    out_error.clear();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(onnx_path, ec)) {
        out_error = "HAND_GLOVE_ONNX_NOT_FOUND";
        std::lock_guard<std::mutex> lk(mutex_);
        last_error_ = out_error;
        return false;
    }

    std::vector<uint8_t> raw_bytes;
    if (!read_file_bytes(onnx_path, raw_bytes) || raw_bytes.empty()) {
        out_error = "HAND_GLOVE_ONNX_READ_FAILED";
        std::lock_guard<std::mutex> lk(mutex_);
        last_error_ = out_error;
        return false;
    }

    const std::string actual_sha = sha256_hex(raw_bytes);
    const std::string expected_norm = to_lower_copy(expected_sha256);
    if (!expected_norm.empty() && !constant_time_equals(actual_sha, expected_norm)) {
        out_error = "HAND_GLOVE_ONNX_SHA256_MISMATCH";
        std::lock_guard<std::mutex> lk(mutex_);
        last_error_ = out_error;
        return false;
    }

    try {
        cv::dnn::Net loaded = cv::dnn::readNetFromONNX(raw_bytes);
        if (loaded.empty()) {
            out_error = "HAND_GLOVE_ONNX_EMPTY_NET";
            std::lock_guard<std::mutex> lk(mutex_);
            last_error_ = out_error;
            return false;
        }
        loaded.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        loaded.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        const int in_sz = config_.input_size;
        const int dims[4] = {1, 3, in_sz, in_sz};
        cv::Mat dummy = cv::Mat::zeros(4, dims, CV_32F);
        loaded.setInput(dummy);
        cv::Mat out = loaded.forward();
        if (out.empty()) {
            out_error = "HAND_GLOVE_ONNX_EMPTY_OUTPUT";
            std::lock_guard<std::mutex> lk(mutex_);
            last_error_ = out_error;
            return false;
        }
        const int rows = (out.dims == 3) ? out.size[1] : out.rows;
        const int cols = (out.dims == 3) ? out.size[2] : out.cols;
        const int expected_cols = 4 + 4 * (config_.reg_max + 1);
        if (cols != expected_cols || resolve_strides_for_grid(rows, in_sz, config_.strides).empty()) {
            out_error = "HAND_GLOVE_ONNX_SHAPE_MISMATCH";
            std::lock_guard<std::mutex> lk(mutex_);
            last_error_ = out_error;
            return false;
        }

        std::lock_guard<std::mutex> lk(mutex_);
        net_ = std::move(loaded);
        onnx_loaded_ = true;
        model_version_ = model_version.empty() ? "nanodet-plus-m-v1" : model_version;
        model_path_ = path_to_utf8(onnx_path);
        model_sha256_ = actual_sha;
        last_error_.clear();
        cached_onnx_dets_.clear();
        frame_counter_ = 0;
        return true;
    } catch (const std::exception& exc) {
        out_error = std::string("HAND_GLOVE_ONNX_LOAD_FAILED:") + exc.what();
        std::lock_guard<std::mutex> lk(mutex_);
        last_error_ = out_error;
        return false;
    }
}

void HandGloveDetector::clear_onnx() {
    std::lock_guard<std::mutex> lk(mutex_);
    onnx_loaded_ = false;
    net_ = cv::dnn::Net{};
    model_version_.clear();
    model_path_.clear();
    model_sha256_.clear();
    cached_onnx_dets_.clear();
}

bool HandGloveDetector::is_onnx_loaded() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return onnx_loaded_;
}

json HandGloveDetector::status_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return json{
        {"backend", onnx_loaded_ ? "nanodet_plus_onnx" : "heuristic_fallback"},
        {"onnx_loaded", onnx_loaded_},
        {"model_version", model_version_.empty() ? json(nullptr) : json(model_version_)},
        {"model_path", model_path_.empty() ? json(nullptr) : json(model_path_)},
        {"model_sha256", model_sha256_.empty() ? json(nullptr) : json(model_sha256_)},
        {"input_size", config_.input_size},
        {"frame_stride", config_.frame_stride},
        {"last_infer_ms", round3(last_infer_ms_)},
        {"last_error", last_error_.empty() ? json(nullptr) : json(last_error_)},
        {"hysteresis", {
            {"enter_frames", config_.bare_hand_enter_frames},
            {"exit_frames", config_.bare_hand_exit_frames},
            {"bare_hand_active", hysteresis_.state().bare_hand_active},
            {"bare_hand_violation", hysteresis_.state().bare_hand_violation},
            {"bare_enter_streak", hysteresis_.state().bare_enter_streak},
            {"bare_clear_streak", hysteresis_.state().bare_clear_streak},
            {"touch_enter_streak", hysteresis_.state().touch_enter_streak},
        }},
    };
}

std::vector<HandGloveDetection> HandGloveDetector::decode_nanodet_plus_output(
    const cv::Mat& raw_output,
    int orig_w,
    int orig_h,
    float letterbox_scale,
    int pad_x,
    int pad_y,
    const HandGloveConfig& cfg
) {
    if (raw_output.empty() || orig_w <= 0 || orig_h <= 0 || letterbox_scale <= 1e-6f) {
        return {};
    }

    constexpr int kNumClasses = 4;
    const int bins = cfg.reg_max + 1;
    const int expected_cols = kNumClasses + 4 * bins;

    cv::Mat f32;
    raw_output.convertTo(f32, CV_32F);
    int rows = 0;
    int cols = 0;
    if (f32.dims == 3 && f32.size[0] == 1) {
        rows = f32.size[1];
        cols = f32.size[2];
        f32 = f32.reshape(1, rows);
    } else if (f32.dims == 2) {
        rows = f32.rows;
        cols = f32.cols;
    } else {
        return {};
    }
    if (cols != expected_cols || rows <= 0) return {};

    const std::vector<int> strides = resolve_strides_for_grid(rows, cfg.input_size, cfg.strides);
    if (strides.empty()) return {};

    bool need_sigmoid = false;
    for (int r = 0; r < rows && !need_sigmoid; ++r) {
        const float* p = f32.ptr<float>(r);
        for (int c = 0; c < kNumClasses; ++c) {
            if (p[c] < -1e-4f || p[c] > 1.0001f) {
                need_sigmoid = true;
                break;
            }
        }
    }

    std::array<std::vector<cv::Rect2d>, kNumClasses> cls_rects;
    std::array<std::vector<float>, kNumClasses> cls_scores;
    std::array<std::vector<Box>, kNumClasses> cls_norm_boxes;

    int row_idx = 0;
    for (int stride : strides) {
        const int grid = cfg.input_size / stride;
        for (int gy = 0; gy < grid; ++gy) {
            for (int gx = 0; gx < grid; ++gx) {
                const float* row = f32.ptr<float>(row_idx++);
                int best_cls = 0;
                double best_score = -1.0;
                for (int c = 0; c < kNumClasses; ++c) {
                    const float raw_v = row[c];
                    const double s = need_sigmoid
                                         ? (1.0 / (1.0 + std::exp(-static_cast<double>(raw_v))))
                                         : std::clamp(static_cast<double>(raw_v), 0.0, 1.0);
                    if (s > best_score) {
                        best_score = s;
                        best_cls = c;
                    }
                }
                const double min_thr = (best_cls == 0)
                                           ? std::max(cfg.score_threshold, cfg.bare_hand_score_threshold)
                                           : cfg.score_threshold;
                if (best_score < min_thr) continue;

                // Decode 4 GFL edge distributions (left, top, right, bottom)
                std::array<double, 4> dist{};
                for (int e = 0; e < 4; ++e) {
                    const float* logits = row + kNumClasses + e * bins;
                    float m = logits[0];
                    for (int k = 1; k < bins; ++k) m = std::max(m, logits[k]);
                    double sum_exp = 0.0;
                    double exp_acc = 0.0;
                    for (int k = 0; k < bins; ++k) {
                        const double w = std::exp(static_cast<double>(logits[k] - m));
                        sum_exp += w;
                        exp_acc += w * static_cast<double>(k);
                    }
                    dist[e] = (sum_exp > 0.0 ? (exp_acc / sum_exp) : 0.0) * static_cast<double>(stride);
                }

                const double cx_net = static_cast<double>(gx * stride);
                const double cy_net = static_cast<double>(gy * stride);
                const double x1_orig = (cx_net - dist[0] - pad_x) / letterbox_scale;
                const double y1_orig = (cy_net - dist[1] - pad_y) / letterbox_scale;
                const double x2_orig = (cx_net + dist[2] - pad_x) / letterbox_scale;
                const double y2_orig = (cy_net + dist[3] - pad_y) / letterbox_scale;

                const double nx1 = std::clamp(x1_orig / orig_w, 0.0, 1.0);
                const double ny1 = std::clamp(y1_orig / orig_h, 0.0, 1.0);
                const double nx2 = std::clamp(x2_orig / orig_w, 0.0, 1.0);
                const double ny2 = std::clamp(y2_orig / orig_h, 0.0, 1.0);
                if (nx2 - nx1 < 0.02 || ny2 - ny1 < 0.02) continue;

                const double cy_norm = 0.5 * (ny1 + ny2);
                if (best_cls == 0 && (cy_norm < cfg.work_zone_cy_min || cy_norm > cfg.work_zone_cy_max)) {
                    continue;
                }

                cls_rects[best_cls].emplace_back(nx1 * orig_w, ny1 * orig_h, (nx2 - nx1) * orig_w, (ny2 - ny1) * orig_h);
                cls_scores[best_cls].push_back(static_cast<float>(best_score));
                cls_norm_boxes[best_cls].push_back(Box{nx1, ny1, nx2, ny2});
            }
        }
    }

    std::vector<HandGloveDetection> results;
    for (int c = 0; c < kNumClasses; ++c) {
        if (cls_rects[c].empty()) continue;
        std::vector<int> keep;
        const float thr = static_cast<float>(c == 0 ? std::max(cfg.score_threshold, cfg.bare_hand_score_threshold)
                                                    : cfg.score_threshold);
        cv::dnn::NMSBoxes(cls_rects[c], cls_scores[c], thr, static_cast<float>(cfg.nms_iou_threshold), keep);
        for (int idx : keep) {
            HandGloveDetection det{};
            det.cls = static_cast<HandGloveClass>(c);
            det.class_name = hand_glove_class_name(det.cls);
            det.score = static_cast<double>(cls_scores[c][idx]);
            det.box = cls_norm_boxes[c][idx];
            det.source = "nanodet_plus_onnx";
            results.push_back(std::move(det));
        }
    }
    return results;
}

void HandGloveDetector::filter_label_isolation(
    std::vector<HandGloveDetection>& dets,
    const std::optional<Box>& label_box
) {
    if (!label_box.has_value()) return;
    dets.erase(
        std::remove_if(dets.begin(), dets.end(), [&](const HandGloveDetection& d) {
            return d.cls == HandGloveClass::ESD_WHITE && box_overlap_fraction(d.box, *label_box) > 0.72;
        }),
        dets.end()
    );
}

void HandGloveDetector::apply_physical_guardrails(
    const cv::Mat& frame_bgr,
    std::vector<HandGloveDetection>& dets,
    const std::optional<Box>&,
    const std::optional<Box>& label_box,
    const HandGloveConfig& cfg,
    bool& out_white_glare_warning
) {
    out_white_glare_warning = false;
    if (frame_bgr.empty() || dets.empty()) return;

    filter_label_isolation(dets, label_box);

    std::vector<HandGloveDetection> kept;
    kept.reserve(dets.size());
    for (auto& d : dets) {
        const cv::Rect rect = norm_box_to_rect(d.box, frame_bgr.cols, frame_bgr.rows);
        if (rect.width < 4 || rect.height < 4) continue;
        cv::Mat patch = frame_bgr(rect);

        if (d.cls == HandGloveClass::ESD_WHITE) {
            cv::Mat hsv, ycrcb;
            cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
            cv::cvtColor(patch, ycrcb, cv::COLOR_BGR2YCrCb);

            int v_over = 0;
            int spec_px = 0;
            double cr_sum = 0.0;
            const int total = patch.rows * patch.cols;
            for (int y = 0; y < patch.rows; ++y) {
                const auto* hp = hsv.ptr<cv::Vec3b>(y);
                const auto* yp = ycrcb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < patch.cols; ++x) {
                    if (hp[x][2] >= 245) ++v_over;
                    if (hp[x][2] >= 250 && hp[x][1] <= 15) ++spec_px;
                    cr_sum += yp[x][1];
                }
            }
            const double over_frac = static_cast<double>(v_over) / std::max(1, total);
            const double spec_frac = static_cast<double>(spec_px) / std::max(1, total);
            if (spec_frac > 0.55) {
                continue;  // Pure clipped specular reflection
            }
            if (over_frac > cfg.white_glove_overexp_thresh) {
                d.glare_warning = true;
                out_white_glare_warning = true;
            }

            // Wrist transition strip skin check (bottom 18% of glove bounding box)
            const int wy0 = std::max(0, static_cast<int>(patch.rows * 0.82));
            int wrist_skin = 0;
            int wrist_total = 0;
            for (int y = wy0; y < patch.rows; ++y) {
                const auto* hp = hsv.ptr<cv::Vec3b>(y);
                const auto* yp = ycrcb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < patch.cols; ++x) {
                    ++wrist_total;
                    if (hp[x][2] >= 120 && yp[x][1] >= 137 && yp[x][1] <= 174 &&
                        yp[x][2] >= 80 && yp[x][2] <= 122) {
                        ++wrist_skin;
                    }
                }
            }
            const double wrist_skin_frac = static_cast<double>(wrist_skin) / std::max(1, wrist_total);
            if (wrist_skin_frac > 0.28 && (cr_sum / std::max(1, total)) > 135.0) {
                d.wrist_skin_flag = true;
            }
        } else if (d.cls == HandGloveClass::ESD_BLACK) {
            cv::Mat hsv, gray, gx, gy, mag;
            cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
            cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
            cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
            cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
            cv::magnitude(gx, gy, mag);
            d.texture_gradient = cv::mean(mag)[0];
            const double mean_v = cv::mean(hsv)[2];
            if (mean_v <= 68.0 && d.texture_gradient < cfg.black_glove_min_texture_grad) {
                continue;  // Flat desktop shadow rejection
            }
        }

        kept.push_back(std::move(d));
    }
    dets = std::move(kept);
}

void HandGloveDetector::summarize_detections(HandGloveInferResult& result) {
    result.left_glove.reset();
    result.right_glove.reset();
    result.bare_hand.reset();
    result.left_glove_class.clear();
    result.right_glove_class.clear();

    double best_lg = -1.0;
    double best_rg = -1.0;
    double best_bh = -1.0;

    for (const auto& d : result.detections) {
        const double area = std::max(0.0, d.box[2] - d.box[0]) * std::max(0.0, d.box[3] - d.box[1]);
        const double cx = 0.5 * (d.box[0] + d.box[2]);
        const double weight = d.score * 10.0 + area * 100.0;
        if (d.cls == HandGloveClass::BARE_HAND) {
            if (weight > best_bh) {
                best_bh = weight;
                result.bare_hand = d.box;
            }
        } else if (is_glove_class(d.cls)) {
            if (cx < 0.50 && weight > best_lg) {
                best_lg = weight;
                result.left_glove = d.box;
                result.left_glove_class = d.class_name;
            } else if (cx >= 0.50 && weight > best_rg) {
                best_rg = weight;
                result.right_glove = d.box;
                result.right_glove_class = d.class_name;
            }
        }
    }
}

HandGloveInferResult HandGloveDetector::detect_candidates(
    const cv::Mat& frame_bgr,
    const cv::Mat& hsv_small,
    const cv::Mat& skin_mask_small,
    const cv::Mat& fg_diff,
    const cv::Mat& motion_diff,
    const cv::Mat& person_exclusion,
    const cv::Mat& specular_mask,
    const cv::Mat& clipped_mask
) {
    const double t0 = monotonic_seconds();
    HandGloveInferResult out{};
    if (frame_bgr.empty()) return out;

    HandGloveConfig cfg_copy;
    bool use_onnx = false;
    int stride_idx = 0;
    std::string ver_copy;
    cv::dnn::Net net_copy;
    std::vector<HandGloveDetection> cached_copy;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;
        use_onnx = onnx_loaded_ && !net_.empty();
        ver_copy = model_version_;
        stride_idx = frame_counter_++;
        if (use_onnx) {
            if (cfg_copy.frame_stride > 1 && (stride_idx % cfg_copy.frame_stride) != 0 && !cached_onnx_dets_.empty()) {
                out.detections = cached_onnx_dets_;
                out.reused_stride_cache = true;
                out.backend = "nanodet_plus_onnx";
                out.model_version = ver_copy;
                out.infer_ms = round3(last_infer_ms_);
                apply_physical_guardrails(frame_bgr, out.detections, std::nullopt, std::nullopt, cfg_copy, out.white_glove_glare_warning);
                return out;
            }
            net_copy = net_;
        }
    }

    if (use_onnx) {
        const int sz = cfg_copy.input_size;
        const float scale = std::min(static_cast<float>(sz) / frame_bgr.cols, static_cast<float>(sz) / frame_bgr.rows);
        const int nw = std::max(1, static_cast<int>(std::round(frame_bgr.cols * scale)));
        const int nh = std::max(1, static_cast<int>(std::round(frame_bgr.rows * scale)));
        const int pad_x = (sz - nw) / 2;
        const int pad_y = (sz - nh) / 2;

        cv::Mat resized;
        cv::resize(frame_bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        cv::Mat canvas(sz, sz, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect(pad_x, pad_y, nw, nh)));

        // Official NanoDet BGR normalization: mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]
        cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 57.63, cv::Size(sz, sz), cv::Scalar(103.53, 116.28, 123.675), false, false, CV_32F);
        net_copy.setInput(blob);
        cv::Mat raw_out = net_copy.forward();
        out.detections = decode_nanodet_plus_output(raw_out, frame_bgr.cols, frame_bgr.rows, scale, pad_x, pad_y, cfg_copy);
        out.backend = "nanodet_plus_onnx";
        out.model_version = ver_copy;
        out.infer_ms = round3((monotonic_seconds() - t0) * 1000.0);

        {
            std::lock_guard<std::mutex> lk(mutex_);
            cached_onnx_dets_ = out.detections;
            last_infer_ms_ = out.infer_ms;
        }
        apply_physical_guardrails(frame_bgr, out.detections, std::nullopt, std::nullopt, cfg_copy, out.white_glove_glare_warning);
        return out;
    }

    // Enhanced heuristic fallback (white, gray, black ESD gloves + bare skin)
    out.backend = "heuristic_fallback";
    const int sw = hsv_small.cols;
    const int sh = hsv_small.rows;
    if (sw <= 0 || sh <= 0) return out;

    const cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    const cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    const double min_hand_area = sw * sh * 0.010;
    const double max_hand_area = sw * sh * 0.11;

    cv::Mat mot_gt_65, mot_gt_55, fg_gt_14;
    cv::compare(motion_diff, 6.5f, mot_gt_65, cv::CMP_GT);
    cv::compare(motion_diff, 5.5f, mot_gt_55, cv::CMP_GT);
    cv::compare(fg_diff, 14.0f, fg_gt_14, cv::CMP_GT);

    // 1. White & Gray ESD Gloves
    cv::Mat glove_raw = cv::Mat::zeros(sh, sw, CV_8U);
    cv::Mat dark_glove_raw = cv::Mat::zeros(sh, sw, CV_8U);
    for (int y = 0; y < sh; ++y) {
        const auto* hp = hsv_small.ptr<cv::Vec3b>(y);
        const float* mp = motion_diff.ptr<float>(y);
        const float* fp = fg_diff.ptr<float>(y);
        const uint8_t* pp = person_exclusion.ptr<uint8_t>(y);
        const uint8_t* sp = specular_mask.ptr<uint8_t>(y);
        uint8_t* gp = glove_raw.ptr<uint8_t>(y);
        uint8_t* dp = dark_glove_raw.ptr<uint8_t>(y);
        for (int x = 0; x < sw; ++x) {
            if (pp[x] != 0 || sp[x] != 0) continue;
            const bool active_motion = (mp[x] > 7.5f || fp[x] > 20.0f);
            if (!active_motion) continue;
            if (hp[x][2] >= 120 && hp[x][1] <= 48) {
                gp[x] = 255;
            } else if (hp[x][2] <= 75 && hp[x][1] <= 55 && mp[x] > 7.5f && fp[x] > 18.0f) {
                dp[x] = 255;
            }
        }
    }

    auto extract_glove_blobs = [&](const cv::Mat& raw_mask, bool is_dark) {
        cv::Mat clean;
        cv::morphologyEx(raw_mask, clean, cv::MORPH_OPEN, kernel_open);
        cv::morphologyEx(clean, clean, cv::MORPH_CLOSE, kernel_close);
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(clean, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& cnt : cnts) {
            const double area = cv::contourArea(cnt);
            if (area < min_hand_area || area > max_hand_area) continue;
            const cv::Rect r = cv::boundingRect(cnt);
            const double cx = (r.x + r.width * 0.5) / sw;
            if (r.y <= 4 || static_cast<double>(r.y) / sh >= 0.82 ||
                static_cast<double>(r.width) / sw > 0.25 || static_cast<double>(r.height) / sh > 0.40) {
                continue;
            }
            const double r_area = static_cast<double>(r.width * r.height);
            if (static_cast<double>(cv::countNonZero(clipped_mask(r))) / r_area > 0.55) continue;
            if (static_cast<double>(cv::countNonZero(mot_gt_65(r))) / r_area < 0.12) continue;
            if (is_dark) {
                if (static_cast<double>(cv::countNonZero(fg_gt_14(r))) / r_area < 0.18) continue;
                if (cx > 0.36 && cx < 0.64) continue;
            }

            HandGloveDetection det{};
            if (is_dark) {
                det.cls = HandGloveClass::ESD_BLACK;
                det.score = 0.86;
            } else {
                const double mean_v = cv::mean(hsv_small(r))[2];
                det.cls = (mean_v >= 180.0) ? HandGloveClass::ESD_WHITE : HandGloveClass::ESD_GRAY;
                det.score = (det.cls == HandGloveClass::ESD_WHITE) ? 0.94 : 0.88;
            }
            det.class_name = hand_glove_class_name(det.cls);
            det.box = Box{
                std::max(0.01, static_cast<double>(r.x) / sw),
                std::max(0.02, static_cast<double>(r.y) / sh),
                std::min(0.99, static_cast<double>(r.x + r.width) / sw),
                std::min(0.99, static_cast<double>(r.y + r.height) / sh),
            };
            det.source = "heuristic_fallback";
            out.detections.push_back(std::move(det));
        }
    };
    extract_glove_blobs(glove_raw, false);
    extract_glove_blobs(dark_glove_raw, true);

    // 2. Bare-Hand Skin Detection (work zone 0.22 <= cy <= 0.88)
    cv::Mat not_person, bare_raw, bare_clean;
    cv::compare(person_exclusion, 0, not_person, cv::CMP_EQ);
    cv::bitwise_and(skin_mask_small, not_person, bare_raw);
    cv::morphologyEx(bare_raw, bare_clean, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(bare_clean, bare_clean, cv::MORPH_CLOSE, kernel_close);
    std::vector<std::vector<cv::Point>> bare_cnts;
    cv::findContours(bare_clean, bare_cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& cnt : bare_cnts) {
        const double area = cv::contourArea(cnt);
        if (area < min_hand_area || area > max_hand_area) continue;
        const cv::Rect r = cv::boundingRect(cnt);
        const double cx = (r.x + r.width * 0.5) / sw;
        const double cy = (r.y + r.height * 0.5) / sh;
        if (r.y <= 4 && cx > 0.22 && cx < 0.78) continue;
        if (static_cast<double>(r.y) / sh >= 0.82) continue;
        if (cy < cfg_copy.work_zone_cy_min || cy > cfg_copy.work_zone_cy_max) continue;
        const double denom = static_cast<double>(r.width * r.height);
        const double hand_fg = static_cast<double>(cv::countNonZero(fg_gt_14(r))) / denom;
        const double hand_mot = static_cast<double>(cv::countNonZero(mot_gt_55(r))) / denom;
        if (hand_fg < 0.18 && hand_mot < 0.08) continue;
        if (static_cast<double>(r.width) / sw > 0.26 || static_cast<double>(r.height) / sh > 0.42) continue;

        HandGloveDetection det{};
        det.cls = HandGloveClass::BARE_HAND;
        det.class_name = "bare_hand";
        det.score = 0.95;
        det.box = Box{
            std::max(0.01, static_cast<double>(r.x) / sw),
            std::max(0.02, static_cast<double>(r.y) / sh),
            std::min(0.99, static_cast<double>(r.x + r.width) / sw),
            std::min(0.99, static_cast<double>(r.y + r.height) / sh),
        };
        det.source = "heuristic_fallback";
        out.detections.push_back(std::move(det));
    }

    apply_physical_guardrails(frame_bgr, out.detections, std::nullopt, std::nullopt, cfg_copy, out.white_glove_glare_warning);
    out.infer_ms = round3((monotonic_seconds() - t0) * 1000.0);
    {
        std::lock_guard<std::mutex> lk(mutex_);
        last_infer_ms_ = out.infer_ms;
    }
    return out;
}

HysteresisState HandGloveDetector::update_hysteresis(bool raw_bare_detected, bool raw_bare_touching_tube) {
    std::lock_guard<std::mutex> lk(mutex_);
    return hysteresis_.update(raw_bare_detected, raw_bare_touching_tube);
}

}  // namespace igbt::vision
