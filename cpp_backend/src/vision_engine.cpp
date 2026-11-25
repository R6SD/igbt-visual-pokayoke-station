#include <igbt/vision_engine.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace igbt {

namespace {

double round_to(double v, int decimals) {
    const double scale = std::pow(10.0, decimals);
    return std::round(v * scale) / scale;
}

std::pair<double, double> box_iou_and_overlap(const Box& a, const Box& b) {
    const double ix1 = std::max(a[0], b[0]);
    const double iy1 = std::max(a[1], b[1]);
    const double ix2 = std::min(a[2], b[2]);
    const double iy2 = std::min(a[3], b[3]);
    const double iw = std::max(0.0, ix2 - ix1);
    const double ih = std::max(0.0, iy2 - iy1);
    const double inter = iw * ih;
    if (inter <= 0.0) {
        return {0.0, 0.0};
    }
    const double area_a = std::max(1e-6, (a[2] - a[0]) * (a[3] - a[1]));
    const double area_b = std::max(1e-6, (b[2] - b[0]) * (b[3] - b[1]));
    return {inter / (area_a + area_b - inter), inter / area_b};
}

Box ema_box(const std::optional<Box>& prev, const Box& curr, double alpha = 0.45) {
    if (!prev.has_value()) {
        return {
            round_to(curr[0], 4),
            round_to(curr[1], 4),
            round_to(curr[2], 4),
            round_to(curr[3], 4),
        };
    }
    const Box& p = *prev;
    return {
        round_to(p[0] * (1.0 - alpha) + curr[0] * alpha, 4),
        round_to(p[1] * (1.0 - alpha) + curr[1] * alpha, 4),
        round_to(p[2] * (1.0 - alpha) + curr[2] * alpha, 4),
        round_to(p[3] * (1.0 - alpha) + curr[3] * alpha, 4),
    };
}

Box clamp_box(double x1, double y1, double x2, double y2, double min_w = 0.04, double min_h = 0.04) {
    const double cx = std::clamp((x1 + x2) * 0.5, min_w * 0.5, 1.0 - min_w * 0.5);
    const double cy = std::clamp((y1 + y2) * 0.5, min_h * 0.5, 1.0 - min_h * 0.5);
    const double hw = std::clamp((x2 - x1) * 0.5, min_w * 0.5, 0.48);
    const double hh = std::clamp((y2 - y1) * 0.5, min_h * 0.5, 0.48);
    const double bx1 = std::max(0.01, std::min(0.99 - min_w, cx - hw));
    const double by1 = std::max(0.01, std::min(0.99 - min_h, cy - hh));
    const double bx2 = std::min(0.99, std::max(bx1 + min_w, cx + hw));
    const double by2 = std::min(0.99, std::max(by1 + min_h, cy + hh));
    return {round_to(bx1, 4), round_to(by1, 4), round_to(bx2, 4), round_to(by2, 4)};
}

std::optional<Box> json_to_box(const json& val) {
    if (!val.is_array() || val.size() != 4) return std::nullopt;
    Box out{};
    for (std::size_t i = 0; i < 4; ++i) {
        if (!val[i].is_number()) return std::nullopt;
        out[i] = val[i].get<double>();
        if (!std::isfinite(out[i])) return std::nullopt;
    }
    if (!valid_box(out)) return std::nullopt;
    return out;
}

json box_to_json(const std::optional<Box>& b) {
    if (!b.has_value()) return nullptr;
    return json::array({(*b)[0], (*b)[1], (*b)[2], (*b)[3]});
}

double threshold_double(const json& thresholds, const char* key, double fallback) {
    if (!thresholds.is_object() || !thresholds.contains(key) || thresholds[key].is_null()) {
        return fallback;
    }
    if (thresholds[key].is_number()) {
        return thresholds[key].get<double>();
    }
    return fallback;
}

std::string to_lower_trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string label_cn(const std::string& cls) {
    if (cls == "complete") return "完整";
    if (cls == "crossed_out") return "A类·已划除";
    if (cls == "covered") return "B类·已遮挡";
    if (cls == "absent") return "C类·已脱落";
    return "未入位/待定";
}

FSMConfig live_fsm_config() {
    FSMConfig cfg{};
    cfg.initial_complete_ms = 350;
    cfg.stable_state_ms = 400;
    cfg.min_stable_frames = 5;
    cfg.c_absent_duration_ms = 1200;
    cfg.max_frame_gap_ms = 220;
    cfg.max_capture_age_ms = 600;
    cfg.hand_leave_stable_ms = 250;
    return cfg;
}

// Dynamic runtime loader for gdi32.dll so we render crisp Microsoft YaHei CJK HUD tags
// without adding gdi32 to static CMake link dependencies.
struct DynamicGdi32 {
    using CreateCompatibleDCFn = HDC(WINAPI*)(HDC);
    using DeleteDCFn = BOOL(WINAPI*)(HDC);
    using CreateFontWFn = HFONT(WINAPI*)(int, int, int, int, int, DWORD, DWORD, DWORD, DWORD, DWORD, DWORD, DWORD, DWORD, LPCWSTR);
    using SelectObjectFn = HGDIOBJ(WINAPI*)(HDC, HGDIOBJ);
    using DeleteObjectFn = BOOL(WINAPI*)(HGDIOBJ);
    using GetTextExtentPoint32WFn = BOOL(WINAPI*)(HDC, LPCWSTR, int, LPSIZE);
    using CreateDIBSectionFn = HBITMAP(WINAPI*)(HDC, const BITMAPINFO*, UINT, VOID**, HANDLE, DWORD);
    using SetBkModeFn = int(WINAPI*)(HDC, int);
    using SetTextColorFn = COLORREF(WINAPI*)(HDC, COLORREF);
    using TextOutWFn = BOOL(WINAPI*)(HDC, int, int, LPCWSTR, int);
    using GdiFlushFn = BOOL(WINAPI*)();

    HMODULE mod{nullptr};
    CreateCompatibleDCFn pCreateCompatibleDC{nullptr};
    DeleteDCFn pDeleteDC{nullptr};
    CreateFontWFn pCreateFontW{nullptr};
    SelectObjectFn pSelectObject{nullptr};
    DeleteObjectFn pDeleteObject{nullptr};
    GetTextExtentPoint32WFn pGetTextExtentPoint32W{nullptr};
    CreateDIBSectionFn pCreateDIBSection{nullptr};
    SetBkModeFn pSetBkMode{nullptr};
    SetTextColorFn pSetTextColor{nullptr};
    TextOutWFn pTextOutW{nullptr};
    GdiFlushFn pGdiFlush{nullptr};
    bool ok{false};

    template <typename Fn>
    static Fn load_sym(HMODULE m, const char* name) {
        return reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(m, name)));
    }

    DynamicGdi32() {
        mod = LoadLibraryW(L"gdi32.dll");
        if (!mod) return;
        pCreateCompatibleDC = load_sym<CreateCompatibleDCFn>(mod, "CreateCompatibleDC");
        pDeleteDC = load_sym<DeleteDCFn>(mod, "DeleteDC");
        pCreateFontW = load_sym<CreateFontWFn>(mod, "CreateFontW");
        pSelectObject = load_sym<SelectObjectFn>(mod, "SelectObject");
        pDeleteObject = load_sym<DeleteObjectFn>(mod, "DeleteObject");
        pGetTextExtentPoint32W = load_sym<GetTextExtentPoint32WFn>(mod, "GetTextExtentPoint32W");
        pCreateDIBSection = load_sym<CreateDIBSectionFn>(mod, "CreateDIBSection");
        pSetBkMode = load_sym<SetBkModeFn>(mod, "SetBkMode");
        pSetTextColor = load_sym<SetTextColorFn>(mod, "SetTextColor");
        pTextOutW = load_sym<TextOutWFn>(mod, "TextOutW");
        pGdiFlush = load_sym<GdiFlushFn>(mod, "GdiFlush");
        ok = pCreateCompatibleDC && pDeleteDC && pCreateFontW && pSelectObject &&
             pDeleteObject && pGetTextExtentPoint32W && pCreateDIBSection &&
             pSetBkMode && pSetTextColor && pTextOutW && pGdiFlush;
    }

    static const DynamicGdi32& instance() {
        static DynamicGdi32 inst;
        return inst;
    }
};

cv::Mat synthesize_reference_tube_frame(const std::string& pipe_color) {
    cv::Mat img(kDefaultCameraHeight, kDefaultCameraWidth, CV_8UC3, cv::Scalar(34, 38, 42));
    // Subtle workshop desk gradient
    for (int y = 0; y < img.rows; ++y) {
        const uint8_t base = static_cast<uint8_t>(30 + (y * 20) / img.rows);
        img.row(y).setTo(cv::Scalar(base, base + 4, base + 8));
    }

    const cv::Scalar tube_bgr = (pipe_color == "blue")
                                    ? cv::Scalar(195, 110, 35)
                                    : cv::Scalar(35, 185, 225);
    cv::Rect tube_rect(
        static_cast<int>(0.18 * img.cols),
        static_cast<int>(0.44 * img.rows),
        static_cast<int>(0.64 * img.cols),
        static_cast<int>(0.12 * img.rows)
    );
    cv::rectangle(img, tube_rect, tube_bgr, cv::FILLED);
    cv::rectangle(img, tube_rect, cv::Scalar(220, 225, 230), 2);

    // White ESD gloves on left and right grip zones
    cv::rectangle(
        img,
        cv::Rect(static_cast<int>(0.19 * img.cols), static_cast<int>(0.41 * img.rows),
                 static_cast<int>(0.10 * img.cols), static_cast<int>(0.28 * img.rows)),
        cv::Scalar(235, 238, 240),
        cv::FILLED
    );
    cv::rectangle(
        img,
        cv::Rect(static_cast<int>(0.71 * img.cols), static_cast<int>(0.41 * img.rows),
                 static_cast<int>(0.10 * img.cols), static_cast<int>(0.28 * img.rows)),
        cv::Scalar(235, 238, 240),
        cv::FILLED
    );

    // Textured white label inside label_roi (ensuring high Laplacian sharpness > 180 and mean brightness ~165)
    cv::Rect label_rect(
        static_cast<int>(0.43 * img.cols),
        static_cast<int>(0.45 * img.rows),
        static_cast<int>(0.14 * img.cols),
        static_cast<int>(0.10 * img.rows)
    );
    cv::rectangle(img, label_rect, cv::Scalar(218, 220, 222), cv::FILLED);
    cv::rectangle(img, label_rect, cv::Scalar(40, 45, 50), 2);
    for (int r = 0; r < 4; ++r) {
        const int ly = label_rect.y + 12 + r * 14;
        if (ly + 4 < label_rect.y + label_rect.height) {
            for (int c = 0; c < 10; ++c) {
                const int lx = label_rect.x + 12 + c * 15;
                if (lx + 9 < label_rect.x + label_rect.width) {
                    cv::rectangle(img, cv::Rect(lx, ly, 9, 5), cv::Scalar(35, 38, 42), cv::FILLED);
                }
            }
        }
    }
    return img;
}

cv::Mat synthesize_glare_frame(const cv::Mat& frame, const Box& label_roi) {
    cv::Mat glare = frame.clone();
    const int h = glare.rows;
    const int w = glare.cols;
    const int x1 = std::clamp(static_cast<int>(label_roi[0] * w), 0, w - 1);
    const int y1 = std::clamp(static_cast<int>(label_roi[1] * h), 0, h - 1);
    const int x2 = std::clamp(static_cast<int>(std::ceil(label_roi[2] * w)), x1 + 1, w);
    const int y2 = std::clamp(static_cast<int>(std::ceil(label_roi[3] * h)), y1 + 1, h);
    const int glare_y2 = std::min(h, y1 + std::max(4, static_cast<int>((y2 - y1) * 0.45)));
    glare(cv::Rect(x1, y1, x2 - x1, glare_y2 - y1)).setTo(cv::Scalar(255, 255, 255));
    return glare;
}

cv::Mat synthesize_bare_hand_frame(const cv::Mat& frame) {
    cv::Mat out = frame.clone();
    const int w = out.cols;
    const int h = out.rows;
    cv::Rect bare_rect(
        static_cast<int>(0.42 * w),
        static_cast<int>(0.40 * h),
        static_cast<int>(0.15 * w),
        static_cast<int>(0.24 * h)
    );
    // Realistic warm skin BGR + red warning border for clear visual evidence
    cv::rectangle(out, bare_rect, cv::Scalar(132, 162, 212), cv::FILLED);
    cv::rectangle(out, bare_rect, cv::Scalar(35, 45, 215), 3);
    return out;
}

std::filesystem::path find_reference_asset(
    const std::filesystem::path& assets_dir,
    bool use_blue
) {
    const std::wstring filename = use_blue
                                      ? L"02_\u84dd\u7ba1_\u7ad9\u7acb\u6301\u7ba1\u68c0\u6d4b\u6846\u53c2\u8003\u56fe.jpg"
                                      : L"01_\u9ec4\u7ba1_\u7ad9\u7acb\u6301\u7ba1\u68c0\u6d4b\u6846\u53c2\u8003\u56fe.jpg";
    std::error_code ec;
    std::vector<std::filesystem::path> roots;
    if (!assets_dir.empty()) {
        roots.push_back(assets_dir);
    }
    const auto cwd = std::filesystem::current_path(ec);
    if (!cwd.empty()) {
        roots.push_back(cwd / "assets");
        roots.push_back(cwd / "IGBT_PokaYoke_Development_Package" / "assets");
        if (cwd.has_parent_path()) {
            roots.push_back(cwd.parent_path() / "IGBT_PokaYoke_Development_Package" / "assets");
        }
    }
    for (const auto& r : roots) {
        const auto candidate = r / filename;
        if (std::filesystem::is_regular_file(candidate, ec)) {
            return candidate;
        }
    }
    return {};
}

}  // namespace

// ============================================================================
// Geometry & Optical Quality Gate
// ============================================================================

bool valid_box(const Box& box) {
    for (double v : box) {
        if (!std::isfinite(v) || v < 0.0 || v > 1.0) return false;
    }
    return box[2] > box[0] && box[3] > box[1];
}

bool valid_optional_box(const std::optional<Box>& box) {
    return box.has_value() && valid_box(*box);
}

double box_intersection_fraction(const Box& box, const Box& region) {
    const double area = std::max(0.0, box[2] - box[0]) * std::max(0.0, box[3] - box[1]);
    if (area <= 0.0) return 0.0;
    const double inter_w = std::max(0.0, std::min(box[2], region[2]) - std::max(box[0], region[0]));
    const double inter_h = std::max(0.0, std::min(box[3], region[3]) - std::max(box[1], region[1]));
    return (inter_w * inter_h) / area;
}

double overlap_of_box_fraction(const Box& box, const Box& region) {
    if (!valid_box(box) || !valid_box(region)) return 0.0;
    return box_intersection_fraction(box, region);
}

std::optional<QualityMetrics> measure_label_quality(const cv::Mat& frame, const Box& label_roi) {
    if (frame.empty() || !valid_box(label_roi)) {
        return std::nullopt;
    }
    const int height = frame.rows;
    const int width = frame.cols;
    const int x1 = std::max(0, static_cast<int>(label_roi[0] * width));
    const int y1 = std::max(0, static_cast<int>(label_roi[1] * height));
    const int x2 = std::min(width, static_cast<int>(std::ceil(label_roi[2] * width)));
    const int y2 = std::min(height, static_cast<int>(std::ceil(label_roi[3] * height)));
    if (x2 - x1 < 8 || y2 - y1 < 8) {
        return std::nullopt;
    }

    cv::Mat crop = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    cv::Mat gray;
    if (crop.channels() == 3) {
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    } else if (crop.channels() == 4) {
        cv::cvtColor(crop, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = crop;
    }

    const double total = static_cast<double>(gray.rows * gray.cols);
    if (total <= 0.0) return std::nullopt;

    cv::Mat over_mask, under_mask, lap;
    cv::compare(gray, 250, over_mask, cv::CMP_GE);
    cv::compare(gray, 5, under_mask, cv::CMP_LE);
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar lap_mean, lap_std;
    cv::meanStdDev(lap, lap_mean, lap_std);

    QualityMetrics m{};
    m.brightness_mean = cv::mean(gray)[0];
    m.overexposed_fraction = static_cast<double>(cv::countNonZero(over_mask)) / total;
    m.underexposed_fraction = static_cast<double>(cv::countNonZero(under_mask)) / total;
    m.laplacian_variance = lap_std[0] * lap_std[0];
    return m;
}

void ViewConfig::validate() const {
    if (calibrated_width <= 0 || calibrated_height <= 0) {
        throw std::invalid_argument("Calibrated camera dimensions must be positive.");
    }
    if (min_stable_frames < 2 || stable_window_ms <= 0.0 || max_frame_gap_ms <= 0.0) {
        throw std::invalid_argument("Stability frame count and timing limits must be positive.");
    }
    if (camera_device_id.empty() || roi_config_version.empty()) {
        throw std::invalid_argument("Camera device ID and ROI config version are required.");
    }
    if (quality_thresholds_approved &&
        (!min_brightness.has_value() || !max_brightness.has_value() ||
         !max_overexposed_fraction.has_value() || !max_underexposed_fraction.has_value() ||
         !min_laplacian_variance.has_value())) {
        throw std::invalid_argument("Approved quality gates require every site-calibrated threshold value.");
    }
}

ViewConfig default_view_config(
    int width,
    int height,
    const std::string& camera_device_id,
    const std::string& roi_config_version,
    const ROISet& rois
) {
    ViewConfig cfg{};
    cfg.camera_device_id = camera_device_id;
    cfg.calibrated_width = width;
    cfg.calibrated_height = height;
    cfg.roi_config_version = roi_config_version;
    cfg.rois = rois;
    cfg.has_left_grip_zone = true;
    cfg.has_right_grip_zone = true;
    cfg.quality_thresholds_approved = true;
    cfg.orientation_detector_ready = true;
    cfg.min_brightness = 55.0;
    cfg.max_brightness = 225.0;
    cfg.max_overexposed_fraction = 0.08;
    cfg.max_underexposed_fraction = 0.15;
    cfg.min_laplacian_variance = 120.0;
    cfg.max_frame_gap_ms = 200.0;
    cfg.stable_window_ms = 400.0;
    cfg.min_stable_frames = 5;
    cfg.validate();
    return cfg;
}

ViewValidEvaluator::ViewValidEvaluator(const ViewConfig& config)
    : config_(config) {
    config_.validate();
}

void ViewValidEvaluator::reset() {
    history_.clear();
    last_frame_id_.reset();
    last_timestamp_.reset();
}

std::pair<std::vector<std::string>, std::optional<QualityMetrics>> ViewValidEvaluator::base_checks(
    const cv::Mat& frame,
    const FrameObservation& obs
) const {
    std::vector<std::string> reasons;
    if (frame.empty() || (frame.dims != 2)) {
        return {{"FRAME_MISSING_OR_INVALID"}, std::nullopt};
    }
    if (frame.cols != config_.calibrated_width || frame.rows != config_.calibrated_height) {
        reasons.push_back("CALIBRATION_RESOLUTION_MISMATCH");
    }
    if (obs.camera_device_id != config_.camera_device_id) {
        reasons.push_back("CAMERA_DEVICE_MISMATCH");
    }
    if (!obs.roi_config_version.has_value() || *obs.roi_config_version != config_.roi_config_version) {
        reasons.push_back("ROI_CONFIG_VERSION_MISMATCH");
    }
    if (!obs.camera_healthy.value_or(false)) {
        reasons.push_back("CAMERA_NOT_CONFIRMED_HEALTHY");
    }
    if (!obs.tube_present.value_or(false)) {
        reasons.push_back("TUBE_NOT_CONFIRMED_IN_VIEW");
    }
    if (!obs.label_window_present.value_or(false)) {
        reasons.push_back("LABEL_WINDOW_OUT_OF_VIEW_OR_SHIFTED");
    }
    if (!obs.label_occluded.has_value() || *obs.label_occluded) {
        reasons.push_back("LABEL_OCCLUDED_OR_OCCLUSION_UNKNOWN");
    }

    const bool rois_valid =
        valid_box(config_.rois.tube_roi) &&
        valid_box(config_.rois.label_roi) &&
        valid_box(config_.rois.label_contact_roi) &&
        (!config_.has_left_grip_zone || valid_box(config_.rois.left_grip_zone)) &&
        (!config_.has_right_grip_zone || valid_box(config_.rois.right_grip_zone));

    if (!rois_valid) {
        reasons.push_back("ROI_CALIBRATION_INVALID");
    } else if (!config_.has_left_grip_zone || !config_.has_right_grip_zone) {
        reasons.push_back("BOTH_GRIP_ZONES_NOT_CALIBRATED");
    } else {
        if (box_intersection_fraction(config_.rois.label_contact_roi, config_.rois.left_grip_zone) > 0.01 ||
            box_intersection_fraction(config_.rois.label_contact_roi, config_.rois.right_grip_zone) > 0.01) {
            reasons.push_back("CONTACT_AND_GRIP_ZONES_OVERLAP");
        }
    }

    if (!obs.front_face_confirmed.value_or(false)) {
        reasons.push_back("FRONT_FACE_NOT_CONFIRMED");
    }
    if (!obs.orientation_reference_visible.value_or(false) ||
        obs.orientation_evidence_source.value_or("") != "machine_reference") {
        reasons.push_back("MACHINE_ORIENTATION_REFERENCE_NOT_CONFIRMED");
    }
    if (!config_.orientation_detector_ready) {
        reasons.push_back("ORIENTATION_DETECTOR_NOT_VALIDATED");
    }

    const bool thresholds_ready =
        config_.quality_thresholds_approved &&
        config_.min_brightness.has_value() &&
        config_.max_brightness.has_value() &&
        config_.max_overexposed_fraction.has_value() &&
        config_.max_underexposed_fraction.has_value() &&
        config_.min_laplacian_variance.has_value();
    if (!thresholds_ready) {
        reasons.push_back("QUALITY_THRESHOLDS_NOT_CALIBRATED");
    }

    // Check BOTH occluder-in-label AND label-in-occluder overlap >= 0.03
    if (valid_box(config_.rois.label_roi)) {
        for (const auto& box : obs.occluder_boxes_norm) {
            if (valid_box(box) &&
                (box_intersection_fraction(box, config_.rois.label_roi) >= 0.03 ||
                 box_intersection_fraction(config_.rois.label_roi, box) >= 0.03)) {
                reasons.push_back("DETECTED_OCCLUDER_OVERLAPS_LABEL_ROI");
                break;
            }
        }
    }

    const auto quality = measure_label_quality(frame, config_.rois.label_roi);
    if (!quality.has_value()) {
        reasons.push_back("LABEL_ROI_QUALITY_UNAVAILABLE");
    } else if (thresholds_ready) {
        if (quality->brightness_mean < *config_.min_brightness ||
            quality->brightness_mean > *config_.max_brightness) {
            reasons.push_back("LABEL_ROI_BRIGHTNESS_OUT_OF_RANGE");
        }
        if (quality->overexposed_fraction > *config_.max_overexposed_fraction) {
            reasons.push_back("LABEL_ROI_OVEREXPOSED");
        }
        if (quality->underexposed_fraction > *config_.max_underexposed_fraction) {
            reasons.push_back("LABEL_ROI_UNDEREXPOSED");
        }
        if (quality->laplacian_variance < *config_.min_laplacian_variance) {
            reasons.push_back("LABEL_ROI_BLURRED");
        }
    }

    if (!obs.tube_center_norm.has_value() || !obs.label_center_norm.has_value()) {
        reasons.push_back("ROI_POSITION_OBSERVATION_MISSING");
    } else {
        const auto [tx, ty] = *obs.tube_center_norm;
        const auto& tr = config_.rois.tube_roi;
        if (valid_box(tr) && !(tr[0] <= tx && tx <= tr[2] && tr[1] <= ty && ty <= tr[3])) {
            reasons.push_back("TUBE_CENTER_OUTSIDE_TUBE_ROI");
        }
        const auto [lx, ly] = *obs.label_center_norm;
        const auto& lr = config_.rois.label_roi;
        const double rx = valid_box(lr) ? (lr[0] + lr[2]) * 0.5 : lx;
        const double ry = valid_box(lr) ? (lr[1] + lr[3]) * 0.5 : ly;
        if (valid_box(lr) && !(lr[0] <= lx && lx <= lr[2] && lr[1] <= ly && ly <= lr[3])) {
            reasons.push_back("LABEL_CENTER_OUTSIDE_LABEL_ROI");
        } else if (std::hypot(lx - rx, ly - ry) > config_.max_label_roi_offset_norm) {
            reasons.push_back("LABEL_ROI_POSITION_SHIFTED");
        }
    }

    return {reasons, quality};
}

ViewResult ViewValidEvaluator::evaluate(const cv::Mat& frame, const FrameObservation& observation) {
    auto [reasons, quality] = base_checks(frame, observation);
    const double now = observation.captured_monotonic_s;

    if (last_timestamp_.has_value()) {
        const double gap_ms = (now - *last_timestamp_) * 1000.0;
        const int64_t frame_gap = last_frame_id_.has_value()
                                      ? (observation.frame_id - *last_frame_id_)
                                      : 1;
        if (gap_ms <= 0.0 || gap_ms > config_.max_frame_gap_ms || frame_gap != 1) {
            reasons.push_back("FRAME_GAP_OR_TIMESTAMP_DISCONTINUITY");
            history_.clear();
        }
    }
    last_timestamp_ = now;
    last_frame_id_ = observation.frame_id;

    const bool centers_present =
        observation.tube_center_norm.has_value() && observation.label_center_norm.has_value();
    if (!reasons.empty() || !centers_present) {
        history_.clear();
        std::vector<std::string> dedup;
        std::unordered_set<std::string> seen;
        if (reasons.empty()) {
            dedup.push_back("VIEW_EVIDENCE_INCOMPLETE");
        } else {
            for (const auto& r : reasons) {
                if (seen.insert(r).second) dedup.push_back(r);
            }
        }
        return ViewResult{false, dedup, quality, 0, config_.roi_config_version};
    }

    history_.push_back(HistoryItem{
        now,
        observation.frame_id,
        *observation.tube_center_norm,
        *observation.label_center_norm,
    });
    const double cutoff = now - config_.stable_window_ms / 1000.0;
    while (static_cast<int>(history_.size()) > config_.min_stable_frames &&
           history_[1].timestamp_s <= cutoff) {
        history_.pop_front();
    }

    const int stable_count = static_cast<int>(history_.size());
    const double stable_duration_ms =
        stable_count > 1 ? (history_.back().timestamp_s - history_.front().timestamp_s) * 1000.0 : 0.0;
    if (stable_count < config_.min_stable_frames || stable_duration_ms < config_.stable_window_ms) {
        return ViewResult{false, {"STABILITY_WINDOW_NOT_MET"}, quality, stable_count, config_.roi_config_version};
    }

    double max_drift = 0.0;
    for (std::size_t i = 0; i < history_.size(); ++i) {
        for (std::size_t j = i + 1; j < history_.size(); ++j) {
            max_drift = std::max(
                max_drift,
                std::hypot(history_[i].tube_center.first - history_[j].tube_center.first,
                           history_[i].tube_center.second - history_[j].tube_center.second)
            );
            max_drift = std::max(
                max_drift,
                std::hypot(history_[i].label_center.first - history_[j].label_center.first,
                           history_[i].label_center.second - history_[j].label_center.second)
            );
        }
    }
    if (max_drift > config_.max_center_drift_norm) {
        history_.clear();
        return ViewResult{false, {"TUBE_OR_LABEL_MOTION_UNSTABLE"}, quality, stable_count, config_.roi_config_version};
    }
    return ViewResult{true, {}, quality, stable_count, config_.roi_config_version};
}

// ============================================================================
// Hand / ESD Glove Contact Candidate Tracker
// ============================================================================

ContactCandidate detect_contact_candidate(
    const std::vector<DetectionItem>& detections,
    const Box& label_contact_roi,
    const std::optional<Box>& left_grip_zone,
    const std::optional<Box>& right_grip_zone,
    const ContactConfig& config
) {
    if (!valid_box(label_contact_roi)) {
        return ContactCandidate{false, false, "none", "contact_roi_invalid"};
    }
    if (!valid_optional_box(left_grip_zone) || !valid_optional_box(right_grip_zone)) {
        return ContactCandidate{false, false, "none", "both_grip_zones_must_be_calibrated"};
    }

    bool saw_hand = false;
    for (const auto& det : detections) {
        if (det.class_name == "hand_or_pen") continue;
        if (det.class_name != "hand_glove" || det.score < config.hand_score_threshold) continue;
        if (!valid_box(det.box)) continue;
        saw_hand = true;
        const bool in_grip =
            overlap_of_box_fraction(det.box, *left_grip_zone) >= config.grip_overlap_ratio_exclusion ||
            overlap_of_box_fraction(det.box, *right_grip_zone) >= config.grip_overlap_ratio_exclusion;
        if (in_grip) continue;
        if (overlap_of_box_fraction(det.box, label_contact_roi) >= config.contact_overlap_ratio) {
            return ContactCandidate{
                true, false, "hand_contact_candidate", "hand_glove_overlaps_label_contact_roi"
            };
        }
    }
    if (saw_hand) {
        return ContactCandidate{false, false, "none", "hand_outside_contact_roi_or_in_grip_zone"};
    }
    return ContactCandidate{false, false, "none", "no_confident_hand_glove_detection"};
}

ContactCandidateTracker::ContactCandidateTracker(int enter_frames, int exit_frames)
    : enter_frames_(enter_frames), exit_frames_(exit_frames) {
    if (enter_frames_ < 1 || exit_frames_ < 1) {
        throw std::invalid_argument("Contact tracker enter/exit frame counts must be positive.");
    }
}

ContactCandidate ContactCandidateTracker::update(const ContactCandidate& candidate, bool view_available) {
    if (!view_available) {
        reset();
        return ContactCandidate{false, false, "none", "frame_or_detector_unavailable"};
    }
    if (candidate.hand_contact_candidate) {
        ++enter_count_;
        exit_count_ = 0;
        if (!inside_ && enter_count_ >= enter_frames_) {
            inside_ = true;
            return ContactCandidate{true, true, "hand_contact_candidate", candidate.reason};
        }
        return ContactCandidate{
            inside_, false, "none", inside_ ? "candidate_continuing" : "entry_debounce"
        };
    }
    enter_count_ = 0;
    if (inside_) {
        ++exit_count_;
        if (exit_count_ < exit_frames_) {
            return ContactCandidate{true, false, "none", "exit_debounce"};
        }
    }
    inside_ = false;
    exit_count_ = 0;
    return ContactCandidate{false, false, "none", candidate.reason};
}

void ContactCandidateTracker::reset() {
    inside_ = false;
    enter_count_ = 0;
    exit_count_ = 0;
}

// ============================================================================
// Deterministic Station Simulation Engine
// ============================================================================

json SimulationRunResult::to_json() const {
    return json{
        {"scenario", scenario},
        {"snapshot", snapshot},
        {"transitions", transitions},
        {"reference_image_path", reference_image_path},
        {"evidence_kind", evidence_kind},
        {"pipe_color", pipe_color},
        {"has_evidence_bytes", !evidence_image_bytes.empty()},
    };
}

SimulationRunResult run_simulation_scenario(
    const std::string& scenario,
    const std::filesystem::path& assets_dir
) {
    const std::string norm_scenario = (scenario == "alarm") ? "initial_alarm" : scenario;
    static const std::unordered_set<std::string> kSupported = {
        "a_pass", "b_pass", "c_pass", "initial_alarm", "bare_hand_alarm", "glare_invalid", "reset"
    };
    if (kSupported.find(norm_scenario) == kSupported.end()) {
        throw std::invalid_argument("Unsupported simulation scenario: " + scenario);
    }

    const bool use_blue = (norm_scenario == "b_pass" || norm_scenario == "glare_invalid");
    const std::string pipe_color = use_blue ? "blue" : "yellow";
    const auto ref_path = find_reference_asset(assets_dir, use_blue);

    cv::Mat base_frame;
    if (!ref_path.empty()) {
        base_frame = imread_unicode(ref_path, cv::IMREAD_COLOR);
    }
    if (base_frame.empty()) {
        base_frame = synthesize_reference_tube_frame(pipe_color);
    } else if (base_frame.cols != kDefaultCameraWidth || base_frame.rows != kDefaultCameraHeight) {
        cv::resize(
            base_frame,
            base_frame,
            cv::Size(kDefaultCameraWidth, kDefaultCameraHeight),
            0,
            0,
            cv::INTER_AREA
        );
    }

    double now = 1000.0;
    int64_t frame_id = 0;
    const ViewConfig view_cfg = default_view_config(base_frame.cols, base_frame.rows);

    FSMConfig fsm_cfg{};
    fsm_cfg.initial_complete_ms = 400;
    fsm_cfg.stable_state_ms = 400;
    fsm_cfg.min_stable_frames = 5;
    fsm_cfg.c_absent_duration_ms = 1200;
    fsm_cfg.max_frame_gap_ms = 200;
    fsm_cfg.hand_leave_stable_ms = 200;

    ViewValidEvaluator view_eval(view_cfg);
    ContactCandidateTracker contact_tracker(2, 2);
    ContactConfig contact_cfg{};
    InspectionFSM fsm(fsm_cfg, [&now]() { return now; });

    const std::vector<DetectionItem> grip_detections = {
        {"hand_glove", 0.96, {0.18, 0.42, 0.29, 0.74}},
        {"hand_glove", 0.95, {0.71, 0.42, 0.82, 0.74}},
    };
    const std::vector<DetectionItem> central_contact_detections = {
        grip_detections[0],
        grip_detections[1],
        {"hand_glove", 0.94, {0.43, 0.42, 0.57, 0.58}},
    };
    const std::vector<DetectionItem> marker_detections = {
        grip_detections[0],
        grip_detections[1],
        {"marker", 0.91, {0.45, 0.46, 0.55, 0.54}},
    };

    std::string suffix = random_token_hex(3);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    std::string prefix = "CYC_SIM";
    if (norm_scenario == "a_pass") prefix = "CYC_A";
    else if (norm_scenario == "b_pass") prefix = "CYC_B";
    else if (norm_scenario == "c_pass") prefix = "CYC_C";
    else if (norm_scenario == "initial_alarm") prefix = "CYC_ALM";
    else if (norm_scenario == "bare_hand_alarm") prefix = "CYC_BARE";
    else if (norm_scenario == "glare_invalid") prefix = "CYC_GLR";
    else if (norm_scenario == "reset") prefix = "CYC_RST";
    const std::string cycle_id = prefix + "_" + suffix;

    auto next_obs = [&](double dt, bool label_occluded = false) {
        now += dt;
        ++frame_id;
        FrameObservation obs{};
        obs.frame_id = frame_id;
        obs.captured_monotonic_s = now;
        obs.camera_device_id = view_cfg.camera_device_id;
        obs.roi_config_version = view_cfg.roi_config_version;
        obs.camera_healthy = true;
        obs.tube_present = true;
        obs.label_window_present = true;
        obs.label_occluded = label_occluded;
        obs.front_face_confirmed = true;
        obs.orientation_reference_visible = true;
        obs.orientation_evidence_source = "machine_reference";
        obs.tube_center_norm = std::make_pair(0.50, 0.50);
        obs.label_center_norm = std::make_pair(0.50, 0.50);
        return obs;
    };

    static const std::unordered_set<std::string> kQualityFailureReasons = {
        "QUALITY_THRESHOLDS_NOT_CALIBRATED",
        "LABEL_ROI_QUALITY_UNAVAILABLE",
        "LABEL_ROI_BRIGHTNESS_OUT_OF_RANGE",
        "LABEL_ROI_OVEREXPOSED",
        "LABEL_ROI_UNDEREXPOSED",
        "LABEL_ROI_BLURRED",
    };

    auto build_snapshot = [&](
        const FrameObservation& obs,
        const ViewResult& vr,
        const std::optional<QualityMetrics>& q,
        bool quality_ok,
        bool left_grip,
        bool right_grip,
        const ContactCandidate& contact,
        const std::string& label_status,
        std::optional<double> label_conf,
        const std::optional<std::string>& event_type,
        bool bare_violation = false
    ) {
        const auto zh = explain_reason_zh(fsm.result_reason(), fsm.outcome());
        json snap{
            {"mode", "shadow"},
            {"interlock_output_enabled", false},
            {"camera_status", "ONLINE"},
            {"online", true},
            {"camera_online", false},
            {"camera_name", "C++20 Station Reference Simulator"},
            {"camera_device_id", view_cfg.camera_device_id},
            {"backend", "DirectShow"},
            {"camera_width", view_cfg.calibrated_width},
            {"camera_height", view_cfg.calibrated_height},
            {"camera_settings_locked", true},
            {"fps", kDefaultTargetFps},
            {"label_status", label_status},
            {"label_confidence", label_conf.has_value() ? json(round_to(*label_conf, 4)) : json(nullptr)},
            {"view_valid", vr.view_valid && !bare_violation},
            {"view_valid_reasons", vr.reasons},
            {"quality_ok", quality_ok},
            {"laplacian_variance", q.has_value() ? json(round_to(q->laplacian_variance, 2)) : json(nullptr)},
            {"brightness", q.has_value() ? json(round_to(q->brightness_mean, 2)) : json(nullptr)},
            {"overexposed_fraction", q.has_value() ? json(round_to(q->overexposed_fraction, 4)) : json(nullptr)},
            {"underexposed_fraction", q.has_value() ? json(round_to(q->underexposed_fraction, 4)) : json(nullptr)},
            {"left_grip_active", left_grip},
            {"right_grip_active", right_grip},
            {"glove_compliant", (left_grip || right_grip) && !bare_violation},
            {"bare_hand_detected", bare_violation},
            {"bare_hand_violation", bare_violation},
            {"alarm_latched", fsm.outcome() == "ALARM"},
            {"label_contact_active", contact.hand_contact_candidate},
            {"hand_contact_candidate", contact.hand_contact_candidate},
            {"contact_evidence_type", "2d_visual_proximity_proxy"},
            {"proves_physical_contact", false},
            {"fsm_state", fsm.state_str()},
            {"outcome", fsm.outcome()},
            {"result_reason", fsm.result_reason().has_value() ? json(*fsm.result_reason()) : json(nullptr)},
            {"operator_message", zh.operator_message},
            {"engineer_message", zh.engineer_message},
            {"c_timer_ms", fsm.c_timer_ms()},
            {"cycle_id", fsm.cycle_id().has_value() ? json(*fsm.cycle_id()) : json(nullptr)},
            {"pipe_color", pipe_color},
            {"operator_anon_id", "OP_STATION_01"},
            {"event_type", event_type.has_value() ? json(*event_type) : json(nullptr)},
            {"frame_id", obs.frame_id},
            {"capture_monotonic_s", round_to(obs.captured_monotonic_s, 4)},
            {"model_version", kHeuristicModelVersion},
            {"inference_backend", kHeuristicBackendLabel},
            {"config_version", view_cfg.roi_config_version},
            {"work_mode", norm_scenario == "bare_hand_alarm" ? "strict" : "debug"},
            {"mirror_horizontal", true},
            {"updated_at_utc", utc_now_iso()},
        };
        return snap;
    };

    auto step_sim = [&](
        const cv::Mat& frame,
        const FrameObservation& obs,
        const std::vector<DetectionItem>& dets,
        const std::string& label_status,
        double label_conf
    ) {
        const ViewResult vr = view_eval.evaluate(frame, obs);
        const auto q = vr.quality.has_value() ? vr.quality : measure_label_quality(frame, view_cfg.rois.label_roi);
        bool quality_ok = q.has_value();
        for (const auto& r : vr.reasons) {
            if (kQualityFailureReasons.find(r) != kQualityFailureReasons.end()) {
                quality_ok = false;
                break;
            }
        }

        bool left_grip = false;
        bool right_grip = false;
        bool hand_in_contact = false;
        bool marker_det = false;
        for (const auto& d : dets) {
            if (d.class_name == "marker" && d.score >= 0.5) marker_det = true;
            if (d.class_name != "hand_glove" || d.score < contact_cfg.hand_score_threshold) continue;
            if (overlap_of_box_fraction(d.box, view_cfg.rois.left_grip_zone) >= contact_cfg.grip_overlap_ratio_exclusion) {
                left_grip = true;
            }
            if (overlap_of_box_fraction(d.box, view_cfg.rois.right_grip_zone) >= contact_cfg.grip_overlap_ratio_exclusion) {
                right_grip = true;
            }
            if (overlap_of_box_fraction(d.box, view_cfg.rois.label_contact_roi) >= contact_cfg.contact_overlap_ratio &&
                overlap_of_box_fraction(d.box, view_cfg.rois.left_grip_zone) < contact_cfg.grip_overlap_ratio_exclusion &&
                overlap_of_box_fraction(d.box, view_cfg.rois.right_grip_zone) < contact_cfg.grip_overlap_ratio_exclusion) {
                hand_in_contact = true;
            }
        }

        const auto raw_cand = detect_contact_candidate(
            dets,
            view_cfg.rois.label_contact_roi,
            view_cfg.rois.left_grip_zone,
            view_cfg.rois.right_grip_zone,
            contact_cfg
        );
        const auto tracked_cand = contact_tracker.update(raw_cand, !frame.empty());
        const bool hand_occluded = !obs.label_occluded.has_value() || *obs.label_occluded ||
            std::find(vr.reasons.begin(), vr.reasons.end(), "DETECTED_OCCLUDER_OVERLAPS_LABEL_ROI") != vr.reasons.end();
        const bool fsm_view_valid = vr.view_valid ||
            (vr.reasons.size() == 1 && vr.reasons[0] == "STABILITY_WINDOW_NOT_MET");

        FSMEvent ev{};
        ev.kind = "observation";
        ev.cycle_id = fsm.cycle_id();
        ev.capture_monotonic_s = obs.captured_monotonic_s;
        ev.frame_id = obs.frame_id;
        ev.label_class = label_status;
        ev.view_valid = fsm_view_valid;
        ev.quality_ok = quality_ok;
        ev.camera_healthy = true;
        ev.camera_settings_locked = true;
        ev.config_match = true;
        ev.tube_present = true;
        ev.label_window_present = true;
        ev.label_roi_clear = !hand_occluded;
        ev.hand_occluded = hand_occluded;
        ev.hand_contact_candidate = tracked_cand.hand_contact_candidate;
        ev.hand_detection_valid = true;
        ev.hand_in_contact_roi = hand_in_contact || tracked_cand.hand_contact_candidate;
        ev.hand_in_label_contact_roi = hand_in_contact || tracked_cand.hand_contact_candidate;
        ev.hand_in_left_grip_zone = tracked_cand.hand_contact_candidate ? false : left_grip;
        ev.hand_in_right_grip_zone = tracked_cand.hand_contact_candidate ? false : right_grip;
        ev.marker_detected = marker_det;
        ev.pipe_color = pipe_color;
        ev.operator_anon_id = "OP_STATION_01";

        const FSMState before = fsm.state();
        fsm.process(ev, now);
        const FSMState after = fsm.state();

        std::optional<std::string> ev_type;
        if (tracked_cand.label_contact_event) {
            ev_type = "HAND_CONTACT_CANDIDATE";
        } else if (before != after) {
            ev_type = "FSM_" + fsm.state_str();
        }
        return build_snapshot(obs, vr, q, quality_ok, left_grip, right_grip, tracked_cand, label_status, label_conf, ev_type);
    };

    FSMEvent start_ev{};
    start_ev.kind = "new_cycle";
    start_ev.cycle_id = cycle_id;
    start_ev.capture_monotonic_s = now;
    start_ev.pipe_color = pipe_color;
    start_ev.operator_anon_id = "OP_STATION_01";
    fsm.process(start_ev, now);

    json snapshot;
    std::vector<uint8_t> evidence_bytes;
    std::string evidence_kind = "initial_label";

    if (norm_scenario == "reset") {
        auto obs = next_obs(0.1);
        snapshot = step_sim(base_frame, obs, grip_detections, "absent", 0.93);
        now += 0.1;
        FSMEvent rst{};
        rst.kind = "reset";
        rst.reason = "AUTHORIZED_RESET";
        rst.reset_authorized = true;
        rst.capture_monotonic_s = now;
        fsm.process(rst, now);
        view_eval.reset();
        contact_tracker.reset();

        const auto q = measure_label_quality(base_frame, view_cfg.rois.label_roi);
        const auto zh = explain_reason_zh(fsm.result_reason(), fsm.outcome());
        snapshot["fsm_state"] = fsm.state_str();
        snapshot["outcome"] = fsm.outcome();
        snapshot["result_reason"] = "AUTHORIZED_RESET";
        snapshot["operator_message"] = zh.operator_message;
        snapshot["engineer_message"] = zh.engineer_message;
        snapshot["c_timer_ms"] = 0;
        snapshot["cycle_id"] = nullptr;
        snapshot["alarm_latched"] = false;
        snapshot["bare_hand_violation"] = false;
        snapshot["view_valid"] = true;
        snapshot["view_valid_reasons"] = json::array();
        snapshot["quality_ok"] = true;
        snapshot["left_grip_active"] = true;
        snapshot["right_grip_active"] = true;
        snapshot["event_type"] = "AUTHORIZED_RESET";
        if (q.has_value()) {
            snapshot["laplacian_variance"] = round_to(q->laplacian_variance, 2);
            snapshot["brightness"] = round_to(q->brightness_mean, 2);
            snapshot["overexposed_fraction"] = round_to(q->overexposed_fraction, 4);
            snapshot["underexposed_fraction"] = round_to(q->underexposed_fraction, 4);
        }
        cv::imencode(".jpg", base_frame, evidence_bytes, {cv::IMWRITE_JPEG_QUALITY, 90});
    } else if (norm_scenario == "initial_alarm") {
        for (int i = 0; i < 5; ++i) {
            auto obs = next_obs(0.1);
            snapshot = step_sim(base_frame, obs, grip_detections, "absent", 0.96);
        }
        evidence_kind = "anomaly";
        cv::imencode(".jpg", base_frame, evidence_bytes, {cv::IMWRITE_JPEG_QUALITY, 90});
    } else if (norm_scenario == "bare_hand_alarm") {
        cv::Mat bare_frame = synthesize_bare_hand_frame(base_frame);
        auto obs = next_obs(0.1, true);
        FSMEvent fault_ev{};
        fault_ev.kind = "observation";
        // Drive initial observation then latch bare-hand SOP violation in strict mode
        snapshot = step_sim(bare_frame, obs, {}, "unknown", 0.45);
        // Record explicit BARE_HAND_VIOLATION alarm transition
        now += 0.04;
        ++frame_id;
        const auto zh = explain_reason_zh(std::string("BARE_HAND_VIOLATION"), "ALARM");
        snapshot["fsm_state"] = "ALARM";
        snapshot["outcome"] = "ALARM";
        snapshot["result_reason"] = "BARE_HAND_VIOLATION";
        snapshot["operator_message"] = zh.operator_message;
        snapshot["engineer_message"] = zh.engineer_message;
        snapshot["bare_hand_detected"] = true;
        snapshot["bare_hand_violation"] = true;
        snapshot["glove_compliant"] = false;
        snapshot["alarm_latched"] = true;
        snapshot["view_valid"] = false;
        snapshot["view_valid_reasons"] = json::array({"BARE_HAND_VIOLATION"});
        snapshot["cycle_id"] = cycle_id;
        snapshot["event_type"] = "FSM_ALARM";
        evidence_kind = "anomaly";
        cv::imencode(".jpg", bare_frame, evidence_bytes, {cv::IMWRITE_JPEG_QUALITY, 90});
    } else {
        for (int i = 0; i < 10; ++i) {
            auto obs = next_obs(0.1);
            snapshot = step_sim(base_frame, obs, grip_detections, "complete", 0.98);
            if (fsm.state() == FSMState::READY) break;
        }
        FSMEvent si{};
        si.kind = "start_inspection";
        si.capture_monotonic_s = now;
        si.frame_id = frame_id;
        fsm.process(si, now);

        if (norm_scenario == "a_pass") {
            for (int i = 0; i < 6; ++i) {
                auto obs = next_obs(0.1);
                snapshot = step_sim(base_frame, obs, marker_detections, "crossed_out", 0.97);
                if (fsm.state() == FSMState::PASS) break;
            }
            evidence_kind = "A";
            cv::imencode(".jpg", base_frame, evidence_bytes, {cv::IMWRITE_JPEG_QUALITY, 90});
        } else if (norm_scenario == "b_pass") {
            for (int i = 0; i < 6; ++i) {
                auto obs = next_obs(0.1);
                snapshot = step_sim(base_frame, obs, grip_detections, "covered", 0.96);
                if (fsm.state() == FSMState::PASS) break;
            }
            evidence_kind = "B";
            cv::imencode(".jpg", base_frame, evidence_bytes, {cv::IMWRITE_JPEG_QUALITY, 90});
        } else if (norm_scenario == "c_pass") {
            for (int i = 0; i < 2; ++i) {
                auto obs = next_obs(0.1, true);
                step_sim(base_frame, obs, central_contact_detections, "complete", 0.72);
            }
            for (int i = 0; i < 20; ++i) {
                auto obs = next_obs(0.1, false);
                snapshot = step_sim(base_frame, obs, grip_detections, "absent", 0.98);
                if (fsm.state() == FSMState::PASS) break;
            }
            evidence_kind = "C";
            cv::imencode(".jpg", base_frame, evidence_bytes, {cv::IMWRITE_JPEG_QUALITY, 90});
        } else if (norm_scenario == "glare_invalid") {
            cv::Mat glare_frame = synthesize_glare_frame(base_frame, view_cfg.rois.label_roi);
            for (int i = 0; i < 2; ++i) {
                auto obs = next_obs(0.1);
                snapshot = step_sim(glare_frame, obs, grip_detections, "complete", 0.61);
            }
            evidence_kind = "anomaly";
            cv::imencode(".jpg", glare_frame, evidence_bytes, {cv::IMWRITE_JPEG_QUALITY, 92});
        }
    }

    SimulationRunResult out{};
    out.scenario = scenario;
    out.snapshot = std::move(snapshot);
    out.transitions = fsm.transition_log_json();
    if (norm_scenario == "bare_hand_alarm") {
        out.transitions.push_back(json{
            {"sequence", out.transitions.size() + 1},
            {"record_type", "STATE_TRANSITION"},
            {"event", {{"kind", "bare_hand_violation"}}},
            {"now_monotonic_s", now},
            {"capture_monotonic_s", now},
            {"frame_id", frame_id},
            {"state_before", "VERIFY_INITIAL"},
            {"state_after", "ALARM"},
            {"outcome", "ALARM"},
            {"reason", "BARE_HAND_VIOLATION"},
            {"cycle_id", cycle_id},
            {"c_timer_ms", 0},
            {"initial_complete_confirmed", false},
        });
    }
    out.reference_image_path = ref_path.empty() ? "" : path_to_utf8(ref_path);
    out.evidence_image_bytes = std::move(evidence_bytes);
    out.evidence_kind = evidence_kind;
    out.pipe_color = pipe_color;
    return out;
}

// ============================================================================
// LiveCameraEngine (720P @ 25.0 FPS Pacer, Dual Model Inference, Auto-Capture)
// ============================================================================

LiveCameraEngine::LiveCameraEngine(
    int device_index,
    ConfigLoader config_loader,
    CycleCallback cycle_callback
)
    : device_index_(device_index),
      config_loader_(std::move(config_loader)),
      cached_config_(json::object()),
      cycle_callback_(std::move(cycle_callback)),
      camera_device_id_("CAM_STATION_01"),
      fsm_(live_fsm_config()),
      contact_tracker_(2, 2) {
    const auto zh = explain_reason_zh(std::string("WAITING_FOR_PART"), "WAITING");
    state_ = json{
        {"mode", "shadow"},
        {"interlock_output_enabled", false},
        {"camera_status", "NOT_CONNECTED"},
        {"online", false},
        {"camera_online", false},
        {"camera_name", camera_name_},
        {"camera_device_id", camera_device_id_},
        {"backend", backend_},
        {"camera_width", width_},
        {"camera_height", height_},
        {"camera_settings_locked", true},
        {"fps", 0.0},
        {"label_status", "unknown"},
        {"label_confidence", nullptr},
        {"view_valid", false},
        {"view_valid_reasons", json::array({"CAMERA_INITIALIZING"})},
        {"quality_ok", false},
        {"laplacian_variance", nullptr},
        {"brightness", nullptr},
        {"overexposed_fraction", nullptr},
        {"underexposed_fraction", nullptr},
        {"left_grip_active", false},
        {"right_grip_active", false},
        {"glove_compliant", false},
        {"bare_hand_detected", false},
        {"bare_hand_violation", false},
        {"alarm_latched", false},
        {"label_contact_active", false},
        {"hand_contact_candidate", false},
        {"contact_evidence_type", "2d_visual_proximity_proxy"},
        {"proves_physical_contact", false},
        {"fsm_state", "WAIT_PART"},
        {"outcome", "WAITING"},
        {"result_reason", "WAITING_FOR_PART"},
        {"operator_message", zh.operator_message},
        {"engineer_message", zh.engineer_message},
        {"c_timer_ms", 0},
        {"cycle_id", nullptr},
        {"pipe_color", nullptr},
        {"operator_anon_id", "OP_STATION_01"},
        {"frame_id", 0},
        {"capture_monotonic_s", 0.0},
        {"model_version", active_model_version_},
        {"inference_backend", kHeuristicBackendLabel},
        {"inference_ms", 0.0},
        {"show_auxiliary_rois", false},
        {"mirror_horizontal", true},
        {"work_mode", "strict"},
        {"hardware_3a", hardware_3a_status_},
        {"tracked_objects", json::array()},
        {"tracked_label_box", nullptr},
        {"config_version", "ROI_V001"},
        {"auto_capture", auto_capture_.to_json()},
        {"live_stream_url", "/api/camera/stream"},
        {"live_crop_url", "/api/camera/stream_crop"},
        {"updated_at_utc", utc_now_iso()},
    };
}

LiveCameraEngine::~LiveCameraEngine() {
    stop(3000);
    async_evidence_writer_.stop_and_drain();
}

std::string LiveCameraEngine::generate_live_cycle_id() {
    static std::atomic<uint64_t> seq{0};
    const uint64_t n = ++seq;
    using namespace std::chrono;
    const auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    std::tm gmt{};
    gmtime_s(&gmt, &tt);
    std::string rand_hex = random_token_hex(4);
    std::transform(rand_hex.begin(), rand_hex.end(), rand_hex.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    std::ostringstream oss;
    oss << "LIVE_" << std::put_time(&gmt, "%Y%m%d_%H%M%S") << "_"
        << std::setw(4) << std::setfill('0') << (n % 10000) << "_" << rand_hex;
    return oss.str();
}

std::chrono::steady_clock::time_point LiveCameraEngine::next_pacer_deadline(
    std::chrono::steady_clock::time_point prev_deadline,
    std::chrono::steady_clock::time_point now,
    std::chrono::microseconds interval
) {
    const auto candidate = prev_deadline + interval;
    // Compensate small jitter without drifting; if stalled by > 2 intervals, resync cleanly to now + interval
    if (now > candidate + interval) {
        return now + interval;
    }
    return candidate;
}

bool LiveCameraEngine::start() {
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    if (worker_thread_.joinable()) {
        if (running_.load() && !stop_requested_.load()) {
            return true;
        }
        worker_thread_.join();
    }
    stop_requested_.store(false);
    running_.store(true);
    worker_thread_ = std::thread(&LiveCameraEngine::capture_loop, this);
    return true;
}

bool LiveCameraEngine::stop(int timeout_ms) {
    (void)timeout_ms;
    {
        std::lock_guard<std::mutex> lk(lifecycle_mutex_);
        stop_requested_.store(true);
        running_.store(false);
    }
    frame_cv_.notify_all();

    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    return true;
}

bool LiveCameraEngine::is_online() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return camera_online_;
}

void LiveCameraEngine::set_cycle_callback(CycleCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    cycle_callback_ = std::move(cb);
}

void LiveCameraEngine::set_config_loader(ConfigLoader loader) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_loader_ = std::move(loader);
}

void LiveCameraEngine::update_config(const json& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (cfg.is_object()) {
        cached_config_ = cfg;
        if (cfg.contains("camera_device_id") && cfg["camera_device_id"].is_string()) {
            const std::string cid = cfg["camera_device_id"].get<std::string>();
            if (!cid.empty()) {
                camera_device_id_ = cid;
                state_["camera_device_id"] = cid;
            }
        }
        if (cfg.contains("roi_config_version") && cfg["roi_config_version"].is_string()) {
            state_["config_version"] = cfg["roi_config_version"].get<std::string>();
        }
    }
}

bool LiveCameraEngine::set_mirror_horizontal(bool mirror) {
    std::lock_guard<std::mutex> lk(mutex_);
    const bool changed = (mirror != mirror_horizontal_);
    mirror_horizontal_ = mirror;
    if (changed) {
        ++mirror_generation_;
        pending_tracker_reset_ = true;
    }
    if (!camera_online_) {
        state_["mirror_horizontal"] = mirror;
    }
    return mirror_horizontal_;
}

bool LiveCameraEngine::get_mirror_horizontal() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return mirror_horizontal_;
}

std::string LiveCameraEngine::set_work_mode(const std::string& mode) {
    const std::string normalized = (mode == "strict") ? "strict" : "debug";
    std::lock_guard<std::mutex> lk(mutex_);
    const std::string prev_mode = work_mode_;
    work_mode_ = normalized;
    ++work_mode_generation_;
    state_["work_mode"] = normalized;

    if (normalized == "debug") {
        alarm_latched_ = false;
        latched_reason_.reset();
        state_["alarm_latched"] = false;
        state_["bare_hand_violation"] = false;
    } else if (prev_mode != "strict") {
        // Clear debug-earned PASS or partial debug cycle while preserving active ALARM latch
        if (!alarm_latched_ && fsm_.state() != FSMState::ALARM) {
            fsm_ = InspectionFSM(live_fsm_config());
            contact_tracker_.reset();
            pass_target_lost_since_.reset();
            ++cycle_generation_;
            pending_tracker_reset_ = true;
            const auto zh = explain_reason_zh(std::string("STRICT_MODE_REVERIFY_REQUIRED"), "WAITING");
            state_["fsm_state"] = "WAIT_PART";
            state_["outcome"] = "WAITING";
            state_["result_reason"] = "STRICT_MODE_REVERIFY_REQUIRED";
            state_["operator_message"] = zh.operator_message;
            state_["engineer_message"] = zh.engineer_message;
            state_["cycle_id"] = nullptr;
            state_["c_timer_ms"] = 0;
            state_["view_valid"] = false;
        }
    }
    return normalized;
}

std::string LiveCameraEngine::get_work_mode() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return work_mode_;
}

bool LiveCameraEngine::set_show_auxiliary_rois(bool show) {
    std::lock_guard<std::mutex> lk(mutex_);
    show_auxiliary_rois_ = show;
    state_["show_auxiliary_rois"] = show;
    return show_auxiliary_rois_;
}

bool LiveCameraEngine::get_show_auxiliary_rois() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return show_auxiliary_rois_;
}

json LiveCameraEngine::trigger_new_cycle(const std::string& pipe_color) {
    const double now = monotonic_seconds();
    std::lock_guard<std::mutex> lk(mutex_);
    simulation_override_until_ = 0.0;
    alarm_latched_ = false;
    latched_reason_.reset();
    latched_cycle_id_.reset();
    pass_target_lost_since_.reset();
    active_cycle_tube_center_.reset();
    active_cycle_pipe_color_.clear();
    pending_tracker_reset_ = true;
    ++cycle_generation_;
    last_emitted_dedup_key_.clear();
    contact_tracker_.reset();
    fsm_ = InspectionFSM(live_fsm_config());

    const std::string cid = generate_live_cycle_id();
    FSMEvent ev{};
    ev.kind = "new_cycle";
    ev.cycle_id = cid;
    ev.capture_monotonic_s = now;
    ev.pipe_color = pipe_color;
    ev.operator_anon_id = "OP_STATION_01";
    fsm_.process(ev, now);

    const auto zh = explain_reason_zh(std::string("AUTHORIZED_RESET"), "WAITING");
    state_["cycle_id"] = cid;
    state_["pipe_color"] = pipe_color;
    state_["fsm_state"] = "WAIT_PART";
    state_["outcome"] = "WAITING";
    state_["result_reason"] = "AUTHORIZED_RESET";
    state_["operator_message"] = zh.operator_message;
    state_["engineer_message"] = zh.engineer_message;
    state_["alarm_latched"] = false;
    state_["bare_hand_violation"] = false;
    state_["c_timer_ms"] = 0;
    state_["updated_at_utc"] = utc_now_iso();
    return state_;
}

void LiveCameraEngine::pause_for_simulation(double seconds) {
    std::lock_guard<std::mutex> lk(mutex_);
    simulation_override_until_ = monotonic_seconds() + std::max(0.0, seconds);
}

json LiveCameraEngine::resume_live() {
    std::lock_guard<std::mutex> lk(mutex_);
    simulation_override_until_ = 0.0;
    pending_tracker_reset_ = true;
    return state_;
}

SimulationRunResult LiveCameraEngine::run_simulation_scenario(
    const std::string& scenario,
    const std::filesystem::path& assets_dir
) {
    if (is_online()) {
        pause_for_simulation(8.0);
    }
    return igbt::run_simulation_scenario(scenario, assets_dir);
}

std::size_t LiveCameraEngine::discard_pending_transitions(const std::vector<std::string>& cycle_ids) {
    std::lock_guard<std::mutex> lk(mutex_);
    const std::size_t before = pending_transitions_.size();
    if (cycle_ids.empty()) {
        pending_transitions_.clear();
        last_evidence_emit_by_cycle_.clear();
    } else {
        const std::unordered_set<std::string> targets(cycle_ids.begin(), cycle_ids.end());
        std::deque<PendingTransition> kept;
        for (auto& item : pending_transitions_) {
            const std::string cid = item.snapshot.value("cycle_id", std::string(""));
            if (targets.find(cid) == targets.end()) {
                kept.push_back(std::move(item));
            }
        }
        pending_transitions_ = std::move(kept);
        for (const auto& cid : targets) {
            last_evidence_emit_by_cycle_.erase(cid);
        }
    }
    if (pending_transitions_.size() < max_pending_transitions_) {
        persistence_overflow_fault_ = false;
    }
    return before - pending_transitions_.size();
}

// ============================================================================
// Pure C++ Dual Model Hot-Load (OpenCV ANN_MLP .xml & OpenCV DNN .onnx)
// ============================================================================

json LiveCameraEngine::load_mlp_model(
    const std::filesystem::path& xml_path,
    const std::string& model_version,
    const std::string& expected_sha256
) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(xml_path, bytes) || bytes.empty()) {
        throw std::runtime_error("MODEL_ARTIFACT_UNAVAILABLE: " + path_to_utf8(xml_path));
    }
    const std::string actual_sha = sha256_hex(bytes);
    if (!expected_sha256.empty()) {
        const std::string exp_norm = to_lower_trim(expected_sha256);
        if (!constant_time_equals(actual_sha, exp_norm)) {
            throw std::invalid_argument(
                "MODEL_SHA256_MISMATCH: expected " + exp_norm + ", got " + actual_sha
            );
        }
    }

    // Attempt to load ANN_MLP from XML string via cv::Algorithm::loadFromString (100% compatible with TrainingManager)
    const std::string xml_str(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    cv::Ptr<cv::ml::ANN_MLP> mlp;
    try {
        mlp = cv::Algorithm::loadFromString<cv::ml::ANN_MLP>(xml_str);
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("MODEL_ARTIFACT_LOAD_FAILED: ") + ex.what());
    }
    if (mlp.empty() || !mlp->isTrained()) {
        throw std::runtime_error("MODEL_ARTIFACT_LOAD_FAILED: ANN_MLP is empty or not trained");
    }

    // Sanity forward pass on 64-D feature vector -> 4 classes
    cv::Mat dummy_in = cv::Mat::zeros(1, 64, CV_32F);
    cv::Mat dummy_out;
    mlp->predict(dummy_in, dummy_out);
    if (dummy_out.empty() || dummy_out.total() != 4) {
        throw std::runtime_error("MODEL_ARTIFACT_LOAD_FAILED: ANN_MLP output dimension must be 4");
    }

    const std::string utf8_path = path_to_utf8(xml_path);
    const std::string backend_lbl = "OpenCV ANN_MLP (" + model_version + ")";
    {
        std::lock_guard<std::mutex> mlk(model_mutex_);
        active_model_kind_ = ActiveModelKind::MLP_XML;
        mlp_model_ = mlp;
        onnx_net_ = cv::dnn::Net{};
        active_model_version_ = model_version;
        active_model_path_ = utf8_path;
        active_model_sha256_ = actual_sha;
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        state_["model_version"] = model_version;
        state_["inference_backend"] = backend_lbl;
    }

    return json{
        {"model_version", model_version},
        {"artifact_format", "opencv_mlp"},
        {"model_path", utf8_path},
        {"onnx_path", utf8_path},
        {"input_size", kDefaultInputSize},
        {"sha256", actual_sha},
    };
}

json LiveCameraEngine::load_onnx_model(
    const std::filesystem::path& onnx_path,
    const std::string& model_version,
    const std::string& expected_sha256,
    int input_size
) {
    const int safe_size = std::clamp(input_size, 32, 1024);
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(onnx_path, bytes) || bytes.empty()) {
        throw std::runtime_error("MODEL_ARTIFACT_UNAVAILABLE: " + path_to_utf8(onnx_path));
    }
    const std::string actual_sha = sha256_hex(bytes);
    if (!expected_sha256.empty()) {
        const std::string exp_norm = to_lower_trim(expected_sha256);
        if (!constant_time_equals(actual_sha, exp_norm)) {
            throw std::invalid_argument(
                "MODEL_SHA256_MISMATCH: expected " + exp_norm + ", got " + actual_sha
            );
        }
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(bytes);
    if (net.empty()) {
        throw std::runtime_error("MODEL_ARTIFACT_LOAD_FAILED: readNetFromONNX returned empty net");
    }

    // Verify 1x3xHxW forward pass produces >= 4 logits
    const int dims[4] = {1, 3, safe_size, safe_size};
    cv::Mat dummy_blob(4, dims, CV_32F, cv::Scalar(0));
    net.setInput(dummy_blob);
    cv::Mat dummy_out = net.forward();
    if (dummy_out.empty() || dummy_out.total() < 4) {
        throw std::runtime_error("MODEL_ARTIFACT_LOAD_FAILED: ONNX output dimension must be >= 4");
    }

    const std::string utf8_path = path_to_utf8(onnx_path);
    const std::string backend_lbl = "OpenCV DNN ONNX (" + model_version + ")";
    {
        std::lock_guard<std::mutex> mlk(model_mutex_);
        active_model_kind_ = ActiveModelKind::ONNX_DNN;
        onnx_net_ = std::move(net);
        mlp_model_.release();
        onnx_input_size_ = safe_size;
        active_model_version_ = model_version;
        active_model_path_ = utf8_path;
        active_model_sha256_ = actual_sha;
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        state_["model_version"] = model_version;
        state_["inference_backend"] = backend_lbl;
    }

    return json{
        {"model_version", model_version},
        {"artifact_format", "onnx"},
        {"model_path", utf8_path},
        {"onnx_path", utf8_path},
        {"input_size", safe_size},
        {"sha256", actual_sha},
    };
}

void LiveCameraEngine::clear_loaded_model() {
    {
        std::lock_guard<std::mutex> mlk(model_mutex_);
        active_model_kind_ = ActiveModelKind::NONE;
        mlp_model_.release();
        onnx_net_ = cv::dnn::Net{};
        active_model_version_ = kHeuristicModelVersion;
        active_model_path_.clear();
        active_model_sha256_.clear();
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        state_["model_version"] = kHeuristicModelVersion;
        state_["inference_backend"] = kHeuristicBackendLabel;
    }
}

// ============================================================================
// Skip-Frame Auto-Capture (A3)
// ============================================================================

AutoCaptureConfig LiveCameraEngine::set_auto_capture(const AutoCaptureConfig& cfg) {
    AutoCaptureConfig sanitized = cfg;
    sanitized.every_n_frames = std::clamp(cfg.every_n_frames, 1, 1000);
    sanitized.max_samples = std::clamp(cfg.max_samples, 1, 10000);
    sanitized.min_diff_mad = std::isfinite(cfg.min_diff_mad)
                                 ? std::clamp(cfg.min_diff_mad, 0.0, 255.0)
                                 : 0.0;
    if (std::find(kLabelClasses.begin(), kLabelClasses.end(), sanitized.label) == kLabelClasses.end() &&
        sanitized.label != "unknown_challenge") {
        sanitized.label = "complete";
    }
    if (sanitized.pipe_color != "blue" && sanitized.pipe_color != "yellow") {
        sanitized.pipe_color = "yellow";
    }
    if (sanitized.physical_tube_id.empty()) {
        sanitized.physical_tube_id = "UNGROUPED";
    }

    std::lock_guard<std::mutex> lk(mutex_);
    if (sanitized.enabled && !auto_capture_.enabled) {
        // Fresh auto-capture session starts counter at 0 unless caller explicitly passed a non-zero seed
        sanitized.captured_count = std::max(0, cfg.captured_count);
        if (sanitized.captured_count >= sanitized.max_samples) {
            sanitized.captured_count = 0;
        }
        last_auto_capture_gray_small_.release();
    } else if (sanitized.enabled) {
        last_auto_capture_gray_small_.release();
    } else {
        sanitized.captured_count = auto_capture_.captured_count;
    }
    auto_capture_ = sanitized;
    state_["auto_capture"] = auto_capture_.to_json();
    return auto_capture_;
}

AutoCaptureConfig LiveCameraEngine::get_auto_capture() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return auto_capture_;
}

void LiveCameraEngine::set_auto_sample_callback(AutoSampleCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto_sample_callback_ = std::move(cb);
}

void LiveCameraEngine::set_evidence_root(const std::filesystem::path& root) {
    async_evidence_writer_.set_evidence_root(root);
}

json LiveCameraEngine::async_evidence_stats() const {
    return json{
        {"ring_buffered", pre_trigger_ring_.size()},
        {"ring_capacity", pre_trigger_ring_.capacity()},
        {"ring_stride", pre_trigger_ring_.stride()},
        {"writer", async_evidence_writer_.stats_json()}
    };
}

void LiveCameraEngine::stop_async_writer() {
    async_evidence_writer_.stop_and_drain();
}

// ============================================================================
// Thread-Safe State, Atomic Capture & MJPEG Streaming
// ============================================================================

json LiveCameraEngine::get_state_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    json copy = state_;
    copy["auto_capture"] = auto_capture_.to_json();
    copy["online"] = camera_online_;
    copy["camera_online"] = camera_online_;
    return copy;
}

cv::Mat LiveCameraEngine::crop_norm(const cv::Mat& frame, const Box& box) {
    if (frame.empty()) return {};
    const int h = frame.rows;
    const int w = frame.cols;
    const int x1 = std::clamp(static_cast<int>(box[0] * w), 0, std::max(0, w - 4));
    const int y1 = std::clamp(static_cast<int>(box[1] * h), 0, std::max(0, h - 4));
    const int x2 = std::clamp(static_cast<int>(box[2] * w), x1 + 4, w);
    const int y2 = std::clamp(static_cast<int>(box[3] * h), y1 + 4, h);
    if (x2 <= x1 || y2 <= y1) return frame.clone();
    return frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
}

AtomicCaptureSnapshot LiveCameraEngine::get_atomic_capture(
    std::optional<Box> custom_box,
    bool encode_jpeg
) const {
    if (custom_box.has_value() && !valid_box(*custom_box)) {
        throw std::invalid_argument("INVALID_CROP_BOX");
    }

    AtomicCaptureSnapshot out{};
    Box effective_box{0.41, 0.42, 0.59, 0.58};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        out.state = !raw_frame_state_.is_null() && raw_frame_state_.is_object()
                        ? raw_frame_state_
                        : state_;
        out.state["auto_capture"] = auto_capture_.to_json();
        if (raw_frame_.empty()) {
            return out;
        }
        out.frame_bgr = raw_frame_.clone();
        if (custom_box.has_value()) {
            effective_box = *custom_box;
            out.crop_box = custom_box;
        } else if (raw_frame_label_box_.has_value()) {
            effective_box = *raw_frame_label_box_;
            out.crop_box = raw_frame_label_box_;
        }
        out.state["tracked_label_box"] = box_to_json(raw_frame_label_box_);
    }

    out.crop_bgr = crop_norm(out.frame_bgr, effective_box);
    if (encode_jpeg && !out.frame_bgr.empty()) {
        cv::imencode(".jpg", out.frame_bgr, out.frame_jpeg, {cv::IMWRITE_JPEG_QUALITY, 92});
    }
    return out;
}

std::vector<uint8_t> LiveCameraEngine::get_raw_frame_jpeg(int quality) const {
    cv::Mat copy;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (raw_frame_.empty()) return {};
        copy = raw_frame_.clone();
    }
    std::vector<uint8_t> encoded;
    cv::imencode(".jpg", copy, encoded, {cv::IMWRITE_JPEG_QUALITY, std::clamp(quality, 50, 100)});
    return encoded;
}

std::vector<uint8_t> LiveCameraEngine::get_hud_jpeg() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return hud_jpeg_;
}

std::vector<uint8_t> LiveCameraEngine::get_crop_jpeg() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return crop_jpeg_;
}

std::pair<std::vector<uint8_t>, uint64_t> LiveCameraEngine::get_hud_jpeg_with_seq() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return {hud_jpeg_, hud_seq_};
}

std::pair<std::vector<uint8_t>, uint64_t> LiveCameraEngine::get_crop_jpeg_with_seq() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return {crop_jpeg_, crop_seq_};
}

cv::Mat LiveCameraEngine::get_tracked_crop_bgr() const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!last_tracked_crop_.empty()) {
        return last_tracked_crop_.clone();
    }
    if (raw_frame_.empty()) return {};
    const Box box = raw_frame_label_box_.value_or(
        tracked_label_box_.value_or(Box{0.41, 0.42, 0.59, 0.58})
    );
    return crop_norm(raw_frame_, box);
}

WaitJpegResult LiveCameraEngine::wait_for_jpeg(uint64_t last_seq, int timeout_ms, bool crop) const {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(std::max(0, timeout_ms));
    std::unique_lock<std::mutex> lk(mutex_);
    while ((crop ? crop_seq_ : hud_seq_) <= last_seq && running_.load() && !stop_requested_.load()) {
        if (frame_cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
            break;
        }
    }
    WaitJpegResult res{};
    if (crop) {
        res.jpeg = crop_jpeg_;
        res.seq = crop_seq_;
    } else {
        res.jpeg = hud_jpeg_;
        res.seq = hud_seq_;
    }
    return res;
}

// ============================================================================
// Camera Capture Loop & 720P @ 25.0 FPS Pacer
// ============================================================================

void LiveCameraEngine::mark_camera_unavailable(const std::string& reason) {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        camera_online_ = false;
        fps_ = 0.0;
        if (monotonic_seconds() >= simulation_override_until_) {
            state_["camera_status"] = (reason == "CAMERA_FEED_LOST") ? "OFFLINE" : "NOT_CONNECTED";
            state_["online"] = false;
            state_["camera_online"] = false;
            state_["fps"] = 0.0;
            state_["view_valid"] = false;
            state_["view_valid_reasons"] = json::array({reason});
            state_["quality_ok"] = false;
            const auto zh = explain_reason_zh(reason, alarm_latched_ ? "ALARM" : "VIEW_INVALID");
            if (!alarm_latched_ && state_.value("outcome", std::string("")) == "PASS") {
                state_["outcome"] = "VIEW_INVALID";
                state_["fsm_state"] = "VIEW_INVALID";
                state_["result_reason"] = reason;
                state_["operator_message"] = zh.operator_message;
                state_["engineer_message"] = zh.engineer_message;
            } else if (!alarm_latched_ && state_.value("outcome", std::string("")) != "ALARM") {
                state_["result_reason"] = reason;
                state_["operator_message"] = zh.operator_message;
                state_["engineer_message"] = zh.engineer_message;
            }
            state_["updated_at_utc"] = utc_now_iso();
        }
    }
    frame_cv_.notify_all();
}

void LiveCameraEngine::capture_loop() {
    constexpr auto kFrameInterval = std::chrono::microseconds(40000);  // 25.0 FPS = 40.0 ms

    auto interruptible_sleep = [&](std::chrono::milliseconds dur) {
        const auto end = std::chrono::steady_clock::now() + dur;
        while (!stop_requested_.load()) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= end) break;
            const auto step = std::min<std::chrono::milliseconds>(
                std::chrono::milliseconds(50),
                std::chrono::duration_cast<std::chrono::milliseconds>(end - now)
            );
            std::this_thread::sleep_for(std::max(std::chrono::milliseconds(1), step));
        }
    };

    class HighResWaitableTimer {
    public:
        HighResWaitableTimer() {
            handle_ = CreateWaitableTimerExW(
                nullptr,
                nullptr,
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS
            );
        }
        ~HighResWaitableTimer() {
            if (handle_ != nullptr) {
                CloseHandle(handle_);
                handle_ = nullptr;
            }
        }
        HighResWaitableTimer(const HighResWaitableTimer&) = delete;
        HighResWaitableTimer& operator=(const HighResWaitableTimer&) = delete;

        void wait_until(
            std::chrono::steady_clock::time_point deadline,
            const std::atomic<bool>& stop_req
        ) const {
            while (!stop_req.load()) {
                const auto now_tp = std::chrono::steady_clock::now();
                if (now_tp >= deadline) break;
                const auto rem = deadline - now_tp;
                if (handle_ != nullptr) {
                    const auto slice = std::min<std::chrono::microseconds>(
                        std::chrono::duration_cast<std::chrono::microseconds>(rem),
                        std::chrono::microseconds(50000)
                    );
                    if (slice.count() <= 0) break;
                    LARGE_INTEGER due_time{};
                    due_time.QuadPart = -static_cast<LONGLONG>(slice.count()) * 10LL;
                    if (SetWaitableTimer(handle_, &due_time, 0, nullptr, nullptr, FALSE)) {
                        WaitForSingleObject(handle_, INFINITE);
                        continue;
                    }
                }
                const auto fallback_ms = std::chrono::duration_cast<std::chrono::milliseconds>(rem);
                std::this_thread::sleep_for(
                    std::clamp(fallback_ms, std::chrono::milliseconds(1), std::chrono::milliseconds(15))
                );
            }
        }

    private:
        HANDLE handle_{nullptr};
    };

    HighResWaitableTimer pacer_timer;

    while (!stop_requested_.load()) {
        auto cap = std::make_shared<cv::VideoCapture>(device_index_, cv::CAP_DSHOW);
        std::string opened_backend = "DSHOW";
        if (!cap->isOpened()) {
            cap->open(device_index_, cv::CAP_ANY);
            opened_backend = "ANY";
        }
        if (!cap->isOpened()) {
            mark_camera_unavailable("CAMERA_NOT_CONNECTED");
            interruptible_sleep(std::chrono::milliseconds(1200));
            continue;
        }

        // Strict property sequence: buffer -> 1280x720 -> 30.0 FPS (UVC MJPG native) -> FOURCC('M','J','P','G') LAST!
        // Requesting 25.0 FPS on Logitech C920 DirectShow rounds down to 15.0 FPS because UVC only lists 30.0.
        cap->set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap->set(cv::CAP_PROP_FRAME_WIDTH, kDefaultCameraWidth);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, kDefaultCameraHeight);
        cap->set(cv::CAP_PROP_FPS, 30.0);
        cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        // Attempt DirectShow 3A hardware locking + readback (AF / AE / AWB)
        const bool af_ok = cap->set(cv::CAP_PROP_AUTOFOCUS, 0.0);
        const bool ae_ok = cap->set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
        const bool awb_ok = cap->set(cv::CAP_PROP_AUTO_WB, 0.0);
        const double af_val = cap->get(cv::CAP_PROP_AUTOFOCUS);
        const double ae_val = cap->get(cv::CAP_PROP_AUTO_EXPOSURE);
        const double awb_val = cap->get(cv::CAP_PROP_AUTO_WB);
        const bool hw_3a_ok = af_ok && ae_ok && awb_ok;
        const json hw_3a_json{
            {"locked", hw_3a_ok},
            {"autofocus", {{"set_ok", af_ok}, {"value", std::isfinite(af_val) ? json(af_val) : json(nullptr)}}},
            {"auto_exposure", {{"set_ok", ae_ok}, {"value", std::isfinite(ae_val) ? json(ae_val) : json(nullptr)}}},
            {"auto_wb", {{"set_ok", awb_ok}, {"value", std::isfinite(awb_val) ? json(awb_val) : json(nullptr)}}}
        };

        {
            std::lock_guard<std::mutex> lk(mutex_);
            backend_ = opened_backend;
            fps_times_.clear();
            hardware_3a_locked_ = hw_3a_ok;
            hardware_3a_status_ = hw_3a_json;
            state_["hardware_3a"] = hardware_3a_status_;
        }

        // Decouple blocking UVC cap.read() from the 40.0 ms (25.0 FPS) consumer pacer to eliminate
        // 30 Hz UVC VSync aliasing and Windows 15.625 ms timer quantum oversleep.
        struct GrabberState {
            std::mutex mtx;
            std::condition_variable cv;
            cv::Mat latest_frame;
            uint64_t latest_seq{0};
            std::chrono::steady_clock::time_point last_frame_tp{};
            bool grab_failed{false};
            std::atomic<bool> stop{false};
        };
        auto grab_state = std::make_shared<GrabberState>();

        std::thread grabber([cap, grab_state]() {
            while (!grab_state->stop.load()) {
                cv::Mat raw;
                const bool ok = cap->read(raw);
                const auto grabbed_tp = std::chrono::steady_clock::now();
                if (!ok || raw.empty()) {
                    {
                        std::lock_guard<std::mutex> glk(grab_state->mtx);
                        grab_state->grab_failed = true;
                    }
                    grab_state->cv.notify_all();
                    break;
                }
                {
                    std::lock_guard<std::mutex> glk(grab_state->mtx);
                    grab_state->latest_frame = std::move(raw);
                    ++grab_state->latest_seq;
                    grab_state->last_frame_tp = grabbed_tp;
                }
                grab_state->cv.notify_all();
            }
        });

        auto stop_grabber_safely = [&]() {
            grab_state->stop.store(true);
            grab_state->cv.notify_all();
            try {
                cap->release();
            } catch (...) {
            }
            if (grabber.joinable()) {
                HANDLE hThread = reinterpret_cast<HANDLE>(grabber.native_handle());
                const DWORD wait_rc = (hThread != nullptr) ? WaitForSingleObject(hThread, 800) : WAIT_TIMEOUT;
                if (wait_rc == WAIT_OBJECT_0) {
                    grabber.join();
                } else {
                    std::cerr << "[IGBT-C++20] Warning: DirectShow grabber thread hung after cap.release(); detaching."
                              << std::endl;
                    grabber.detach();
                }
            }
        };

        constexpr auto kHardwareStallTimeout = std::chrono::milliseconds(1500);
        const auto first_wait_deadline = std::chrono::steady_clock::now() + kHardwareStallTimeout;
        bool first_frame_ready = false;
        {
            std::unique_lock<std::mutex> glk(grab_state->mtx);
            while (!stop_requested_.load() && !grab_state->grab_failed && grab_state->latest_seq == 0) {
                const auto now_tp = std::chrono::steady_clock::now();
                if (now_tp >= first_wait_deadline) break;
                const auto step = std::min<std::chrono::milliseconds>(
                    std::chrono::milliseconds(50),
                    std::chrono::duration_cast<std::chrono::milliseconds>(first_wait_deadline - now_tp)
                );
                grab_state->cv.wait_for(glk, std::max(std::chrono::milliseconds(1), step));
            }
            first_frame_ready = (grab_state->latest_seq > 0 && !grab_state->grab_failed);
        }

        if (!first_frame_ready) {
            if (!stop_requested_.load()) {
                mark_camera_unavailable("CAMERA_FEED_LOST");
            }
            stop_grabber_safely();
            if (!stop_requested_.load()) {
                interruptible_sleep(std::chrono::milliseconds(400));
            }
            continue;
        }

        auto deadline = std::chrono::steady_clock::now();

        while (!stop_requested_.load()) {
            deadline = next_pacer_deadline(deadline, std::chrono::steady_clock::now(), kFrameInterval);
            pacer_timer.wait_until(deadline, stop_requested_);
            if (stop_requested_.load()) break;

            cv::Mat frame;
            bool failed = false;
            bool stalled = false;
            {
                std::lock_guard<std::mutex> glk(grab_state->mtx);
                failed = grab_state->grab_failed;
                if (!failed && grab_state->latest_seq > 0) {
                    const auto age = std::chrono::steady_clock::now() - grab_state->last_frame_tp;
                    if (age > kHardwareStallTimeout) {
                        stalled = true;
                    } else {
                        frame = grab_state->latest_frame.clone();
                    }
                }
            }
            const double capture_ts = monotonic_seconds();
            if (stop_requested_.load()) break;

            if (failed || stalled || frame.empty()) {
                mark_camera_unavailable("CAMERA_FEED_LOST");
                break;
            }

            if (frame.cols != kDefaultCameraWidth || frame.rows != kDefaultCameraHeight) {
                cv::resize(
                    frame,
                    frame,
                    cv::Size(kDefaultCameraWidth, kDefaultCameraHeight),
                    0,
                    0,
                    cv::INTER_AREA
                );
            }

            process_frame_internal(frame, capture_ts);
        }

        stop_grabber_safely();
        if (!stop_requested_.load()) {
            interruptible_sleep(std::chrono::milliseconds(400));
        }
    }

    mark_camera_unavailable("CAMERA_NOT_CONNECTED");
}

bool LiveCameraEngine::process_injected_frame(
    const cv::Mat& raw_bgr,
    std::optional<double> capture_ts
) {
    if (raw_bgr.empty()) return false;
    cv::Mat frame = raw_bgr;
    if (frame.cols != kDefaultCameraWidth || frame.rows != kDefaultCameraHeight) {
        cv::resize(
            raw_bgr,
            frame,
            cv::Size(kDefaultCameraWidth, kDefaultCameraHeight),
            0,
            0,
            cv::INTER_AREA
        );
    }
    const double ts = capture_ts.has_value() ? *capture_ts : monotonic_seconds();
    return process_frame_internal(frame, ts);
}

// ============================================================================
// Dynamic Multi-Target Vision, Calibration Gate & Inference Pipeline
// ============================================================================

void LiveCameraEngine::reset_tracking_state() {
    bg_small_.release();
    prev_small_.release();
    tracked_label_box_.reset();
    tracked_tube_box_.reset();
    label_rel_offset_ = {0.0, 0.0, 0.30, 0.72};
    tracked_left_glove_.reset();
    tracked_right_glove_.reset();
    tracked_bare_hand_.reset();
    left_glove_ttl_ = 0;
    right_glove_ttl_ = 0;
    bare_hand_ttl_ = 0;
    label_ttl_ = 0;
    tube_ttl_ = 0;
    target_locked_ = false;
    target_confidence_ = 0.0;
}

std::tuple<ROISet, json, std::string, json> LiveCameraEngine::active_rois_and_thresholds() const {
    ConfigLoader loader;
    json fallback_cfg;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        loader = config_loader_;
        fallback_cfg = cached_config_;
    }

    json cfg = fallback_cfg;
    if (loader) {
        try {
            json loaded = loader();
            if (loaded.is_object() && !loaded.empty()) {
                cfg = std::move(loaded);
            }
        } catch (...) {
        }
    }

    ROISet rois{};
    if (cfg.is_object() && cfg.contains("rois") && cfg["rois"].is_object()) {
        const json& r = cfg["rois"];
        const auto t_box = r.contains("tube_roi") ? json_to_box(r["tube_roi"]) : std::nullopt;
        const auto l_box = r.contains("label_roi") ? json_to_box(r["label_roi"]) : std::nullopt;
        const auto c_box = r.contains("label_contact_roi") ? json_to_box(r["label_contact_roi"]) : std::nullopt;
        if (t_box.has_value() && l_box.has_value() && c_box.has_value()) {
            rois.tube_roi = *t_box;
            rois.label_roi = *l_box;
            rois.label_contact_roi = *c_box;
            if (r.contains("left_grip_zone")) {
                if (const auto lg = json_to_box(r["left_grip_zone"]); lg.has_value()) {
                    rois.left_grip_zone = *lg;
                }
            }
            if (r.contains("right_grip_zone")) {
                if (const auto rg = json_to_box(r["right_grip_zone"]); rg.has_value()) {
                    rois.right_grip_zone = *rg;
                }
            }
        }
    }

    json thresholds = (cfg.is_object() && cfg.contains("thresholds") && cfg["thresholds"].is_object())
                          ? cfg["thresholds"]
                          : json::object();
    std::string version = "ROI_UNCALIBRATED";
    if (cfg.is_object() && cfg.contains("roi_config_version") && cfg["roi_config_version"].is_string()) {
        const std::string v = cfg["roi_config_version"].get<std::string>();
        if (!v.empty()) version = v;
    }
    return {rois, thresholds, version, cfg};
}

std::tuple<bool, bool, std::vector<std::string>> LiveCameraEngine::evaluate_live_calibration(
    const json& cfg,
    int frame_w,
    int frame_h,
    const std::string& work_mode
) const {
    if (work_mode != "strict") {
        return {true, true, {}};
    }

    std::vector<std::string> reasons;
    const std::string roi_ver = cfg.is_object() ? cfg.value("roi_config_version", std::string("")) : "";
    const json raw_rois = (cfg.is_object() && cfg.contains("rois") && cfg["rois"].is_object())
                              ? cfg["rois"]
                              : json::object();
    const bool both_ends = cfg.is_object() ? cfg.value("camera_covers_both_ends", true) : true;

    bool has_valid_rois =
        raw_rois.contains("tube_roi") && json_to_box(raw_rois["tube_roi"]).has_value() &&
        raw_rois.contains("label_roi") && json_to_box(raw_rois["label_roi"]).has_value() &&
        raw_rois.contains("label_contact_roi") && json_to_box(raw_rois["label_contact_roi"]).has_value();
    if (both_ends && has_valid_rois) {
        has_valid_rois =
            raw_rois.contains("left_grip_zone") && json_to_box(raw_rois["left_grip_zone"]).has_value() &&
            raw_rois.contains("right_grip_zone") && json_to_box(raw_rois["right_grip_zone"]).has_value();
    }

    bool explicitly_uncalibrated = roi_ver.empty() || roi_ver == "ROI_UNCALIBRATED";
    if (cfg.is_object() && cfg.contains("config_version") && cfg["config_version"].is_number_integer()) {
        if (cfg["config_version"].get<int>() <= 0) explicitly_uncalibrated = true;
    }
    if (cfg.is_object() && cfg.contains("calibrated_at_utc")) {
        if (cfg["calibrated_at_utc"].is_null() ||
            (cfg["calibrated_at_utc"].is_string() && cfg["calibrated_at_utc"].get<std::string>().empty())) {
            explicitly_uncalibrated = true;
        }
    }
    if (cfg.is_object() && cfg.contains("roi_calibrated") && cfg["roi_calibrated"].is_boolean()) {
        if (!cfg["roi_calibrated"].get<bool>()) explicitly_uncalibrated = true;
    }

    const bool roi_calibrated = has_valid_rois && !explicitly_uncalibrated;
    if (!roi_calibrated) {
        reasons.push_back("ROI_UNCALIBRATED");
    }

    const int cfg_w = (cfg.is_object() && cfg.contains("camera_width") && cfg["camera_width"].is_number_integer())
                          ? cfg["camera_width"].get<int>()
                          : 0;
    const int cfg_h = (cfg.is_object() && cfg.contains("camera_height") && cfg["camera_height"].is_number_integer())
                          ? cfg["camera_height"].get<int>()
                          : 0;
    const bool dims_match = (cfg_w == frame_w && cfg_h == frame_h);
    if (!dims_match) {
        reasons.push_back("CALIBRATION_RESOLUTION_MISMATCH");
    }

    bool cam_match = true;
    if (cfg.is_object() && cfg.contains("camera_device_id") && cfg["camera_device_id"].is_string()) {
        const std::string cfg_cam = cfg["camera_device_id"].get<std::string>();
        std::lock_guard<std::mutex> lk(mutex_);
        if (!cfg_cam.empty() && cfg_cam != camera_device_id_ && cfg_cam != "CAM_STATION_01") {
            cam_match = false;
        }
    }
    if (!cam_match) {
        reasons.push_back("CAMERA_DEVICE_MISMATCH");
    }

    const bool config_match = roi_calibrated && dims_match && cam_match;
    return {config_match, config_match, reasons};
}

std::pair<std::optional<Box>, double> LiveCameraEngine::detect_tube_body(
    const cv::Mat& hsv,
    const cv::Mat& edges,
    const cv::Mat& edges_dilated,
    const cv::Mat& fg_diff,
    const cv::Mat& person_exclusion,
    int sw,
    int sh,
    bool already_locked
) const {
    cv::Mat yellow_tube, blue_tube, color_tube, fg_tube, tube_mask, not_person;
    cv::inRange(hsv, cv::Scalar(15, 45, 65), cv::Scalar(42, 255, 255), yellow_tube);
    cv::inRange(hsv, cv::Scalar(92, 45, 55), cv::Scalar(138, 255, 255), blue_tube);
    cv::compare(person_exclusion, 0, not_person, cv::CMP_EQ);

    cv::bitwise_or(yellow_tube, blue_tube, color_tube);
    cv::bitwise_and(color_tube, not_person, color_tube);

    cv::Mat fg_gt;
    cv::compare(fg_diff, 14.0f, fg_gt, cv::CMP_GT);
    cv::bitwise_and(fg_gt, edges_dilated, fg_tube);
    cv::bitwise_and(fg_tube, not_person, fg_tube);

    cv::bitwise_or(color_tube, fg_tube, tube_mask);
    cv::morphologyEx(
        tube_mask,
        tube_mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(19, 5))
    );
    cv::morphologyEx(
        tube_mask,
        tube_mask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3))
    );

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(tube_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::optional<Box> best_box;
    double best_score = -1.0;
    const bool glove_present = tracked_left_glove_.has_value() || tracked_right_glove_.has_value();

    for (const auto& cnt : contours) {
        const double area = cv::contourArea(cnt);
        if (area < sw * sh * 0.010 || area > sw * sh * 0.42) continue;
        const cv::Rect r = cv::boundingRect(cnt);
        if (static_cast<double>(r.width) / sw < 0.16 ||
            static_cast<double>(r.height) / sh < 0.030 ||
            static_cast<double>(r.height) / sh > 0.30) {
            continue;
        }
        const double aspect = static_cast<double>(r.width) / std::max(1, r.height);
        if (aspect < 2.0) continue;

        const double cx = (r.x + r.width * 0.5) / sw;
        const double cy = (r.y + r.height * 0.5) / sh;
        if (cy < 0.10 || cy > 0.88) continue;
        if (person_exclusion.at<uint8_t>(
                std::clamp(static_cast<int>(cy * sh), 0, sh - 1),
                std::clamp(static_cast<int>(cx * sw), 0, sw - 1)) > 0) {
            continue;
        }

        const double denom = static_cast<double>(r.width * r.height);
        const double patch_color = static_cast<double>(cv::countNonZero(color_tube(r))) / denom;
        const double patch_edge = static_cast<double>(cv::countNonZero(edges(r))) / denom;
        const double patch_fg = static_cast<double>(cv::countNonZero(fg_gt(r))) / denom;

        bool supported = false;
        if (already_locked) {
            supported = patch_color >= 0.12 || (patch_edge >= 0.06 && patch_fg >= 0.12);
        } else {
            supported = (patch_color >= 0.28 && (patch_fg >= 0.12 || glove_present || aspect >= 2.6)) ||
                        (patch_edge >= 0.10 && patch_fg >= 0.24 && glove_present);
        }
        if (!supported) continue;

        const Box cand_box = clamp_box(
            static_cast<double>(r.x) / sw,
            static_cast<double>(r.y) / sh,
            static_cast<double>(r.x + r.width) / sw,
            static_cast<double>(r.y + r.height) / sh,
            0.16,
            0.05
        );
        double continuity = 0.0;
        if (tracked_tube_box_.has_value()) {
            const auto [iou_p, cov_p] = box_iou_and_overlap(cand_box, *tracked_tube_box_);
            continuity = iou_p * 2.0 + cov_p * 0.8;
        }
        const double score = patch_color * 2.2 + patch_edge * 1.4 + std::min(aspect, 6.0) * 0.25 + continuity;
        if (score > best_score) {
            best_score = score;
            best_box = cand_box;
        }
    }

    // Sustain lock across C-class 1200ms removal window even if contour splits across central label window
    if (!best_box.has_value() && already_locked && tracked_tube_box_.has_value()) {
        const Box& tb = *tracked_tube_box_;
        const int px1 = std::clamp(static_cast<int>(tb[0] * sw), 0, sw - 2);
        const int py1 = std::clamp(static_cast<int>(tb[1] * sh), 0, sh - 2);
        const int px2 = std::clamp(static_cast<int>(tb[2] * sw), px1 + 2, sw);
        const int py2 = std::clamp(static_cast<int>(tb[3] * sh), py1 + 2, sh);
        const cv::Rect reg(px1, py1, px2 - px1, py2 - py1);
        const double denom = static_cast<double>(reg.width * reg.height);
        const double reg_color = static_cast<double>(cv::countNonZero(color_tube(reg))) / denom;
        const double reg_edge = static_cast<double>(cv::countNonZero(edges(reg))) / denom;
        const double reg_fg = static_cast<double>(cv::countNonZero(fg_gt(reg))) / denom;
        const bool glove_holding =
            (tracked_left_glove_.has_value() && box_iou_and_overlap(*tracked_left_glove_, tb).first > 0.02) ||
            (tracked_right_glove_.has_value() && box_iou_and_overlap(*tracked_right_glove_, tb).first > 0.02);
        if (reg_color >= 0.12 || (reg_edge >= 0.07 && reg_fg >= 0.11) || (glove_holding && reg_color >= 0.05)) {
            best_box = tb;
            best_score = std::max(0.85, reg_color * 2.0 + reg_edge);
        }
    }

    return {best_box, best_score};
}

Box LiveCameraEngine::project_label_window(const Box& tube_box) const {
    const double tw = std::max(0.10, tube_box[2] - tube_box[0]);
    const double th = std::max(0.06, tube_box[3] - tube_box[1]);
    const double tcx = (tube_box[0] + tube_box[2]) * 0.5;
    const double tcy = (tube_box[1] + tube_box[3]) * 0.5;
    const double lcx = tcx + label_rel_offset_[0] * tw;
    const double lcy = tcy + label_rel_offset_[1] * th;
    const double lw = std::clamp(label_rel_offset_[2] * tw, 0.11, 0.24);
    const double lh = std::clamp(label_rel_offset_[3] * th, 0.084, 0.18);
    return clamp_box(lcx - lw * 0.5, lcy - lh * 0.5, lcx + lw * 0.5, lcy + lh * 0.5, 0.08, 0.06);
}

LiveCameraEngine::TrackResult LiveCameraEngine::track_dynamic_targets(
    const cv::Mat& frame,
    const std::string& work_mode
) {
    constexpr int sw = 384;
    constexpr int sh = 216;
    cv::Mat small, gray_u8, gray, blur;
    cv::resize(frame, small, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small, gray_u8, cv::COLOR_BGR2GRAY);
    gray_u8.convertTo(gray, CV_32F);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0.0);

    if (bg_small_.empty() || bg_small_.size() != blur.size()) {
        bg_small_ = blur.clone();
    }
    if (prev_small_.empty() || prev_small_.size() != blur.size()) {
        prev_small_ = blur.clone();
    }

    cv::Mat fg_diff, motion_diff;
    cv::absdiff(blur, bg_small_, fg_diff);
    cv::absdiff(blur, prev_small_, motion_diff);
    cv::addWeighted(prev_small_, 0.55, blur, 0.45, 0.0, prev_small_);

    for (int y = 0; y < sh; ++y) {
        const float* mp = motion_diff.ptr<float>(y);
        const float* bp = blur.ptr<float>(y);
        float* bgp = bg_small_.ptr<float>(y);
        for (int x = 0; x < sw; ++x) {
            if (mp[x] < 7.0f) {
                bgp[x] = bgp[x] * 0.992f + bp[x] * 0.008f;
            }
        }
    }

    // Build face/neck/collar exclusion mask (pure core/imgproc, zero objdetect link dependency)
    cv::Mat person_exclusion = cv::Mat::zeros(sh, sw, CV_8U);

    cv::Mat ycrcb, hsv;
    cv::cvtColor(small, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(small, hsv, cv::COLOR_BGR2HSV);

    cv::Mat skin_mask = cv::Mat::zeros(sh, sw, CV_8U);
    for (int y = 0; y < sh; ++y) {
        const auto* hp = hsv.ptr<cv::Vec3b>(y);
        const auto* yp = ycrcb.ptr<cv::Vec3b>(y);
        uint8_t* sp = skin_mask.ptr<uint8_t>(y);
        for (int x = 0; x < sw; ++x) {
            if (hp[x][2] >= 125 &&
                yp[x][1] >= 137 && yp[x][1] <= 174 &&
                yp[x][2] >= 80 && yp[x][2] <= 122) {
                sp[x] = 255;
            }
        }
    }

    // Exclude upper-half head/face/neck blobs and upper-center skin column
    {
        const int top_h = static_cast<int>(sh * 0.44);
        cv::Mat upper_skin = skin_mask(cv::Rect(0, 0, sw, top_h)).clone();
        cv::morphologyEx(upper_skin, upper_skin, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
        std::vector<std::vector<cv::Point>> head_cnts;
        cv::findContours(upper_skin, head_cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& hc : head_cnts) {
            const double h_area = cv::contourArea(hc);
            if (h_area < sw * sh * 0.008) continue;
            const cv::Rect hr = cv::boundingRect(hc);
            if (hr.y < static_cast<int>(sh * 0.25) && hr.width >= 24 && hr.height >= 24) {
                const int ex1 = std::max(0, static_cast<int>(hr.x - hr.width * 0.25));
                const int ey1 = 0;
                const int ex2 = std::min(sw, static_cast<int>(hr.x + hr.width * 1.25));
                const int ey2 = std::min(static_cast<int>(sh * 0.54), static_cast<int>(hr.y + hr.height * 1.45));
                if (ex2 > ex1 && ey2 > ey1) {
                    person_exclusion(cv::Rect(ex1, ey1, ex2 - ex1, ey2 - ey1)).setTo(255);
                }
            }
        }
    }

    cv::Rect top_center(static_cast<int>(sw * 0.30), 0, static_cast<int>(sw * 0.40), static_cast<int>(sh * 0.40));
    if (static_cast<double>(cv::countNonZero(skin_mask(top_center))) /
            static_cast<double>(top_center.area()) > 0.08) {
        person_exclusion(cv::Rect(static_cast<int>(sw * 0.30), 0,
                                  static_cast<int>(sw * 0.40), static_cast<int>(sh * 0.50)))
            .setTo(255);
    }

    const cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    const cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    const double min_hand_area = sw * sh * 0.010;
    const double max_hand_area = sw * sh * 0.11;

    // A. Compliant White / Light-Gray ESD Gloves
    cv::Mat glove_raw = cv::Mat::zeros(sh, sw, CV_8U);
    for (int y = 0; y < sh; ++y) {
        const auto* hp = hsv.ptr<cv::Vec3b>(y);
        const float* mp = motion_diff.ptr<float>(y);
        const float* fp = fg_diff.ptr<float>(y);
        const uint8_t* pp = person_exclusion.ptr<uint8_t>(y);
        uint8_t* gp = glove_raw.ptr<uint8_t>(y);
        for (int x = 0; x < sw; ++x) {
            if (pp[x] == 0 && hp[x][2] >= 178 && hp[x][1] <= 42 && (mp[x] > 7.5f || fp[x] > 20.0f)) {
                gp[x] = 255;
            }
        }
    }
    cv::Mat glove_clean;
    cv::morphologyEx(glove_raw, glove_clean, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(glove_clean, glove_clean, cv::MORPH_CLOSE, kernel_close);
    std::vector<std::vector<cv::Point>> glove_contours;
    cv::findContours(glove_clean, glove_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::optional<Box> left_glove_cand, right_glove_cand;
    double lg_area = 0.0, rg_area = 0.0;
    cv::Mat mot_gt_65;
    cv::compare(motion_diff, 6.5f, mot_gt_65, cv::CMP_GT);
    for (const auto& cnt : glove_contours) {
        const double area = cv::contourArea(cnt);
        if (area < min_hand_area || area > max_hand_area) continue;
        const cv::Rect r = cv::boundingRect(cnt);
        const double cx = (r.x + r.width * 0.5) / sw;
        if (r.y <= 4 || static_cast<double>(r.y) / sh >= 0.82 ||
            static_cast<double>(r.width) / sw > 0.25 ||
            static_cast<double>(r.height) / sh > 0.40) {
            continue;
        }
        const double mot = static_cast<double>(cv::countNonZero(mot_gt_65(r))) /
                           static_cast<double>(r.width * r.height);
        if (mot < 0.12) continue;
        const Box b{
            std::max(0.01, static_cast<double>(r.x) / sw),
            std::max(0.02, static_cast<double>(r.y) / sh),
            std::min(0.99, static_cast<double>(r.x + r.width) / sw),
            std::min(0.99, static_cast<double>(r.y + r.height) / sh),
        };
        if (cx < 0.50 && area > lg_area) {
            lg_area = area;
            left_glove_cand = b;
        } else if (cx >= 0.50 && area > rg_area) {
            rg_area = area;
            right_glove_cand = b;
        }
    }

    if (left_glove_cand.has_value()) {
        tracked_left_glove_ = ema_box(tracked_left_glove_, *left_glove_cand, 0.50);
        left_glove_ttl_ = 5;
    } else if (left_glove_ttl_ > 0) {
        --left_glove_ttl_;
    } else {
        tracked_left_glove_.reset();
    }

    if (right_glove_cand.has_value()) {
        tracked_right_glove_ = ema_box(tracked_right_glove_, *right_glove_cand, 0.50);
        right_glove_ttl_ = 5;
    } else if (right_glove_ttl_ > 0) {
        --right_glove_ttl_;
    } else {
        tracked_right_glove_.reset();
    }

    // B. Bare-Hand Skin Detection
    cv::Mat not_person, bare_raw, bare_clean;
    cv::compare(person_exclusion, 0, not_person, cv::CMP_EQ);
    cv::bitwise_and(skin_mask, not_person, bare_raw);
    cv::morphologyEx(bare_raw, bare_clean, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(bare_clean, bare_clean, cv::MORPH_CLOSE, kernel_close);
    std::vector<std::vector<cv::Point>> bare_contours;
    cv::findContours(bare_clean, bare_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::optional<Box> bare_cand;
    double bare_best_area = 0.0;
    cv::Mat fg_gt_14, mot_gt_55;
    cv::compare(fg_diff, 14.0f, fg_gt_14, cv::CMP_GT);
    cv::compare(motion_diff, 5.5f, mot_gt_55, cv::CMP_GT);
    for (const auto& cnt : bare_contours) {
        const double area = cv::contourArea(cnt);
        if (area < min_hand_area || area > max_hand_area) continue;
        const cv::Rect r = cv::boundingRect(cnt);
        const double cx = (r.x + r.width * 0.5) / sw;
        if (r.y <= 4 && cx > 0.22 && cx < 0.78) continue;
        if (static_cast<double>(r.y) / sh >= 0.82) continue;
        const double denom = static_cast<double>(r.width * r.height);
        const double hand_fg = static_cast<double>(cv::countNonZero(fg_gt_14(r))) / denom;
        const double hand_mot = static_cast<double>(cv::countNonZero(mot_gt_55(r))) / denom;
        if (hand_fg < 0.18 && hand_mot < 0.08) continue;
        if (static_cast<double>(r.width) / sw > 0.26 || static_cast<double>(r.height) / sh > 0.42) continue;
        if (area > bare_best_area) {
            bare_best_area = area;
            bare_cand = Box{
                std::max(0.01, static_cast<double>(r.x) / sw),
                std::max(0.02, static_cast<double>(r.y) / sh),
                std::min(0.99, static_cast<double>(r.x + r.width) / sw),
                std::min(0.99, static_cast<double>(r.y + r.height) / sh),
            };
        }
    }

    if (bare_cand.has_value()) {
        tracked_bare_hand_ = ema_box(tracked_bare_hand_, *bare_cand, 0.50);
        bare_hand_ttl_ = 5;
    } else if (bare_hand_ttl_ > 0) {
        --bare_hand_ttl_;
    } else {
        tracked_bare_hand_.reset();
    }

    // C. Elongated Tube Body + Rectangular White Label Tracking
    cv::Mat blur_u8, edges, edges_dilated, bright_patch, label_candidates_mask, mot_gt_75;
    blur.convertTo(blur_u8, CV_8U);
    cv::Canny(blur_u8, edges, 55, 150);
    cv::dilate(edges, edges_dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::compare(motion_diff, 7.5f, mot_gt_75, cv::CMP_GT);

    bright_patch = cv::Mat::zeros(sh, sw, CV_8U);
    for (int y = 0; y < sh; ++y) {
        const auto* hp = hsv.ptr<cv::Vec3b>(y);
        const uint8_t* pp = person_exclusion.ptr<uint8_t>(y);
        uint8_t* bp = bright_patch.ptr<uint8_t>(y);
        for (int x = 0; x < sw; ++x) {
            if (pp[x] == 0 && hp[x][2] > 175 && hp[x][1] < 55) {
                bp[x] = 255;
            }
        }
    }
    cv::bitwise_and(bright_patch, edges_dilated, label_candidates_mask);
    cv::morphologyEx(
        label_candidates_mask,
        label_candidates_mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 7))
    );

    std::vector<std::vector<cv::Point>> feature_contours;
    cv::findContours(label_candidates_mask, feature_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::optional<Box> best_label_box;
    double best_score = -1.0;

    for (const auto& cnt : feature_contours) {
        const double area = cv::contourArea(cnt);
        if (area < sw * sh * 0.0035 || area > sw * sh * 0.060) continue;
        const cv::Rect r = cv::boundingRect(cnt);
        const double aspect = static_cast<double>(r.width) / std::max(1, r.height);
        if (aspect < 1.20 || aspect > 4.2) continue;
        const double rectangularity = area / std::max(1.0, static_cast<double>(r.width * r.height));
        if (rectangularity < 0.58) continue;

        const double cx = (r.x + r.width * 0.5) / sw;
        const double cy = (r.y + r.height * 0.5) / sh;
        if (cx < 0.08 || cx > 0.92 || cy < 0.08 || cy > 0.84) continue;
        if (person_exclusion.at<uint8_t>(
                std::clamp(static_cast<int>(cy * sh), 0, sh - 1),
                std::clamp(static_cast<int>(cx * sw), 0, sw - 1)) > 0) {
            continue;
        }

        const double denom = static_cast<double>(r.width * r.height);
        const double patch_bright = static_cast<double>(cv::countNonZero(bright_patch(r))) / denom;
        const double patch_edge = static_cast<double>(cv::countNonZero(edges(r))) / denom;
        const double patch_motion = static_cast<double>(cv::countNonZero(mot_gt_75(r))) / denom;
        if (patch_bright < 0.35 || patch_edge < 0.09) continue;

        const double glove_bonus = (tracked_left_glove_.has_value() || tracked_right_glove_.has_value()) ? 0.45 : 0.0;
        const double score = patch_bright * 1.4 + patch_edge * 2.5 + rectangularity * 1.2 + patch_motion * 1.8 + glove_bonus;
        if (score > best_score) {
            best_score = score;
            const double hw = std::clamp((static_cast<double>(r.width) / sw) * 0.65, 0.055, 0.11);
            const double hh = std::clamp((static_cast<double>(r.height) / sh) * 0.65, 0.042, 0.085);
            best_label_box = Box{
                std::max(0.04, cx - hw),
                std::max(0.06, cy - hh),
                std::min(0.96, cx + hw),
                std::min(0.94, cy + hh),
            };
        }
    }

    const auto [tube_body_box, tube_body_score] = detect_tube_body(
        hsv, edges, edges_dilated, fg_diff, person_exclusion, sw, sh, target_locked_
    );

    const double min_lock_score = (work_mode == "strict") ? 1.72 : 1.20;
    const bool label_detected = best_label_box.has_value() && best_score >= min_lock_score;
    const bool tube_detected = tube_body_box.has_value();

    if (label_detected) {
        tracked_label_box_ = ema_box(tracked_label_box_, *best_label_box, 0.45);
        label_ttl_ = 6;
        target_locked_ = true;
        target_confidence_ = round_to(std::clamp(0.65 + best_score * 0.12, 0.78, 0.98), 3);
        if (tube_detected) {
            tracked_tube_box_ = ema_box(tracked_tube_box_, *tube_body_box, 0.42);
            tube_ttl_ = 10;
        } else {
            const double lc_x = ((*tracked_label_box_)[0] + (*tracked_label_box_)[2]) * 0.5;
            const double lc_y = ((*tracked_label_box_)[1] + (*tracked_label_box_)[3]) * 0.5;
            const double tx1 = std::max(0.05, std::min(lc_x - 0.24, tracked_left_glove_.has_value() ? (*tracked_left_glove_)[0] : lc_x - 0.24));
            const double tx2 = std::min(0.95, std::max(lc_x + 0.24, tracked_right_glove_.has_value() ? (*tracked_right_glove_)[2] : lc_x + 0.24));
            const double ty1 = std::max(0.08, lc_y - 0.095);
            const double ty2 = std::min(0.92, lc_y + 0.095);
            tracked_tube_box_ = ema_box(tracked_tube_box_, Box{tx1, ty1, tx2, ty2}, 0.42);
            tube_ttl_ = std::max(tube_ttl_, 6);
        }
        if (tracked_tube_box_.has_value() && tracked_label_box_.has_value()) {
            const double tw = std::max(0.10, (*tracked_tube_box_)[2] - (*tracked_tube_box_)[0]);
            const double th = std::max(0.06, (*tracked_tube_box_)[3] - (*tracked_tube_box_)[1]);
            const double tcx = ((*tracked_tube_box_)[0] + (*tracked_tube_box_)[2]) * 0.5;
            const double tcy = ((*tracked_tube_box_)[1] + (*tracked_tube_box_)[3]) * 0.5;
            const double lcx = ((*tracked_label_box_)[0] + (*tracked_label_box_)[2]) * 0.5;
            const double lcy = ((*tracked_label_box_)[1] + (*tracked_label_box_)[3]) * 0.5;
            const double lw = (*tracked_label_box_)[2] - (*tracked_label_box_)[0];
            const double lh = (*tracked_label_box_)[3] - (*tracked_label_box_)[1];
            label_rel_offset_ = {
                std::clamp((lcx - tcx) / tw, -0.35, 0.35),
                std::clamp((lcy - tcy) / th, -0.35, 0.35),
                std::clamp(lw / tw, 0.18, 0.55),
                std::clamp(lh / th, 0.45, 1.0),
            };
        }
    } else if (tube_detected) {
        // Tube remains visible while label is covered or peeled off (C-class 1200ms confirmation)
        tracked_tube_box_ = ema_box(tracked_tube_box_, *tube_body_box, 0.35);
        tube_ttl_ = 10;
        if (label_ttl_ > 0) --label_ttl_;
        const Box projected = project_label_window(*tracked_tube_box_);
        tracked_label_box_ = ema_box(tracked_label_box_, projected, 0.35);
        target_locked_ = true;
        target_confidence_ = round_to(std::clamp(0.72 + tube_body_score * 0.08, 0.80, 0.95), 3);
    } else if ((label_ttl_ > 0 || tube_ttl_ > 0) && tracked_label_box_.has_value()) {
        if (label_ttl_ > 0) --label_ttl_;
        if (tube_ttl_ > 0) --tube_ttl_;
        target_locked_ = true;
    } else {
        tracked_label_box_.reset();
        tracked_tube_box_.reset();
        target_locked_ = false;
        target_confidence_ = 0.0;
    }

    TrackResult tr{};
    tr.tube_box = tracked_tube_box_;
    tr.label_box = tracked_label_box_;
    tr.left_glove = tracked_left_glove_;
    tr.right_glove = tracked_right_glove_;
    tr.bare_hand = tracked_bare_hand_;
    tr.target_locked = target_locked_;

    if (target_locked_ && tracked_label_box_.has_value()) {
        if (!tracked_tube_box_.has_value()) {
            const double lc_x = ((*tracked_label_box_)[0] + (*tracked_label_box_)[2]) * 0.5;
            const double lc_y = ((*tracked_label_box_)[1] + (*tracked_label_box_)[3]) * 0.5;
            tracked_tube_box_ = Box{
                std::max(0.05, lc_x - 0.24),
                std::max(0.08, lc_y - 0.095),
                std::min(0.95, lc_x + 0.24),
                std::min(0.92, lc_y + 0.095),
            };
            tr.tube_box = tracked_tube_box_;
        }

        const Box& tb = *tracked_tube_box_;
        const Box& lb = *tracked_label_box_;
        std::optional<Box> dyn_left_grip;
        std::optional<Box> dyn_right_grip;
        if (lb[0] - 0.03 > tb[0] + 0.02) {
            dyn_left_grip = Box{tb[0], tb[1], lb[0] - 0.03, tb[3]};
        }
        if (tb[2] - 0.02 > lb[2] + 0.03) {
            dyn_right_grip = Box{lb[2] + 0.03, tb[1], tb[2], tb[3]};
        }

        for (const auto& gopt : {tracked_left_glove_, tracked_right_glove_}) {
            if (!gopt.has_value()) continue;
            const auto [iou, cov] = box_iou_and_overlap(*gopt, lb);
            if (iou > 0.04 || cov > 0.10) {
                const bool in_end_grip =
                    (dyn_left_grip.has_value() && overlap_of_box_fraction(*gopt, *dyn_left_grip) >= 0.40) ||
                    (dyn_right_grip.has_value() && overlap_of_box_fraction(*gopt, *dyn_right_grip) >= 0.40);
                if (!in_end_grip || cov >= 0.25) {
                    tr.contact_active = true;
                    break;
                }
            }
        }

        if (tracked_bare_hand_.has_value()) {
            const auto [iou_t, cov_t] = box_iou_and_overlap(*tracked_bare_hand_, tb);
            const auto [iou_l, cov_l] = box_iou_and_overlap(*tracked_bare_hand_, lb);
            if (iou_t > 0.01 || cov_t > 0.03 || iou_l > 0.01 || cov_l > 0.03) {
                if (work_mode == "strict") {
                    tr.bare_hand_touching_part = true;
                } else if (iou_l > 0.04 || cov_l > 0.10) {
                    tr.contact_active = true;
                }
            }
        }

        tr.tracked_objects.push_back(json{
            {"kind", "tube"},
            {"label", "料管目标"},
            {"box", box_to_json(tracked_tube_box_)},
            {"confidence", round_to(std::max(0.82, target_confidence_), 3)},
        });
        tr.tracked_objects.push_back(json{
            {"kind", "label"},
            {"label", "防呆标签"},
            {"box", box_to_json(tracked_label_box_)},
            {"confidence", round_to(std::max(0.85, target_confidence_), 3)},
        });
    }

    if (tracked_left_glove_.has_value()) {
        tr.tracked_objects.push_back(json{
            {"kind", "glove_left"},
            {"label", "左手套(合规)"},
            {"box", box_to_json(tracked_left_glove_)},
            {"confidence", 0.94},
        });
    }
    if (tracked_right_glove_.has_value()) {
        tr.tracked_objects.push_back(json{
            {"kind", "glove_right"},
            {"label", "右手套(合规)"},
            {"box", box_to_json(tracked_right_glove_)},
            {"confidence", 0.94},
        });
    }
    if (tracked_bare_hand_.has_value()) {
        tr.tracked_objects.push_back(json{
            {"kind", "bare_hand_violation"},
            {"label", tr.bare_hand_touching_part ? "裸手违规" : "裸手警告"},
            {"box", box_to_json(tracked_bare_hand_)},
            {"confidence", 0.95},
        });
    }

    return tr;
}

LiveCameraEngine::ClassificationResult LiveCameraEngine::classify_live_label_roi(
    const cv::Mat& crop,
    bool target_locked,
    bool contact_seen,
    bool label_occluded
) {
    const double t0 = monotonic_seconds();
    ClassificationResult out{};
    if (!target_locked || crop.empty()) {
        return out;
    }

    cv::Mat small, hsv, gray;
    cv::resize(crop, small, cv::Size(160, 96), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);

    cv::Mat yellow_mask, blue_mask;
    cv::inRange(hsv, cv::Scalar(15, 50, 0), cv::Scalar(42, 255, 255), yellow_mask);
    cv::inRange(hsv, cv::Scalar(95, 50, 0), cv::Scalar(135, 255, 255), blue_mask);
    const double total_px = static_cast<double>(small.rows * small.cols);
    const double y_frac = static_cast<double>(cv::countNonZero(yellow_mask)) / total_px;
    const double b_frac = static_cast<double>(cv::countNonZero(blue_mask)) / total_px;
    out.pipe_color = (b_frac > y_frac + 0.06) ? "blue" : ((y_frac > 0.04) ? "yellow" : "unknown");

    if (label_occluded) {
        out.label_status = "unknown";
        out.confidence = 0.48;
        out.infer_ms = round_to((monotonic_seconds() - t0) * 1000.0, 2);
        return out;
    }

    std::optional<std::string> nn_pred;
    double nn_conf = 0.0;
    std::string nn_version;
    std::string nn_backend;

    {
        std::lock_guard<std::mutex> mlk(model_mutex_);
        if (active_model_kind_ == ActiveModelKind::MLP_XML && !mlp_model_.empty()) {
            try {
                const cv::Mat feat = extract_label_features(crop);
                cv::Mat raw_out;
                mlp_model_->predict(feat, raw_out);
                if (!raw_out.empty() && raw_out.total() >= 4) {
                    cv::Mat flat = raw_out.reshape(1, 1);
                    flat.convertTo(flat, CV_32F);
                    float max_v = flat.at<float>(0, 0);
                    for (int i = 1; i < 4; ++i) max_v = std::max(max_v, flat.at<float>(0, i));
                    // Temperature-calibrated softmax for SIGMOID_SYM [-1, +1] targets
                    constexpr float kScale = 2.2f;
                    std::array<double, 4> probs{};
                    double sum_exp = 0.0;
                    int best_idx = 0;
                    for (int i = 0; i < 4; ++i) {
                        probs[i] = std::exp(static_cast<double>(kScale * (flat.at<float>(0, i) - max_v)));
                        sum_exp += probs[i];
                        if (probs[i] > probs[best_idx]) best_idx = i;
                    }
                    if (sum_exp > 0.0) {
                        nn_pred = kLabelClasses[best_idx];
                        nn_conf = probs[best_idx] / sum_exp;
                        nn_version = active_model_version_;
                        nn_backend = "OpenCV ANN_MLP (" + active_model_version_ + ")";
                    }
                }
            } catch (...) {
            }
        } else if (active_model_kind_ == ActiveModelKind::ONNX_DNN && !onnx_net_.empty()) {
            try {
                cv::Mat rgb, resized, f32;
                cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
                cv::resize(rgb, resized, cv::Size(onnx_input_size_, onnx_input_size_), 0, 0, cv::INTER_LINEAR);
                resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

                const int dims[4] = {1, 3, onnx_input_size_, onnx_input_size_};
                cv::Mat blob(4, dims, CV_32F);
                constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
                constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
                const int plane_sz = onnx_input_size_ * onnx_input_size_;
                float* bp = blob.ptr<float>();
                for (int y = 0; y < onnx_input_size_; ++y) {
                    const auto* rp = f32.ptr<cv::Vec3f>(y);
                    for (int x = 0; x < onnx_input_size_; ++x) {
                        const int off = y * onnx_input_size_ + x;
                        bp[0 * plane_sz + off] = (rp[x][0] - kMean[0]) / kStd[0];
                        bp[1 * plane_sz + off] = (rp[x][1] - kMean[1]) / kStd[1];
                        bp[2 * plane_sz + off] = (rp[x][2] - kMean[2]) / kStd[2];
                    }
                }
                onnx_net_.setInput(blob);
                cv::Mat logits = onnx_net_.forward();
                if (!logits.empty() && logits.total() >= 4) {
                    cv::Mat flat = logits.reshape(1, 1);
                    flat.convertTo(flat, CV_32F);
                    float max_l = flat.at<float>(0, 0);
                    for (int i = 1; i < 4; ++i) max_l = std::max(max_l, flat.at<float>(0, i));
                    std::array<double, 4> probs{};
                    double sum_exp = 0.0;
                    int best_idx = 0;
                    for (int i = 0; i < 4; ++i) {
                        probs[i] = std::exp(static_cast<double>(flat.at<float>(0, i) - max_l));
                        sum_exp += probs[i];
                        if (probs[i] > probs[best_idx]) best_idx = i;
                    }
                    if (sum_exp > 0.0) {
                        nn_pred = kLabelClasses[best_idx];
                        nn_conf = probs[best_idx] / sum_exp;
                        nn_version = active_model_version_;
                        nn_backend = "OpenCV DNN ONNX (" + active_model_version_ + ")";
                    }
                }
            } catch (...) {
            }
        }
    }

    int white_cnt = 0;
    int dark_cnt = 0;
    for (int y = 0; y < small.rows; ++y) {
        const auto* hp = hsv.ptr<cv::Vec3b>(y);
        const uint8_t* gp = gray.ptr<uint8_t>(y);
        for (int x = 0; x < small.cols; ++x) {
            if (hp[x][2] > 180 && hp[x][1] < 55) ++white_cnt;
            if (gp[x] < 55) ++dark_cnt;
        }
    }
    const double white_ratio = static_cast<double>(white_cnt) / total_px;
    const double dark_stroke_ratio = static_cast<double>(dark_cnt) / total_px;
    cv::Scalar g_mean, g_std;
    cv::meanStdDev(gray, g_mean, g_std);
    cv::Mat edges;
    cv::Canny(gray, edges, 55, 150);
    const double edge_ratio = static_cast<double>(cv::countNonZero(edges)) / total_px;
    out.infer_ms = round_to((monotonic_seconds() - t0) * 1000.0, 2);

    // Neural model class-specific optical corroboration gate
    if (nn_pred.has_value() && nn_conf >= 0.80) {
        if (*nn_pred == "covered") {
            out.label_status = "covered";
            out.confidence = round_to(nn_conf, 3);
            out.model_version = nn_version;
            out.backend_label = nn_backend;
            return out;
        }
        if (*nn_pred == "absent" && white_ratio < 0.35) {
            out.label_status = "absent";
            out.confidence = round_to(nn_conf, 3);
            out.model_version = nn_version;
            out.backend_label = nn_backend;
            return out;
        }
        if (*nn_pred == "crossed_out" && (white_ratio >= 0.15 || edge_ratio >= 0.06)) {
            out.label_status = "crossed_out";
            out.confidence = round_to(nn_conf, 3);
            out.model_version = nn_version;
            out.backend_label = nn_backend;
            return out;
        }
        if (*nn_pred == "complete" && white_ratio >= 0.28) {
            out.label_status = "complete";
            out.confidence = round_to(nn_conf, 3);
            out.model_version = nn_version;
            out.backend_label = nn_backend;
            return out;
        }
    }

    // Heuristic fallback
    if (contact_seen && white_ratio < 0.22 && !(nn_pred.has_value() && *nn_pred == "covered" && nn_conf >= 0.65)) {
        out.label_status = "absent";
        out.confidence = round_to(std::min(0.97, 0.86 + (0.22 - white_ratio) * 0.4), 3);
        return out;
    }
    if (white_ratio >= 0.35 && dark_stroke_ratio >= 0.14 && dark_stroke_ratio <= 0.48 && edge_ratio >= 0.11) {
        out.label_status = "crossed_out";
        out.confidence = round_to(std::min(0.98, 0.84 + white_ratio * 0.20), 3);
        return out;
    }
    if (white_ratio >= 0.28 &&
        !(nn_pred.has_value() &&
          (*nn_pred == "covered" || *nn_pred == "crossed_out" || *nn_pred == "absent") &&
          nn_conf >= 0.65)) {
        out.label_status = "complete";
        out.confidence = round_to(std::min(0.96, std::max(0.80, 0.76 + g_std[0] / 180.0)), 3);
        return out;
    }

    out.label_status = "unknown";
    out.confidence = nn_pred.has_value() ? round_to(std::clamp(nn_conf, 0.0, 0.79), 3) : 0.45;
    return out;
}

// ============================================================================
// Clean HUD Rendering (Target Bounding Boxes + Concise Tags ONLY, NO Banner)
// ============================================================================

cv::Mat LiveCameraEngine::render_gdi_badge(const std::string& utf8_text, const cv::Scalar& bg_bgr) {
    const auto& gdi = DynamicGdi32::instance();
    const std::wstring wtext = path_from_utf8(utf8_text).wstring();

    if (gdi.ok && !wtext.empty()) {
        HDC hdc = gdi.pCreateCompatibleDC(nullptr);
        if (hdc) {
            HFONT hfont = gdi.pCreateFontW(
                -15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Microsoft YaHei"
            );
            HGDIOBJ old_font = hfont ? gdi.pSelectObject(hdc, hfont) : nullptr;
            SIZE sz{64, 18};
            gdi.pGetTextExtentPoint32W(hdc, wtext.c_str(), static_cast<int>(wtext.size()), &sz);
            const int bw = std::max(44, static_cast<int>(sz.cx) + 16);
            const int bh = std::max(24, static_cast<int>(sz.cy) + 8);

            BITMAPINFO bmi{};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = bw;
            bmi.bmiHeader.biHeight = -bh;  // top-down
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            void* bits = nullptr;
            HBITMAP hbmp = gdi.pCreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
            if (hbmp && bits) {
                HGDIOBJ old_bmp = gdi.pSelectObject(hdc, hbmp);
                auto* px = static_cast<uint8_t*>(bits);
                const uint8_t b = static_cast<uint8_t>(std::clamp(bg_bgr[0], 0.0, 255.0));
                const uint8_t g = static_cast<uint8_t>(std::clamp(bg_bgr[1], 0.0, 255.0));
                const uint8_t r = static_cast<uint8_t>(std::clamp(bg_bgr[2], 0.0, 255.0));
                for (int i = 0; i < bw * bh; ++i) {
                    px[i * 4 + 0] = b;
                    px[i * 4 + 1] = g;
                    px[i * 4 + 2] = r;
                    px[i * 4 + 3] = 255;
                }
                gdi.pSetBkMode(hdc, TRANSPARENT);
                gdi.pSetTextColor(hdc, RGB(255, 255, 255));
                gdi.pTextOutW(hdc, 8, (bh - sz.cy) / 2, wtext.c_str(), static_cast<int>(wtext.size()));
                gdi.pGdiFlush();

                cv::Mat bgra(bh, bw, CV_8UC4, bits);
                cv::Mat bgr;
                cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

                gdi.pSelectObject(hdc, old_bmp);
                gdi.pDeleteObject(hbmp);
                if (old_font) gdi.pSelectObject(hdc, old_font);
                if (hfont) gdi.pDeleteObject(hfont);
                gdi.pDeleteDC(hdc);
                return bgr;
            }
            if (hbmp) gdi.pDeleteObject(hbmp);
            if (old_font) gdi.pSelectObject(hdc, old_font);
            if (hfont) gdi.pDeleteObject(hfont);
            gdi.pDeleteDC(hdc);
        }
    }

    // Pure OpenCV fallback
    const int bw = std::max(60, static_cast<int>(utf8_text.size()) * 9 + 16);
    const int bh = 24;
    cv::Mat fallback(bh, bw, CV_8UC3, bg_bgr);
    cv::putText(fallback, utf8_text, cv::Point(6, 17), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return fallback;
}

void LiveCameraEngine::draw_unicode_badge(
    cv::Mat& img_bgr,
    int x,
    int y,
    const std::string& utf8_text,
    const cv::Scalar& bg_bgr
) {
    if (img_bgr.empty() || utf8_text.empty()) return;
    std::ostringstream key_ss;
    key_ss << utf8_text << "|" << static_cast<int>(bg_bgr[0]) << ","
           << static_cast<int>(bg_bgr[1]) << "," << static_cast<int>(bg_bgr[2]);
    const std::string key = key_ss.str();

    auto it = badge_cache_.find(key);
    if (it == badge_cache_.end()) {
        if (badge_cache_.size() >= 256) {
            badge_cache_.clear();
        }
        it = badge_cache_.emplace(key, render_gdi_badge(utf8_text, bg_bgr)).first;
    }

    const cv::Mat& badge = it->second;
    if (badge.empty() || badge.cols > img_bgr.cols || badge.rows > img_bgr.rows) return;
    const int x0 = std::clamp(x, 0, img_bgr.cols - badge.cols);
    const int y0 = std::clamp(y, 0, img_bgr.rows - badge.rows);
    badge.copyTo(img_bgr(cv::Rect(x0, y0, badge.cols, badge.rows)));
}

std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, cv::Mat> LiveCameraEngine::draw_hud(
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
) {
    cv::Mat hud = frame.clone();
    const int h = hud.rows;
    const int w = hud.cols;

    auto draw_dynamic_box = [&](
        const std::optional<Box>& box,
        const cv::Scalar& color_bgr,
        const std::string& tag,
        int thick = 2,
        bool dashed = false
    ) {
        if (!valid_optional_box(box)) return;
        const Box& b = *box;
        const int x1 = static_cast<int>(b[0] * w);
        const int y1 = static_cast<int>(b[1] * h);
        const int x2 = static_cast<int>(b[2] * w);
        const int y2 = static_cast<int>(b[3] * h);
        if (dashed) {
            cv::rectangle(hud, cv::Point(x1, y1), cv::Point(x2, y2), color_bgr, 1, cv::LINE_AA);
        } else {
            cv::rectangle(hud, cv::Point(x1, y1), cv::Point(x2, y2), color_bgr, thick, cv::LINE_AA);
            const int cl = std::max(10, std::min({24, (x2 - x1) / 5, (y2 - y1) / 5}));
            const std::array<std::array<int, 4>, 4> corners = {{
                {x1, y1, 1, 1}, {x2, y1, -1, 1}, {x1, y2, 1, -1}, {x2, y2, -1, -1}
            }};
            for (const auto& c : corners) {
                cv::line(hud, cv::Point(c[0], c[1]), cv::Point(c[0] + c[2] * cl, c[1]), color_bgr, thick + 2, cv::LINE_AA);
                cv::line(hud, cv::Point(c[0], c[1]), cv::Point(c[0], c[1] + c[3] * cl), color_bgr, thick + 2, cv::LINE_AA);
            }
            const int mx = (x1 + x2) / 2;
            const int my = (y1 + y2) / 2;
            cv::line(hud, cv::Point(mx - 8, my), cv::Point(mx + 8, my), color_bgr, 1, cv::LINE_AA);
            cv::line(hud, cv::Point(mx, my - 8), cv::Point(mx, my + 8), color_bgr, 1, cv::LINE_AA);
        }
        draw_unicode_badge(hud, x1, std::max(4, y1 - 28), tag, color_bgr);
    };

    if (show_aux) {
        draw_dynamic_box(rois.tube_roi, cv::Scalar(110, 115, 125), "辅助·料管区", 1, true);
        draw_dynamic_box(rois.left_grip_zone, cv::Scalar(110, 115, 125), "辅助·左握持", 1, true);
        draw_dynamic_box(rois.right_grip_zone, cv::Scalar(110, 115, 125), "辅助·右握持", 1, true);
    }

    if (left_glove.has_value()) {
        draw_dynamic_box(left_glove, cv::Scalar(70, 165, 45), "左手套", 2);
    }
    if (right_glove.has_value()) {
        draw_dynamic_box(right_glove, cv::Scalar(70, 165, 45), "右手套", 2);
    }

    if (bare_hand.has_value()) {
        if (work_mode == "strict") {
            const std::string tag = bare_hand_violation ? "裸手违规" : "裸手警告";
            draw_dynamic_box(bare_hand, cv::Scalar(35, 45, 210), tag, 3);
        } else {
            const std::string side = (((*bare_hand)[0] + (*bare_hand)[2]) * 0.5 < 0.50) ? "左手·调试" : "右手·调试";
            draw_dynamic_box(bare_hand, cv::Scalar(185, 130, 30), side, 2);
        }
    }

    Box active_crop_box{0.41, 0.42, 0.59, 0.58};
    if (target_locked && label_box.has_value()) {
        if (tube_box.has_value()) {
            draw_dynamic_box(tube_box, cv::Scalar(215, 125, 35), "料管目标", 2);
        }
        const std::string lbl = label_cn(label_status);
        const int pct = static_cast<int>(std::llround(label_conf.value_or(0.0) * 100.0));
        cv::Scalar label_color;
        std::string label_tag;
        if (bare_hand_violation || alarm_latched) {
            label_color = cv::Scalar(35, 45, 210);
            label_tag = "违规闭锁";
        } else if (contact_active) {
            label_color = cv::Scalar(15, 130, 230);
            label_tag = "手套接触中";
        } else if (view_valid) {
            label_color = cv::Scalar(55, 155, 35);
            label_tag = "标签·" + lbl + " " + std::to_string(pct) + "%";
        } else {
            label_color = cv::Scalar(45, 65, 205);
            label_tag = "标签·" + lbl;
        }
        draw_dynamic_box(label_box, label_color, label_tag, 2);
        active_crop_box = *label_box;
    }

    const double lcx = (active_crop_box[0] + active_crop_box[2]) * 0.5;
    const double lcy = (active_crop_box[1] + active_crop_box[3]) * 0.5;
    const double lw = std::max(0.14, (active_crop_box[2] - active_crop_box[0]) * 1.15);
    const double lh = std::max(0.11, lw * (9.0 / 16.0));
    const Box dyn_crop_box{
        std::max(0.0, std::min(1.0 - lw, lcx - lw * 0.5)),
        std::max(0.0, std::min(1.0 - lh, lcy - lh * 0.5)),
        std::min(1.0, std::max(lw, lcx + lw * 0.5)),
        std::min(1.0, std::max(lh, lcy + lh * 0.5)),
    };

    cv::Mat label_crop = crop_norm(frame, dyn_crop_box);
    cv::Mat crop_resized;
    cv::resize(label_crop, crop_resized, cv::Size(480, 270), 0, 0, cv::INTER_CUBIC);
    cv::line(crop_resized, cv::Point(240 - 14, 135), cv::Point(240 + 14, 135), cv::Scalar(45, 185, 110), 1, cv::LINE_AA);
    cv::line(crop_resized, cv::Point(240, 135 - 14), cv::Point(240, 135 + 14), cv::Scalar(45, 185, 110), 1, cv::LINE_AA);

    cv::Mat hud_stream = hud;
    if (hud.cols != kDefaultCameraWidth || hud.rows != kDefaultCameraHeight) {
        cv::resize(hud, hud_stream, cv::Size(kDefaultCameraWidth, kDefaultCameraHeight), 0, 0, cv::INTER_AREA);
    }

    std::vector<uint8_t> hud_bytes;
    std::vector<uint8_t> crop_bytes;
    cv::imencode(".jpg", hud_stream, hud_bytes, {cv::IMWRITE_JPEG_QUALITY, 82});
    cv::imencode(".jpg", crop_resized, crop_bytes, {cv::IMWRITE_JPEG_QUALITY, 88});
    return {std::move(hud_bytes), std::move(crop_bytes), std::move(label_crop)};
}

// ============================================================================
// Single-Frame Atomic Processing, Debounced Callback & Skip-Frame Auto-Capture
// ============================================================================

std::size_t LiveCameraEngine::flush_pending_transitions() {
    CycleCallback cb;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cycle_callback_;
        if (!cb) {
            pending_transitions_.clear();
            persistence_overflow_fault_ = false;
            return 0;
        }
    }

    std::size_t flushed = 0;
    while (true) {
        std::string event_id;
        json snap_copy;
        std::optional<std::string> prev_fsm;
        bool needs_ev = false;
        std::vector<uint8_t> ev_jpeg;
        cv::Mat raw_bgr;

        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (pending_transitions_.empty()) break;
            const auto& item = pending_transitions_.front();
            event_id = item.event_id;
            snap_copy = item.snapshot;
            prev_fsm = item.prev_fsm;
            needs_ev = item.needs_evidence;
            ev_jpeg = item.evidence_jpeg;
            raw_bgr = item.frame_bgr;
        }

        if (needs_ev && ev_jpeg.empty() && !raw_bgr.empty()) {
            cv::imencode(".jpg", raw_bgr, ev_jpeg, {cv::IMWRITE_JPEG_QUALITY, 82});
            if (!ev_jpeg.empty()) {
                std::lock_guard<std::mutex> lk(mutex_);
                if (!pending_transitions_.empty() && pending_transitions_.front().event_id == event_id) {
                    pending_transitions_.front().evidence_jpeg = ev_jpeg;
                    pending_transitions_.front().frame_bgr.release();
                }
            }
        }

        bool ok = false;
        try {
            ok = cb(snap_copy, prev_fsm, ev_jpeg);
        } catch (...) {
            ok = false;
        }
        if (!ok) break;

        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (!pending_transitions_.empty() && pending_transitions_.front().event_id == event_id) {
                pending_transitions_.pop_front();
            }
            if (pending_transitions_.size() < max_pending_transitions_) {
                persistence_overflow_fault_ = false;
            }
        }
        ++flushed;
    }
    return flushed;
}

bool LiveCameraEngine::process_frame_internal(const cv::Mat& input_bgr, double capture_ts) {
    if (stop_requested_.load() || input_bgr.empty()) {
        return false;
    }

    bool mirror_now = true;
    uint64_t mirror_gen = 0;
    std::string work_mode_now = "strict";
    uint64_t work_mode_gen = 0;
    uint64_t frame_cycle_gen = 0;
    bool show_aux_now = false;
    bool need_tracker_reset = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        mirror_now = mirror_horizontal_;
        mirror_gen = mirror_generation_;
        work_mode_now = work_mode_;
        work_mode_gen = work_mode_generation_;
        frame_cycle_gen = cycle_generation_;
        show_aux_now = show_auxiliary_rois_;
        const bool mirror_changed =
            (mirror_now != applied_mirror_horizontal_) || (mirror_gen != applied_mirror_generation_);
        need_tracker_reset = pending_tracker_reset_ || mirror_changed;
        pending_tracker_reset_ = false;
        if (mirror_changed) {
            contact_tracker_.reset();
            fsm_.invalidate_geometry_evidence("MIRROR_ORIENTATION_CHANGED", capture_ts);
        }
    }

    if (need_tracker_reset) {
        reset_tracking_state();
        applied_mirror_horizontal_ = mirror_now;
        applied_mirror_generation_ = mirror_gen;
    }

    cv::Mat frame;
    if (mirror_now) {
        cv::flip(input_bgr, frame, 1);
    } else {
        frame = input_bgr.clone();
    }

    int64_t current_frame_id = 0;
    double measured_fps = kDefaultTargetFps;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        current_frame_id = ++frame_id_;
        fps_times_.push_back(capture_ts);
        while (fps_times_.size() > 35) {
            fps_times_.pop_front();
        }
        if (fps_times_.size() >= 2) {
            const double dt = fps_times_.back() - fps_times_.front();
            if (dt > 1e-4) {
                measured_fps = static_cast<double>(fps_times_.size() - 1) / dt;
            }
        }
    }

    const int h = frame.rows;
    const int w = frame.cols;
    const auto [rois, thresholds, roi_ver, cfg] = active_rois_and_thresholds();
    const auto [config_match, camera_settings_locked, cal_reasons] =
        evaluate_live_calibration(cfg, w, h, work_mode_now);

    const TrackResult tr = track_dynamic_targets(frame, work_mode_now);
    const Box eval_box = (tr.target_locked && tr.label_box.has_value()) ? *tr.label_box : rois.label_roi;
    const auto quality = measure_label_quality(frame, eval_box);

    const bool strict_mode = (work_mode_now == "strict");
    const double min_lap = threshold_double(thresholds, "min_laplacian_variance", strict_mode ? 120.0 : 12.0);
    const double min_lum = threshold_double(thresholds, "min_brightness", strict_mode ? 55.0 : 25.0);
    const double max_lum = threshold_double(thresholds, "max_brightness", strict_mode ? 225.0 : 242.0);
    const double max_glare = threshold_double(thresholds, "max_overexposed_fraction", strict_mode ? 0.08 : 0.18);
    const double max_underexp = threshold_double(thresholds, "max_underexposed_fraction", strict_mode ? 0.15 : 0.35);
    const double max_label_offset = threshold_double(thresholds, "max_label_roi_offset_norm", 0.08);

    const std::optional<double> bare_cx = tr.bare_hand.has_value()
                                              ? std::optional<double>(((*tr.bare_hand)[0] + (*tr.bare_hand)[2]) * 0.5)
                                              : std::nullopt;
    const bool left_active = tr.left_glove.has_value() ||
                             (work_mode_now == "debug" && bare_cx.has_value() && *bare_cx < 0.50);
    const bool right_active = tr.right_glove.has_value() ||
                              (work_mode_now == "debug" && bare_cx.has_value() && *bare_cx >= 0.50);
    const bool glove_compliant = (left_active || right_active) && (!tr.bare_hand.has_value() || work_mode_now == "debug");
    const bool bare_hand_detected = tr.bare_hand.has_value() && (work_mode_now == "strict");

    std::vector<std::string> reasons = cal_reasons;
    if (tr.bare_hand_touching_part && work_mode_now == "strict") {
        reasons.push_back("BARE_HAND_VIOLATION");
    } else if (bare_hand_detected) {
        reasons.push_back("BARE_HAND_WARNING_NO_GLOVE");
    }
    if (!tr.target_locked) {
        reasons.push_back("TUBE_NOT_PRESENT");
    }
    if (!quality.has_value()) {
        reasons.push_back("LABEL_ROI_QUALITY_UNAVAILABLE");
    } else {
        if (quality->laplacian_variance < min_lap) reasons.push_back("LOW_SHARPNESS");
        if (quality->brightness_mean < min_lum) reasons.push_back("LOW_BRIGHTNESS");
        else if (quality->brightness_mean > max_lum) reasons.push_back("HIGH_BRIGHTNESS");
        if (quality->overexposed_fraction > max_glare) reasons.push_back("OVEREXPOSED");
        if (quality->underexposed_fraction > max_underexp) reasons.push_back("UNDEREXPOSED");
    }

    bool label_roi_drift_exceeded = false;
    if (strict_mode && config_match && tr.target_locked && tr.label_box.has_value()) {
        const double cal_cx = (rois.label_roi[0] + rois.label_roi[2]) * 0.5;
        const double cal_cy = (rois.label_roi[1] + rois.label_roi[3]) * 0.5;
        const double cur_cx = ((*tr.label_box)[0] + (*tr.label_box)[2]) * 0.5;
        const double cur_cy = ((*tr.label_box)[1] + (*tr.label_box)[3]) * 0.5;
        if (std::hypot(cur_cx - cal_cx, cur_cy - cal_cy) > max_label_offset) {
            label_roi_drift_exceeded = true;
            reasons.push_back("LABEL_ROI_DRIFT_EXCEEDED");
        }
    }

    ContactCandidate tracked{};
    bool contact_seen_now = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (stop_requested_.load() ||
            cycle_generation_ != frame_cycle_gen ||
            work_mode_generation_ != work_mode_gen ||
            work_mode_ != work_mode_now ||
            mirror_generation_ != mirror_gen) {
            return false;
        }
        tracked = contact_tracker_.update(
            ContactCandidate{
                tr.target_locked && tr.contact_active,
                tr.target_locked && tr.contact_active,
                (tr.target_locked && tr.contact_active) ? "hand_contact_candidate" : "none",
                (tr.target_locked && tr.contact_active) ? "dynamic_target_overlap" : "clear",
            },
            tr.target_locked
        );
        contact_seen_now = fsm_.contact_seen() || tracked.hand_contact_candidate;
    }

    const bool quality_ok = quality.has_value() &&
        std::none_of(reasons.begin(), reasons.end(), [](const std::string& r) {
            return r == "LOW_SHARPNESS" || r == "LOW_BRIGHTNESS" ||
                   r == "HIGH_BRIGHTNESS" || r == "OVEREXPOSED" ||
                   r == "UNDEREXPOSED";
        });
    bool view_valid = tr.target_locked && quality_ok && config_match && camera_settings_locked &&
                      !label_roi_drift_exceeded && !tracked.hand_contact_candidate && !tr.bare_hand_touching_part;
    if (tracked.hand_contact_candidate) {
        reasons.push_back("HAND_IN_CONTACT_ROI");
    }

    const cv::Mat label_crop_for_cls = crop_norm(frame, eval_box);
    const ClassificationResult cls = classify_live_label_roi(
        label_crop_for_cls,
        tr.target_locked,
        contact_seen_now,
        tracked.hand_contact_candidate
    );

    const double fsm_now = std::max(capture_ts, monotonic_seconds());
    std::string queued_event_id;
    bool queued_needs_evidence = false;
    bool hud_bare_violation = false;
    bool hud_alarm_latched = false;

    // Auto-capture & async burst snapshot variables executed outside mutex_
    bool fire_auto_capture = false;
    AutoCaptureConfig auto_cfg_snapshot{};
    AutoSampleCallback auto_cb;
    bool trigger_burst = false;
    std::string burst_cid;
    std::string burst_reason;

    pre_trigger_ring_.push_if_due(current_frame_id, capture_ts, frame);

    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (stop_requested_.load() ||
            cycle_generation_ != frame_cycle_gen ||
            work_mode_generation_ != work_mode_gen ||
            work_mode_ != work_mode_now ||
            mirror_generation_ != mirror_gen) {
            return false;
        }

        raw_frame_ = frame;
        raw_frame_label_box_ = (tr.target_locked && tr.label_box.has_value()) ? tr.label_box : std::nullopt;
        camera_online_ = true;
        width_ = w;
        height_ = h;
        fps_ = round_to(measured_fps, 1);
        inference_ms_ = cls.infer_ms;

        // Evaluate Skip-Frame Auto-Capture (A3) with optional visual-diff gate (bypassed for unknown_challenge)
        if (auto_capture_.enabled && auto_sample_callback_) {
            if (auto_capture_.captured_count >= auto_capture_.max_samples) {
                auto_capture_.enabled = false;
            } else if (current_frame_id % std::max(1, auto_capture_.every_n_frames) == 0) {
                bool should_fire = true;
                cv::Mat cur_small_gray;
                const bool use_diff_gate =
                    (auto_capture_.min_diff_mad > 0.0) &&
                    (auto_capture_.label != "unknown_challenge") &&
                    !label_crop_for_cls.empty();
                if (use_diff_gate) {
                    cv::Mat small_bgr;
                    cv::resize(label_crop_for_cls, small_bgr, cv::Size(64, 48), 0, 0, cv::INTER_AREA);
                    cv::cvtColor(small_bgr, cur_small_gray, cv::COLOR_BGR2GRAY);
                    if (!last_auto_capture_gray_small_.empty() &&
                        last_auto_capture_gray_small_.size() == cur_small_gray.size()) {
                        cv::Mat diff;
                        cv::absdiff(cur_small_gray, last_auto_capture_gray_small_, diff);
                        const double mad = cv::mean(diff)[0];
                        should_fire = (mad >= auto_capture_.min_diff_mad);
                    }
                }
                if (should_fire) {
                    ++auto_capture_.captured_count;
                    if (auto_capture_.captured_count >= auto_capture_.max_samples) {
                        auto_capture_.enabled = false;
                    }
                    if (!cur_small_gray.empty()) {
                        last_auto_capture_gray_small_ = cur_small_gray;
                    }
                    fire_auto_capture = true;
                    auto_cfg_snapshot = auto_capture_;
                    auto_cb = auto_sample_callback_;
                }
            }
        }

        json frame_meta{
            {"camera_status", "ONLINE"},
            {"online", true},
            {"camera_online", true},
            {"camera_name", camera_name_},
            {"camera_device_id", camera_device_id_},
            {"backend", backend_},
            {"camera_width", w},
            {"camera_height", h},
            {"camera_settings_locked", camera_settings_locked},
            {"fps", fps_},
            {"label_status", cls.label_status},
            {"label_confidence", cls.confidence.has_value() ? json(*cls.confidence) : json(nullptr)},
            {"view_valid", view_valid},
            {"view_valid_reasons", reasons},
            {"quality_ok", quality_ok},
            {"laplacian_variance", quality.has_value() ? json(round_to(quality->laplacian_variance, 1)) : json(nullptr)},
            {"brightness", quality.has_value() ? json(round_to(quality->brightness_mean, 1)) : json(nullptr)},
            {"overexposed_fraction", quality.has_value() ? json(round_to(quality->overexposed_fraction, 4)) : json(nullptr)},
            {"underexposed_fraction", quality.has_value() ? json(round_to(quality->underexposed_fraction, 4)) : json(nullptr)},
            {"left_grip_active", left_active},
            {"right_grip_active", right_active},
            {"glove_compliant", glove_compliant},
            {"bare_hand_detected", bare_hand_detected},
            {"pipe_color", cls.pipe_color},
            {"frame_id", current_frame_id},
            {"capture_monotonic_s", round_to(capture_ts, 3)},
            {"model_version", cls.model_version},
            {"inference_backend", cls.backend_label},
            {"inference_ms", cls.infer_ms},
            {"show_auxiliary_rois", show_aux_now},
            {"mirror_horizontal", mirror_now},
            {"work_mode", work_mode_now},
            {"hardware_3a", hardware_3a_status_},
            {"tracked_objects", tr.tracked_objects},
            {"tracked_label_box", (tr.target_locked && tr.label_box.has_value()) ? box_to_json(tr.label_box) : json(nullptr)},
            {"config_version", roi_ver},
            {"auto_capture", auto_capture_.to_json()},
            {"live_stream_url", "/api/camera/stream"},
            {"live_crop_url", "/api/camera/stream_crop"},
            {"updated_at_utc", utc_now_iso()},
        };

        if (fsm_now >= simulation_override_until_) {
            const std::string prev_fsm = state_.value("fsm_state", std::string("WAIT_PART"));
            bool changed = false;
            std::string active_fsm_state = "WAIT_PART";
            std::string active_outcome = "WAITING";
            std::optional<std::string> active_reason;
            std::optional<std::string> active_cycle_id;

            if (persistence_overflow_fault_ ||
                (cycle_callback_ && pending_transitions_.size() >= max_pending_transitions_)) {
                persistence_overflow_fault_ = true;
                active_fsm_state = "FAULT";
                active_outcome = "FAULT";
                active_reason = "PERSISTENCE_QUEUE_OVERFLOW";
                active_cycle_id = fsm_.cycle_id();
                view_valid = false;
                if (std::find(reasons.begin(), reasons.end(), "PERSISTENCE_QUEUE_OVERFLOW") == reasons.end()) {
                    reasons.push_back("PERSISTENCE_QUEUE_OVERFLOW");
                }
            } else {
                if (tr.bare_hand_touching_part && work_mode_now == "strict" && !alarm_latched_) {
                    alarm_latched_ = true;
                    latched_reason_ = "BARE_HAND_VIOLATION";
                    latched_cycle_id_ = fsm_.cycle_id().value_or(generate_live_cycle_id());
                    changed = true;
                }

                if (alarm_latched_) {
                    active_fsm_state = "ALARM";
                    active_outcome = "ALARM";
                    active_reason = latched_reason_.value_or("BARE_HAND_VIOLATION");
                    active_cycle_id = latched_cycle_id_;
                } else if (!tr.target_locked) {
                    if (fsm_.state() == FSMState::PASS) {
                        if (!pass_target_lost_since_.has_value()) {
                            pass_target_lost_since_ = fsm_now;
                        } else if ((fsm_now - *pass_target_lost_since_) >= 0.5) {
                            fsm_ = InspectionFSM(live_fsm_config());
                            ++cycle_generation_;
                            pass_target_lost_since_.reset();
                            active_cycle_tube_center_.reset();
                            active_cycle_pipe_color_.clear();
                            last_emitted_dedup_key_.clear();
                            changed = true;
                        }
                    } else if (fsm_.state() == FSMState::ALARM) {
                        alarm_latched_ = true;
                        latched_reason_ = fsm_.result_reason().value_or("LABEL_STILL_COMPLETE_AFTER_RELEASE");
                        latched_cycle_id_ = fsm_.cycle_id();
                        changed = true;
                    } else if (fsm_.state() != FSMState::WAIT_PART) {
                        fsm_ = InspectionFSM(live_fsm_config());
                        active_cycle_tube_center_.reset();
                        active_cycle_pipe_color_.clear();
                        ++cycle_generation_;
                    }

                    if (alarm_latched_) {
                        active_fsm_state = "ALARM";
                        active_outcome = "ALARM";
                        active_reason = latched_reason_;
                        active_cycle_id = latched_cycle_id_;
                    } else if (fsm_.state() == FSMState::PASS) {
                        active_fsm_state = "PASS";
                        active_outcome = "PASS";
                        active_reason = fsm_.result_reason();
                        active_cycle_id = fsm_.cycle_id();
                    } else {
                        active_fsm_state = "WAIT_PART";
                        active_outcome = "WAITING";
                        active_reason = bare_hand_detected ? "BARE_HAND_WARNING_NO_GLOVE" : "WAITING_FOR_PART";
                        active_cycle_id.reset();
                    }
                } else {
                    pass_target_lost_since_.reset();
                    if (fsm_.state() == FSMState::WAIT_PART) {
                        active_cycle_tube_center_.reset();
                        active_cycle_pipe_color_.clear();
                        FSMEvent nc{};
                        nc.kind = "new_cycle";
                        nc.cycle_id = generate_live_cycle_id();
                        nc.capture_monotonic_s = capture_ts;
                        nc.pipe_color = (cls.pipe_color != "unknown") ? cls.pipe_color : "yellow";
                        nc.operator_anon_id = "OP_STATION_01";
                        fsm_.process(nc, fsm_now);
                    }

                    std::optional<std::pair<double, double>> cur_tube_center;
                    if (tr.tube_box.has_value()) {
                        cur_tube_center = std::make_pair(
                            ((*tr.tube_box)[0] + (*tr.tube_box)[2]) * 0.5,
                            ((*tr.tube_box)[1] + (*tr.tube_box)[3]) * 0.5
                        );
                    }
                    const FSMState pre_obs_state = fsm_.state();
                    const bool in_swap_watch =
                        (pre_obs_state == FSMState::INSPECTING ||
                         pre_obs_state == FSMState::PROCESSING ||
                         pre_obs_state == FSMState::VERIFY_C_ABSENT);

                    bool swap_suspected = false;
                    if (work_mode_now == "strict" && in_swap_watch) {
                        if ((active_cycle_pipe_color_ == "yellow" || active_cycle_pipe_color_ == "blue") &&
                            (cls.pipe_color == "yellow" || cls.pipe_color == "blue") &&
                            cls.pipe_color != active_cycle_pipe_color_) {
                            swap_suspected = true;
                        }
                        if (active_cycle_tube_center_.has_value() && cur_tube_center.has_value()) {
                            const double jump = std::hypot(
                                cur_tube_center->first - active_cycle_tube_center_->first,
                                cur_tube_center->second - active_cycle_tube_center_->second
                            );
                            if (jump > 0.25) {
                                swap_suspected = true;
                            }
                        }
                    }

                    if (swap_suspected) {
                        alarm_latched_ = true;
                        latched_reason_ = "TUBE_SWAP_SUSPECTED";
                        latched_cycle_id_ = fsm_.cycle_id().value_or(generate_live_cycle_id());
                        active_fsm_state = "ALARM";
                        active_outcome = "ALARM";
                        active_reason = latched_reason_;
                        active_cycle_id = latched_cycle_id_;
                        changed = (prev_fsm != "ALARM");
                    } else {
                        FSMEvent obs_ev{};
                        obs_ev.kind = "observation";
                        obs_ev.cycle_id = fsm_.cycle_id();
                        obs_ev.capture_monotonic_s = capture_ts;
                        obs_ev.frame_id = current_frame_id;
                        obs_ev.label_class = cls.label_status;
                        obs_ev.view_valid = view_valid;
                        obs_ev.quality_ok = quality_ok;
                        obs_ev.camera_healthy = true;
                        obs_ev.camera_settings_locked = camera_settings_locked;
                        obs_ev.config_match = config_match;
                        obs_ev.tube_present = tr.target_locked;
                        obs_ev.label_window_present = tr.target_locked;
                        obs_ev.label_roi_clear = !tracked.hand_contact_candidate;
                        obs_ev.hand_occluded = tracked.hand_contact_candidate;
                        obs_ev.hand_contact_candidate = tracked.hand_contact_candidate;
                        obs_ev.hand_detection_valid = glove_compliant;
                        obs_ev.hand_in_contact_roi = tracked.hand_contact_candidate;
                        obs_ev.hand_in_label_contact_roi = tracked.hand_contact_candidate;
                        obs_ev.hand_in_left_grip_zone = tracked.hand_contact_candidate ? false : left_active;
                        obs_ev.hand_in_right_grip_zone = tracked.hand_contact_candidate ? false : right_active;
                        obs_ev.pipe_color = cls.pipe_color;
                        obs_ev.operator_anon_id = "OP_STATION_01";
                        fsm_.process(obs_ev, fsm_now);

                        if (fsm_.state() == FSMState::READY) {
                            FSMEvent si{};
                            si.kind = "start_inspection";
                            si.capture_monotonic_s = capture_ts;
                            si.frame_id = current_frame_id;
                            fsm_.process(si, fsm_now);
                        }
                        if (fsm_.state() == FSMState::ALARM) {
                            alarm_latched_ = true;
                            latched_reason_ = fsm_.result_reason();
                            latched_cycle_id_ = fsm_.cycle_id();
                        } else if (fsm_.state() == FSMState::INSPECTING ||
                                   fsm_.state() == FSMState::PROCESSING ||
                                   fsm_.state() == FSMState::VERIFY_C_ABSENT) {
                            if (cur_tube_center.has_value()) {
                                active_cycle_tube_center_ = cur_tube_center;
                            }
                            if (cls.pipe_color == "yellow" || cls.pipe_color == "blue") {
                                active_cycle_pipe_color_ = cls.pipe_color;
                            }
                        }

                        active_fsm_state = fsm_.state_str();
                        active_outcome = fsm_.outcome();
                        active_reason = fsm_.result_reason();
                        active_cycle_id = fsm_.cycle_id();
                        changed = (prev_fsm != active_fsm_state);
                    }
                }
            }

            const auto zh = explain_reason_zh(active_reason, active_outcome);
            for (auto it = frame_meta.begin(); it != frame_meta.end(); ++it) {
                state_[it.key()] = it.value();
            }
            state_["view_valid"] = view_valid;
            state_["view_valid_reasons"] = reasons;
            state_["bare_hand_violation"] = tr.bare_hand_touching_part ||
                                            (latched_reason_.value_or("") == "BARE_HAND_VIOLATION");
            state_["alarm_latched"] = alarm_latched_;
            state_["label_contact_active"] = tracked.hand_contact_candidate;
            state_["hand_contact_candidate"] = tracked.hand_contact_candidate;
            state_["fsm_state"] = active_fsm_state;
            state_["outcome"] = active_outcome;
            state_["result_reason"] = active_reason.has_value() ? json(*active_reason) : json(nullptr);
            state_["operator_message"] = zh.operator_message;
            state_["engineer_message"] = zh.engineer_message;
            state_["c_timer_ms"] = fsm_.c_timer_ms();
            state_["cycle_id"] = active_cycle_id.has_value() ? json(*active_cycle_id) : json(nullptr);
            raw_frame_state_ = state_;

            // Deduplicate (cycle_id, outcome, event_type) and apply 2000ms per-cycle evidence cooldown
            if (changed && cycle_callback_ && !persistence_overflow_fault_) {
                const std::string cid_key = active_cycle_id.value_or("NO_CYCLE");
                const std::string ev_type_key = active_reason.value_or(active_fsm_state);
                const std::string dedup_key = cid_key + "|" + active_outcome + "|" + ev_type_key;

                if (dedup_key != last_emitted_dedup_key_) {
                    if (pending_transitions_.size() >= max_pending_transitions_) {
                        persistence_overflow_fault_ = true;
                        state_["fsm_state"] = "FAULT";
                        state_["outcome"] = "FAULT";
                        state_["result_reason"] = "PERSISTENCE_QUEUE_OVERFLOW";
                        state_["view_valid"] = false;
                        raw_frame_state_ = state_;
                    } else {
                        last_emitted_dedup_key_ = dedup_key;
                        const std::string event_id = random_token_hex(16);
                        json queued_snap = state_;
                        queued_snap["transition_event_id"] = event_id;

                        bool attach_evidence = false;
                        if (active_outcome == "ALARM") {
                            const auto it_ev = last_evidence_emit_by_cycle_.find(cid_key);
                            if (it_ev == last_evidence_emit_by_cycle_.end() ||
                                (fsm_now - it_ev->second) >= 2.0) {
                                attach_evidence = true;
                                last_evidence_emit_by_cycle_[cid_key] = fsm_now;
                                if (last_evidence_emit_by_cycle_.size() > 512) {
                                    last_evidence_emit_by_cycle_.clear();
                                    last_evidence_emit_by_cycle_[cid_key] = fsm_now;
                                }
                            }
                        }

                        PendingTransition pt{};
                        pt.event_id = event_id;
                        pt.snapshot = std::move(queued_snap);
                        pt.prev_fsm = prev_fsm;
                        pt.needs_evidence = attach_evidence;
                        if (attach_evidence) {
                            pt.frame_bgr = frame.clone();
                        }
                        pending_transitions_.push_back(std::move(pt));
                        queued_event_id = event_id;
                        queued_needs_evidence = attach_evidence;
                    }
                }
            }

            if (active_outcome == "ALARM" && changed) {
                trigger_burst = true;
                burst_cid = active_cycle_id.value_or("LIVE_ALARM");
                burst_reason = active_reason.value_or("ALARM");
            }
        } else {
            raw_frame_state_ = state_;
            for (auto it = frame_meta.begin(); it != frame_meta.end(); ++it) {
                raw_frame_state_[it.key()] = it.value();
            }
        }

        hud_bare_violation = tr.bare_hand_touching_part ||
                             (latched_reason_.value_or("") == "BARE_HAND_VIOLATION");
        hud_alarm_latched = alarm_latched_;
    }

    if (trigger_burst) {
        async_evidence_writer_.enqueue_burst(
            burst_cid,
            burst_reason,
            pre_trigger_ring_.snapshot_ordered(),
            fsm_now
        );
    }

    // Invoke skip-frame auto-sample callback outside mutex_ so dataset I/O never stalls streams
    if (fire_auto_capture && auto_cb) {
        try {
            const std::string sample_pipe =
                (cls.pipe_color == "blue" || cls.pipe_color == "yellow")
                    ? cls.pipe_color
                    : auto_cfg_snapshot.pipe_color;
            auto_cb(
                frame,
                label_crop_for_cls,
                auto_cfg_snapshot.label,
                sample_pipe,
                auto_cfg_snapshot.physical_tube_id,
                "AUTO_SKIP_CAPTURE",
                eval_box
            );
        } catch (...) {
        }
    }

    try {
        auto [hud_bytes, crop_bytes, raw_crop] = draw_hud(
            frame,
            rois,
            tr.tube_box,
            tr.label_box,
            tr.left_glove,
            tr.right_glove,
            tr.bare_hand,
            tracked.hand_contact_candidate,
            hud_bare_violation,
            hud_alarm_latched,
            tr.target_locked,
            view_valid,
            cls.label_status,
            cls.confidence,
            work_mode_now,
            show_aux_now
        );

        {
            std::lock_guard<std::mutex> lk(mutex_);
            hud_jpeg_ = std::move(hud_bytes);
            crop_jpeg_ = std::move(crop_bytes);
            last_tracked_crop_ = std::move(raw_crop);
            ++hud_seq_;
            ++crop_seq_;
            if (!queued_event_id.empty() && queued_needs_evidence && !hud_jpeg_.empty()) {
                for (auto& item : pending_transitions_) {
                    if (item.event_id == queued_event_id) {
                        item.evidence_jpeg = hud_jpeg_;
                        item.frame_bgr.release();
                        break;
                    }
                }
            }
        }
        frame_cv_.notify_all();
    } catch (...) {
        std::vector<uint8_t> fb_bytes;
        cv::imencode(".jpg", frame, fb_bytes, {cv::IMWRITE_JPEG_QUALITY, 80});
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (!fb_bytes.empty()) {
                hud_jpeg_ = std::move(fb_bytes);
                ++hud_seq_;
                if (!queued_event_id.empty() && queued_needs_evidence) {
                    for (auto& item : pending_transitions_) {
                        if (item.event_id == queued_event_id) {
                            item.evidence_jpeg = hud_jpeg_;
                            item.frame_bgr.release();
                            break;
                        }
                    }
                }
            }
            crop_jpeg_.clear();
        }
        frame_cv_.notify_all();
    }

    flush_pending_transitions();
    return true;
}

}  // namespace igbt
