#include <igbt/vision/tube_geometry.hpp>

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace igbt::vision {

namespace {

constexpr double kPi = 3.14159265358979323846;

double round3(double v) {
    return std::round(v * 1000.0) / 1000.0;
}

struct Segment2D {
    double x1{0.0}, y1{0.0}, x2{0.0}, y2{0.0};
    double cx{0.0}, cy{0.0}, length{0.0}, angle_deg{0.0};
    double ux{1.0}, uy{0.0};
    double mean_dy{0.0};
};

struct Line2D {
    double vx{1.0}, vy{0.0}, x0{0.0}, y0{0.0}, angle_deg{0.0};
};

double point_line_dist(double px, double py, const Line2D& line) {
    return std::abs((px - line.x0) * line.vy - (py - line.y0) * line.vx);
}

std::optional<Line2D> make_line_from_segment(const Segment2D& s) {
    if (s.length < 1e-3) return std::nullopt;
    Line2D l{s.ux, s.uy, s.cx, s.cy, 0.0};
    if (l.vx < 0.0) {
        l.vx = -l.vx;
        l.vy = -l.vy;
    }
    l.angle_deg = std::atan2(l.vy, l.vx) * 180.0 / kPi;
    return l;
}

std::optional<Line2D> fit_line_from_segments(const std::vector<const Segment2D*>& segs) {
    if (segs.empty()) return std::nullopt;
    std::vector<cv::Point2f> pts;
    pts.reserve(segs.size() * 5);
    for (const Segment2D* s : segs) {
        for (int k = 0; k < 5; ++k) {
            const float t = static_cast<float>(k) / 4.0f;
            pts.emplace_back(static_cast<float>(s->x1 + t * (s->x2 - s->x1)),
                             static_cast<float>(s->y1 + t * (s->y2 - s->y1)));
        }
    }
    if (pts.size() < 2) return std::nullopt;
    cv::Vec4f fitted;
    cv::fitLine(pts, fitted, cv::DIST_HUBER, 0.0, 0.01, 0.01);
    double vx = static_cast<double>(fitted[0]);
    double vy = static_cast<double>(fitted[1]);
    const double nrm = std::hypot(vx, vy);
    if (nrm < 1e-6) return std::nullopt;
    vx /= nrm;
    vy /= nrm;
    if (vx < 0.0) {
        vx = -vx;
        vy = -vy;
    }
    return Line2D{vx, vy, static_cast<double>(fitted[2]), static_cast<double>(fitted[3]),
                  std::atan2(vy, vx) * 180.0 / kPi};
}

std::optional<cv::Point2d> intersect_normal_ray_with_line(
    const cv::Point2d& axis_pt,
    const cv::Point2d& normal_vec,
    const Line2D& wall
) {
    const double denom = normal_vec.x * wall.vy - normal_vec.y * wall.vx;
    if (std::abs(denom) < 1e-6) return std::nullopt;
    const double cross = (axis_pt.x - wall.x0) * wall.vy - (axis_pt.y - wall.y0) * wall.vx;
    const double t = -cross / denom;
    return cv::Point2d(axis_pt.x + t * normal_vec.x, axis_pt.y + t * normal_vec.y);
}

bool all_finite(const cv::Mat& m) {
    for (int r = 0; r < m.rows; ++r) {
        const double* p = m.ptr<double>(r);
        for (int c = 0; c < m.cols; ++c) {
            if (!std::isfinite(p[c])) return false;
        }
    }
    return true;
}

bool extract_midband_segments(
    const cv::Mat& small_bgr,
    const cv::Rect& mid_roi,
    const std::vector<Box>& hand_boxes_norm,
    double max_angle_deg,
    std::vector<Segment2D>& out_segs
) {
    cv::Mat roi_bgr = small_bgr(mid_roi);
    cv::Mat roi_hsv, roi_gray;
    cv::cvtColor(roi_bgr, roi_hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(roi_bgr, roi_gray, cv::COLOR_BGR2GRAY);

    cv::Mat forbidden_mask = cv::Mat::zeros(mid_roi.size(), CV_8U);
    for (int y = 0; y < mid_roi.height; ++y) {
        const auto* hp = roi_hsv.ptr<cv::Vec3b>(y);
        uint8_t* fp = forbidden_mask.ptr<uint8_t>(y);
        for (int x = 0; x < mid_roi.width; ++x) {
            if (hp[x][2] >= 250 && hp[x][1] <= 15) fp[x] = 255;
        }
    }
    if (cv::countNonZero(forbidden_mask) > 0) {
        cv::dilate(forbidden_mask, forbidden_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    }
    for (const auto& hb : hand_boxes_norm) {
        const int hx1 = std::max(0, static_cast<int>(std::floor(hb[0] * small_bgr.cols)) - 6 - mid_roi.x);
        const int hy1 = std::max(0, static_cast<int>(std::floor(hb[1] * small_bgr.rows)) - 6 - mid_roi.y);
        const int hx2 = std::min(mid_roi.width, static_cast<int>(std::ceil(hb[2] * small_bgr.cols)) + 6 - mid_roi.x);
        const int hy2 = std::min(mid_roi.height, static_cast<int>(std::ceil(hb[3] * small_bgr.rows)) + 6 - mid_roi.y);
        if (hx2 > hx1 && hy2 > hy1) {
            forbidden_mask(cv::Rect(hx1, hy1, hx2 - hx1, hy2 - hy1)).setTo(255);
        }
    }

    int valid_cols = 0;
    for (int x = 0; x < mid_roi.width; ++x) {
        if (cv::countNonZero(forbidden_mask.col(x)) < static_cast<int>(mid_roi.height * 0.75)) {
            ++valid_cols;
        }
    }
    if (static_cast<double>(valid_cols) / std::max(1, mid_roi.width) < 0.20) return false;

    // Smooth 1px background lines, then blend raw & CLAHE directional Sobel BEFORE zeroing forbidden_mask
    cv::Mat smooth_gray, clahe_gray, grad_raw16, grad_clahe16, abs_raw, abs_clahe, grad_y;
    cv::GaussianBlur(roi_gray, smooth_gray, cv::Size(5, 3), 0.0);
    cv::Sobel(smooth_gray, grad_raw16, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(grad_raw16, abs_raw);

    cv::createCLAHE(1.6, cv::Size(8, 8))->apply(smooth_gray, clahe_gray);
    cv::Sobel(clahe_gray, grad_clahe16, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(grad_clahe16, abs_clahe);

    cv::addWeighted(abs_raw, 0.65, abs_clahe, 0.35, 0.0, grad_y);

    cv::Mat grad_suppress;
    cv::dilate(forbidden_mask, grad_suppress, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    grad_y.setTo(0, grad_suppress);

    double max_grad = 0.0;
    cv::minMaxLoc(grad_y, nullptr, &max_grad);
    if (max_grad < 32.0) return false;

    cv::Mat edge_bin;
    const double otsu_t = cv::threshold(grad_y, edge_bin, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(grad_y, edge_bin, std::max({48.0, otsu_t, max_grad * 0.38}), 255.0, cv::THRESH_BINARY);

    std::vector<cv::Vec4i> raw_lines;
    const double min_seg_len = std::clamp(mid_roi.width * 0.12, 18.0, 32.0);
    cv::HoughLinesP(edge_bin, raw_lines, 1.0, kPi / 360.0, 16, min_seg_len, 14.0);

    out_segs.clear();
    out_segs.reserve(raw_lines.size());
    for (const auto& l : raw_lines) {
        double x1 = l[0] + mid_roi.x, y1 = l[1] + mid_roi.y;
        double x2 = l[2] + mid_roi.x, y2 = l[3] + mid_roi.y;
        if (x2 < x1) {
            std::swap(x1, x2);
            std::swap(y1, y2);
        }
        const double dx = x2 - x1, dy = y2 - y1, len = std::hypot(dx, dy);
        if (len < min_seg_len) continue;
        const double ang = std::atan2(dy, dx) * 180.0 / kPi;
        if (std::abs(ang) > max_angle_deg) continue;

        double dy_acc = 0.0;
        for (double t : {0.2, 0.5, 0.8}) {
            const int sx = std::clamp(static_cast<int>(std::llround(l[0] + t * (l[2] - l[0]))), 0, grad_raw16.cols - 1);
            const int sy = std::clamp(static_cast<int>(std::llround(l[1] + t * (l[3] - l[1]))), 0, grad_raw16.rows - 1);
            int16_t best_v = grad_raw16.at<int16_t>(sy, sx);
            for (int oy = -1; oy <= 1; ++oy) {
                const int yy = std::clamp(sy + oy, 0, grad_raw16.rows - 1);
                const int16_t cand_v = grad_raw16.at<int16_t>(yy, sx);
                if (std::abs(cand_v) > std::abs(best_v)) best_v = cand_v;
            }
            dy_acc += static_cast<double>(best_v);
        }
        out_segs.push_back(Segment2D{
            x1, y1, x2, y2, (x1 + x2) * 0.5, (y1 + y2) * 0.5, len, ang, dx / len, dy / len, dy_acc / 3.0
        });
    }
    return !out_segs.empty();
}

bool ransac_fit_parallel_walls(
    const std::vector<Segment2D>& segs,
    double mid_cx,
    double center_y,
    double expected_w_px,
    double tb_h,
    int work_h,
    bool strict_prior,
    const TubeAlignConfig& cfg,
    Line2D& out_top,
    Line2D& out_bot,
    double& out_inlier_ratio
) {
    std::vector<double> angles;
    angles.reserve(segs.size());
    for (const auto& s : segs) angles.push_back(s.angle_deg);
    std::sort(angles.begin(), angles.end());
    const double med_rad = (angles.empty() ? 0.0 : angles[angles.size() / 2]) * kPi / 180.0;
    const double nx_med = -std::sin(med_rad), ny_med = std::cos(med_rad);

    std::vector<const Segment2D*> top_cands, bot_cands;
    for (const auto& s : segs) {
        const double d_perp = (s.cx - mid_cx) * nx_med + (s.cy - center_y) * ny_med;
        (d_perp < 0.0 ? top_cands : bot_cands).push_back(&s);
    }
    if (top_cands.empty() || bot_cands.empty()) return false;

    const double min_pair_w = work_h * 0.04;
    const double max_pair_w = std::min(
        work_h * 0.40,
        std::max({tb_h * 1.25, expected_w_px * 2.0, work_h * 0.25})
    );

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    const std::size_t combos = top_cands.size() * bot_cands.size();
    if (combos <= 240) {
        pairs.reserve(combos);
        for (std::size_t i = 0; i < top_cands.size(); ++i)
            for (std::size_t j = 0; j < bot_cands.size(); ++j) pairs.emplace_back(i, j);
    } else {
        std::mt19937 rng(20261126u);
        std::uniform_int_distribution<std::size_t> dt(0, top_cands.size() - 1), db(0, bot_cands.size() - 1);
        pairs.reserve(240);
        for (int k = 0; k < 240; ++k) pairs.emplace_back(dt(rng), db(rng));
    }

    double total_seg_len = 0.0;
    for (const auto& s : segs) total_seg_len += s.length;
    total_seg_len = std::max(1.0, total_seg_len);

    double best_score = -1e9;
    std::optional<Line2D> best_top_init, best_bot_init;
    for (const auto& [ti, bi] : pairs) {
        const Segment2D* ts = top_cands[ti];
        const Segment2D* bs = bot_cands[bi];
        if (std::abs(ts->angle_deg - bs->angle_deg) > cfg.max_parallel_diff_deg * 1.6) continue;
        auto lt = make_line_from_segment(*ts), lb = make_line_from_segment(*bs);
        if (!lt || !lb) continue;

        const double avg_ang = 0.5 * (lt->angle_deg + lb->angle_deg) * kPi / 180.0;
        const cv::Point2d nvec(-std::sin(avg_ang), std::cos(avg_ang));
        const cv::Point2d mid_c(0.5 * (ts->cx + bs->cx), 0.5 * (ts->cy + bs->cy));
        auto pt_t = intersect_normal_ray_with_line(mid_c, nvec, *lt);
        auto pt_b = intersect_normal_ray_with_line(mid_c, nvec, *lb);
        if (!pt_t || !pt_b) continue;
        const double sep = std::hypot(pt_b->x - pt_t->x, pt_b->y - pt_t->y);
        if (sep < min_pair_w || sep > max_pair_w) continue;

        double inlier_len = 0.0;
        for (const Segment2D* p : top_cands) {
            if (point_line_dist(p->x1, p->y1, *lt) <= 3.5 && point_line_dist(p->x2, p->y2, *lt) <= 3.5)
                inlier_len += p->length;
        }
        for (const Segment2D* p : bot_cands) {
            if (point_line_dist(p->x1, p->y1, *lb) <= 3.5 && point_line_dist(p->x2, p->y2, *lb) <= 3.5)
                inlier_len += p->length;
        }
        const double width_term = strict_prior
                                      ? ((std::abs(sep - expected_w_px) / std::max(1.0, expected_w_px) <= cfg.max_width_rel_dev * 1.5)
                                             ? 6.0
                                             : 0.0)
                                      : ((sep <= tb_h * 1.05) ? std::min(8.0, (sep / std::max(1.0, tb_h)) * 8.0) : 0.0);
        const double polarity_bonus = (ts->mean_dy * bs->mean_dy < 0.0) ? 14.0 : 0.0;
        const double score = (inlier_len / total_seg_len) * 100.0 + width_term + polarity_bonus -
                             std::abs(lt->angle_deg - lb->angle_deg) * 2.5;
        if (score > best_score) {
            best_score = score;
            best_top_init = lt;
            best_bot_init = lb;
        }
    }
    if (!best_top_init || !best_bot_init) return false;

    std::vector<const Segment2D*> top_inliers, bot_inliers;
    double inlier_len_sum = 0.0;
    for (const Segment2D* p : top_cands) {
        if (point_line_dist(p->x1, p->y1, *best_top_init) <= 3.5 &&
            point_line_dist(p->x2, p->y2, *best_top_init) <= 3.5) {
            top_inliers.push_back(p);
            inlier_len_sum += p->length;
        }
    }
    for (const Segment2D* p : bot_cands) {
        if (point_line_dist(p->x1, p->y1, *best_bot_init) <= 3.5 &&
            point_line_dist(p->x2, p->y2, *best_bot_init) <= 3.5) {
            bot_inliers.push_back(p);
            inlier_len_sum += p->length;
        }
    }

    auto l_top = fit_line_from_segments(top_inliers);
    auto l_bot = fit_line_from_segments(bot_inliers);
    if (!l_top || !l_bot) return false;
    out_top = *l_top;
    out_bot = *l_bot;
    out_inlier_ratio = std::clamp(inlier_len_sum / total_seg_len, 0.0, 1.0);
    return true;
}

}  // namespace

std::string tube_align_status_to_string(TubeAlignStatus status) {
    switch (status) {
        case TubeAlignStatus::NO_TUBE: return "NO_TUBE";
        case TubeAlignStatus::ALIGNED: return "ALIGNED";
        case TubeAlignStatus::REUSED_LOW_CONF: return "REUSED_LOW_CONF";
        case TubeAlignStatus::TUBE_ALIGN_FAIL: return "TUBE_ALIGN_FAIL";
    }
    return "NO_TUBE";
}

json TubeAlignResult::to_json() const {
    json quad = json::array();
    if (has_quad) {
        for (const auto& pt : quad_norm) quad.push_back({round3(pt.x), round3(pt.y)});
    }
    return json{
        {"status", tube_align_status_to_string(status)},
        {"geometry_ok", geometry_ok},
        {"reused_prev_matrix", reused_prev_matrix},
        {"hard_failed", hard_failed},
        {"consecutive_failures", consecutive_failures},
        {"angle_deg", round3(angle_deg)},
        {"parallel_diff_deg", round3(parallel_diff_deg)},
        {"condition_number", round3(condition_number)},
        {"det_2x2", round3(det_2x2)},
        {"measured_width_px", round3(measured_width_px)},
        {"expected_width_px", round3(expected_width_px)},
        {"width_rel_error", round3(width_rel_error)},
        {"inlier_ratio", round3(inlier_ratio)},
        {"used_perspective", used_perspective},
        {"fail_reason", fail_reason.empty() ? json(nullptr) : json(fail_reason)},
        {"quad_norm", std::move(quad)},
        {"align_ms", round3(align_ms)},
    };
}

TubeGeometryAligner::TubeGeometryAligner(const TubeAlignConfig& config)
    : config_(config) {}

void TubeGeometryAligner::reset() {
    has_prev_valid_ = false;
    consecutive_failures_ = 0;
    ema_cx_ = ema_cy_ = ema_angle_deg_ = ema_width_px_ = ema_length_px_ = 0.0;
    prev_homography_.release();
    prev_quad_norm_.fill(cv::Point2f(0.0f, 0.0f));
    last_status_ = json{{"status", "NO_TUBE"}, {"geometry_ok", false}, {"reused_prev_matrix", false},
                        {"hard_failed", false}, {"consecutive_failures", 0}};
}

void TubeGeometryAligner::update_config(const TubeAlignConfig& config) { config_ = config; }
json TubeGeometryAligner::last_status_json() const { return last_status_; }

bool TubeGeometryAligner::check_homography_gate(
    const cv::Mat& H_norm,
    double max_cond,
    double min_det,
    double max_det,
    double& out_cond,
    double& out_det
) {
    out_cond = 999.0;
    out_det = 0.0;
    if (H_norm.empty() || H_norm.rows != 3 || H_norm.cols != 3) return false;
    cv::Mat h64;
    if (H_norm.type() == CV_64F) h64 = H_norm;
    else H_norm.convertTo(h64, CV_64F);
    if (!all_finite(h64)) return false;

    const double h22 = h64.at<double>(2, 2);
    if (std::abs(h22) < 1e-9) return false;
    cv::Mat sub = h64(cv::Rect(0, 0, 2, 2)).clone() / h22;
    out_det = sub.at<double>(0, 0) * sub.at<double>(1, 1) - sub.at<double>(0, 1) * sub.at<double>(1, 0);
    if (!std::isfinite(out_det) || out_det < min_det || out_det > max_det) return false;

    cv::Mat w;
    cv::SVD::compute(sub, w, cv::SVD::NO_UV);
    if (w.rows < 2) return false;
    const double s_max = w.at<double>(0, 0), s_min = w.at<double>(1, 0);
    if (!std::isfinite(s_max) || !std::isfinite(s_min) || s_min < 1e-9) return false;
    out_cond = s_max / s_min;
    return std::isfinite(out_cond) && out_cond <= max_cond;
}

void TubeGeometryAligner::warp_outputs(
    const cv::Mat& frame_bgr,
    const cv::Mat& H_full,
    const std::array<cv::Point2f, 4>& quad_norm,
    const std::optional<Box>& label_box_norm,
    cv::Mat& out_tube_bgr,
    cv::Mat& out_label_bgr
) const {
    if (frame_bgr.empty() || H_full.empty()) return;
    cv::warpPerspective(frame_bgr, out_tube_bgr, H_full, cv::Size(config_.out_width, config_.out_height),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    double u_center = 0.5, u_span = 0.30;
    const double quad_len_px = std::max<double>(10.0, std::hypot(
        static_cast<double>(quad_norm[1].x - quad_norm[0].x) * frame_bgr.cols,
        static_cast<double>(quad_norm[1].y - quad_norm[0].y) * frame_bgr.rows));
    if (label_box_norm.has_value()) {
        const Box& lb = *label_box_norm;
        const std::vector<cv::Point2f> in_pts = {cv::Point2f(
            static_cast<float>((lb[0] + lb[2]) * 0.5 * frame_bgr.cols),
            static_cast<float>((lb[1] + lb[3]) * 0.5 * frame_bgr.rows))};
        std::vector<cv::Point2f> out_pts;
        cv::perspectiveTransform(in_pts, out_pts, H_full);
        if (!out_pts.empty() && std::isfinite(out_pts[0].x)) {
            u_center = std::clamp(static_cast<double>(out_pts[0].x) / config_.out_width, 0.20, 0.80);
        }
        const double lbl_w_px = std::max(1.0, (lb[2] - lb[0]) * frame_bgr.cols);
        u_span = std::clamp((lbl_w_px / quad_len_px) * 1.25, 0.22, 0.44);
    }

    const double xr1 = std::max(0.0, u_center - u_span * 0.5) * config_.out_width;
    const double xr2 = std::max(xr1 + 8.0, std::min(1.0, u_center + u_span * 0.5) * config_.out_width);
    const double yr1 = -0.08 * config_.out_height, yr2 = 1.08 * config_.out_height;
    const double sx = static_cast<double>(config_.label_out_width) / (xr2 - xr1);
    const double sy = static_cast<double>(config_.label_out_height) / (yr2 - yr1);
    cv::Mat S_sub = (cv::Mat_<double>(3, 3) << sx, 0.0, -sx * xr1, 0.0, sy, -sy * yr1, 0.0, 0.0, 1.0);
    cv::warpPerspective(frame_bgr, out_label_bgr, S_sub * H_full,
                        cv::Size(config_.label_out_width, config_.label_out_height),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

TubeAlignResult TubeGeometryAligner::handle_failure(
    const cv::Mat& frame_bgr,
    const std::optional<Box>& label_box_norm,
    const std::string& reason,
    double t0_s,
    bool force_hard_if_no_history,
    const TubeAlignResult* diagnostics
) {
    ++consecutive_failures_;
    TubeAlignResult res = diagnostics ? *diagnostics : TubeAlignResult{};
    res.geometry_ok = false;
    res.consecutive_failures = consecutive_failures_;
    res.fail_reason = reason;

    const bool hard_now = (consecutive_failures_ >= config_.hard_fail_frames) ||
                          (!has_prev_valid_ && force_hard_if_no_history);
    if (!hard_now && consecutive_failures_ <= config_.max_reuse_frames) {
        res.status = TubeAlignStatus::REUSED_LOW_CONF;
        res.hard_failed = false;
        if (has_prev_valid_ && !prev_homography_.empty()) {
            res.reused_prev_matrix = true;
            res.homography_3x3 = prev_homography_.clone();
            res.quad_norm = prev_quad_norm_;
            res.has_quad = true;
            res.angle_deg = ema_angle_deg_;
            warp_outputs(frame_bgr, res.homography_3x3, res.quad_norm, label_box_norm,
                         res.rectified_tube_bgr, res.rectified_label_bgr);
        }
    } else {
        res.status = TubeAlignStatus::TUBE_ALIGN_FAIL;
        res.hard_failed = true;
        res.reused_prev_matrix = false;
        res.homography_3x3.release();
        res.rectified_tube_bgr.release();
        res.rectified_label_bgr.release();
        res.has_quad = false;
    }

    res.align_ms = (monotonic_seconds() - t0_s) * 1000.0;
    last_status_ = res.to_json();
    return res;
}

TubeAlignResult TubeGeometryAligner::align(
    const cv::Mat& frame_bgr,
    const std::optional<Box>& tube_box_norm,
    const std::vector<Box>& hand_boxes_norm,
    const std::optional<Box>& label_box_norm,
    const std::optional<double>& calibrated_tube_height_norm
) {
    const double t0 = monotonic_seconds();
    if (frame_bgr.empty() || !tube_box_norm.has_value()) {
        reset();
        TubeAlignResult empty_res{};
        empty_res.status = TubeAlignStatus::NO_TUBE;
        empty_res.align_ms = (monotonic_seconds() - t0) * 1000.0;
        last_status_ = empty_res.to_json();
        return empty_res;
    }

    const int ww = config_.work_width, wh = config_.work_height;
    cv::Mat small_bgr;
    cv::resize(frame_bgr, small_bgr, cv::Size(ww, wh), 0, 0, cv::INTER_AREA);

    const Box& tb = *tube_box_norm;
    const double tb_x1 = std::clamp(tb[0], 0.0, 0.98) * ww, tb_y1 = std::clamp(tb[1], 0.0, 0.98) * wh;
    const double tb_x2 = std::clamp(tb[2], tb[0] + 0.02, 1.0) * ww, tb_y2 = std::clamp(tb[3], tb[1] + 0.02, 1.0) * wh;
    const double tb_w = std::max(24.0, tb_x2 - tb_x1), tb_h = std::max(12.0, tb_y2 - tb_y1);

    double expected_w_px = tb_h;
    bool strict_prior = false;
    if (calibrated_tube_height_norm.has_value() && *calibrated_tube_height_norm >= 0.04 && *calibrated_tube_height_norm <= 0.35) {
        expected_w_px = (*calibrated_tube_height_norm) * wh;
        strict_prior = true;
    } else if (has_prev_valid_ && ema_width_px_ > 8.0) {
        expected_w_px = ema_width_px_;
        strict_prior = true;
    }

    const double tb_cy = 0.5 * (tb_y1 + tb_y2);
    const double center_y = (has_prev_valid_ && std::abs(ema_cy_ - tb_cy) >= 6.0 &&
                             std::abs(ema_cy_ - tb_cy) < expected_w_px * 1.4)
                                ? ema_cy_
                                : tb_cy;
    const double search_half_h = std::min(wh * 0.30, std::max(tb_h * 0.85, 24.0));
    const int mid_x1 = std::clamp(static_cast<int>(std::floor(tb_x1 + tb_w * config_.midband_start_frac)), 0, ww - 16);
    const int mid_x2 = std::clamp(static_cast<int>(std::ceil(tb_x1 + tb_w * config_.midband_end_frac)), mid_x1 + 16, ww);
    const int roi_y1 = std::clamp(static_cast<int>(std::floor(center_y - search_half_h)), 0, wh - 16);
    const int roi_y2 = std::clamp(static_cast<int>(std::ceil(center_y + search_half_h)), roi_y1 + 16, wh);
    const cv::Rect mid_roi(mid_x1, roi_y1, mid_x2 - mid_x1, roi_y2 - roi_y1);
    if (mid_roi.width < 20 || mid_roi.height < 16) {
        return handle_failure(frame_bgr, label_box_norm, "INSUFFICIENT_MIDBAND_EDGES", t0, false);
    }

    std::vector<Segment2D> segs;
    if (!extract_midband_segments(small_bgr, mid_roi, hand_boxes_norm, config_.max_angle_deg, segs)) {
        return handle_failure(frame_bgr, label_box_norm, "INSUFFICIENT_MIDBAND_EDGES", t0, false);
    }

    Line2D l_top{}, l_bot{};
    double inlier_ratio = 0.0;
    if (!ransac_fit_parallel_walls(segs, mid_roi.x + mid_roi.width * 0.5, center_y,
                                   expected_w_px, tb_h, wh, strict_prior, config_,
                                   l_top, l_bot, inlier_ratio)) {
        return handle_failure(frame_bgr, label_box_norm, "PARALLEL_PAIR_NOT_FOUND", t0, false);
    }

    const double parallel_diff_deg = std::abs(l_top.angle_deg - l_bot.angle_deg);
    const double raw_angle_deg = 0.5 * (l_top.angle_deg + l_bot.angle_deg);
    const double raw_ang_rad = raw_angle_deg * kPi / 180.0;
    const cv::Point2d u_axis(std::cos(raw_ang_rad), std::sin(raw_ang_rad));
    const cv::Point2d n_axis(-u_axis.y, u_axis.x);
    const cv::Point2d probe_c(0.5 * (tb_x1 + tb_x2), center_y);
    auto top_mid = intersect_normal_ray_with_line(probe_c, n_axis, l_top);
    auto bot_mid = intersect_normal_ray_with_line(probe_c, n_axis, l_bot);
    if (!top_mid || !bot_mid) {
        return handle_failure(frame_bgr, label_box_norm, "PARALLEL_PAIR_NOT_FOUND", t0, false);
    }

    const double measured_w_px = std::hypot(bot_mid->x - top_mid->x, bot_mid->y - top_mid->y);
    const double measured_cx = 0.5 * (top_mid->x + bot_mid->x);
    const double measured_cy = 0.5 * (top_mid->y + bot_mid->y);
    const double width_rel_err = std::abs(measured_w_px - expected_w_px) / std::max(1.0, expected_w_px);

    TubeAlignResult diag{};
    diag.angle_deg = raw_angle_deg;
    diag.parallel_diff_deg = parallel_diff_deg;
    diag.measured_width_px = measured_w_px;
    diag.expected_width_px = expected_w_px;
    diag.width_rel_error = width_rel_err;
    diag.inlier_ratio = inlier_ratio;

    if (parallel_diff_deg > config_.max_parallel_diff_deg) {
        return handle_failure(frame_bgr, label_box_norm, "PARALLEL_DIFF_EXCEEDED", t0, false, &diag);
    }
    if (strict_prior && width_rel_err > config_.max_width_rel_dev) {
        return handle_failure(frame_bgr, label_box_norm, "WIDTH_DEV_EXCEEDED", t0, false, &diag);
    }
    if (has_prev_valid_ && std::abs(raw_angle_deg - ema_angle_deg_) > config_.max_angle_jump_deg) {
        return handle_failure(frame_bgr, label_box_norm, "ANGLE_JUMP_EXCEEDED", t0, false, &diag);
    }

    const double alpha = has_prev_valid_ ? config_.ema_alpha : 1.0;
    const double sm_cx = has_prev_valid_ ? (alpha * measured_cx + (1.0 - alpha) * ema_cx_) : measured_cx;
    const double sm_cy = has_prev_valid_ ? (alpha * measured_cy + (1.0 - alpha) * ema_cy_) : measured_cy;
    const double sm_ang = has_prev_valid_ ? (alpha * raw_angle_deg + (1.0 - alpha) * ema_angle_deg_) : raw_angle_deg;
    const double sm_w = has_prev_valid_ ? (alpha * measured_w_px + (1.0 - alpha) * ema_width_px_) : measured_w_px;
    const double raw_len_px = std::max(tb_w / std::max(0.5, std::abs(std::cos(raw_ang_rad))), measured_w_px * 2.5);
    const double sm_len = has_prev_valid_ ? (alpha * raw_len_px + (1.0 - alpha) * ema_length_px_) : raw_len_px;

    const cv::Point2d sm_c(sm_cx, sm_cy);
    auto tl_w = intersect_normal_ray_with_line(sm_c - u_axis * (0.5 * sm_len), n_axis, l_top);
    auto tr_w = intersect_normal_ray_with_line(sm_c + u_axis * (0.5 * sm_len), n_axis, l_top);
    auto br_w = intersect_normal_ray_with_line(sm_c + u_axis * (0.5 * sm_len), n_axis, l_bot);
    auto bl_w = intersect_normal_ray_with_line(sm_c - u_axis * (0.5 * sm_len), n_axis, l_bot);
    if (!tl_w || !tr_w || !br_w || !bl_w) {
        return handle_failure(frame_bgr, label_box_norm, "COND_NUMBER_EXCEEDED", t0, true, &diag);
    }

    const double sx_full = static_cast<double>(frame_bgr.cols) / ww;
    const double sy_full = static_cast<double>(frame_bgr.rows) / wh;
    const std::vector<cv::Point2f> src_full = {
        cv::Point2f(static_cast<float>(tl_w->x * sx_full), static_cast<float>(tl_w->y * sy_full)),
        cv::Point2f(static_cast<float>(tr_w->x * sx_full), static_cast<float>(tr_w->y * sy_full)),
        cv::Point2f(static_cast<float>(br_w->x * sx_full), static_cast<float>(br_w->y * sy_full)),
        cv::Point2f(static_cast<float>(bl_w->x * sx_full), static_cast<float>(bl_w->y * sy_full)),
    };

    const double L_full = std::max(20.0, static_cast<double>(cv::norm(src_full[1] - src_full[0])));
    const double W_full = std::max(8.0, static_cast<double>(cv::norm(src_full[3] - src_full[0])));
    const std::vector<cv::Point2f> dst_rect = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(config_.out_width), 0.0f),
        cv::Point2f(static_cast<float>(config_.out_width), static_cast<float>(config_.out_height)),
        cv::Point2f(0.0f, static_cast<float>(config_.out_height)),
    };

    cv::Mat H_full;
    bool used_persp = false;
    if (std::abs(raw_angle_deg) < config_.perspective_angle_thresh_deg && parallel_diff_deg < 2.0) {
        const std::vector<cv::Point2f> canonical_rect = {
            cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(L_full), 0.0f),
            cv::Point2f(static_cast<float>(L_full), static_cast<float>(W_full)), cv::Point2f(0.0f, static_cast<float>(W_full)),
        };
        cv::Mat aff = cv::estimateAffinePartial2D(src_full, canonical_rect);
        if (!aff.empty() && aff.rows == 2 && aff.cols == 3) {
            cv::Mat aff64, M_rigid = cv::Mat::eye(3, 3, CV_64F);
            aff.convertTo(aff64, CV_64F);
            aff64.copyTo(M_rigid(cv::Rect(0, 0, 3, 2)));
            cv::Mat M_scale = (cv::Mat_<double>(3, 3) <<
                config_.out_width / L_full, 0.0, 0.0,
                0.0, config_.out_height / W_full, 0.0,
                0.0, 0.0, 1.0);
            H_full = M_scale * M_rigid;
        }
    }
    if (H_full.empty()) {
        used_persp = true;
        H_full = cv::getPerspectiveTransform(src_full, dst_rect);
    }

    const double full_cx = 0.25 * (src_full[0].x + src_full[1].x + src_full[2].x + src_full[3].x);
    const double full_cy = 0.25 * (src_full[0].y + src_full[1].y + src_full[2].y + src_full[3].y);
    cv::Mat T_src_inv = (cv::Mat_<double>(3, 3) << 1.0, 0.0, full_cx, 0.0, 1.0, full_cy, 0.0, 0.0, 1.0);
    cv::Mat T_dst = (cv::Mat_<double>(3, 3) << 1.0, 0.0, -0.5 * config_.out_width, 0.0, 1.0, -0.5 * config_.out_height, 0.0, 0.0, 1.0);
    cv::Mat S_inv = (cv::Mat_<double>(3, 3) << L_full / config_.out_width, 0.0, 0.0, 0.0, W_full / config_.out_height, 0.0, 0.0, 0.0, 1.0);
    cv::Mat H_centered = T_dst * H_full * T_src_inv;
    cv::Mat H_norm = S_inv * H_centered;

    double cond = 999.0, det = 0.0;
    bool gate_ok = check_homography_gate(H_norm, config_.max_cond_number, config_.min_det_2x2, config_.max_det_2x2, cond, det);
    if (gate_ok) {
        const double h22 = H_centered.at<double>(2, 2);
        const double proj_bound = std::abs(H_centered.at<double>(2, 0) / h22) * (L_full * 0.5) +
                                  std::abs(H_centered.at<double>(2, 1) / h22) * (W_full * 0.5);
        if (!std::isfinite(proj_bound) || proj_bound >= 0.65) gate_ok = false;
    }

    diag.condition_number = cond;
    diag.det_2x2 = det;
    diag.used_perspective = used_persp;
    if (!gate_ok) {
        return handle_failure(frame_bgr, label_box_norm, "COND_NUMBER_EXCEEDED", t0, true, &diag);
    }

    std::array<cv::Point2f, 4> quad_norm{};
    for (int i = 0; i < 4; ++i) {
        quad_norm[i] = cv::Point2f(src_full[i].x / frame_bgr.cols, src_full[i].y / frame_bgr.rows);
    }

    TubeAlignResult res = diag;
    res.status = TubeAlignStatus::ALIGNED;
    res.geometry_ok = true;
    res.angle_deg = sm_ang;
    res.homography_3x3 = H_full.clone();
    res.quad_norm = quad_norm;
    res.has_quad = true;
    warp_outputs(frame_bgr, res.homography_3x3, res.quad_norm, label_box_norm, res.rectified_tube_bgr, res.rectified_label_bgr);

    has_prev_valid_ = true;
    consecutive_failures_ = 0;
    ema_cx_ = sm_cx;
    ema_cy_ = sm_cy;
    ema_angle_deg_ = sm_ang;
    ema_width_px_ = sm_w;
    ema_length_px_ = sm_len;
    prev_homography_ = H_full.clone();
    prev_quad_norm_ = quad_norm;

    res.align_ms = (monotonic_seconds() - t0) * 1000.0;
    last_status_ = res.to_json();
    return res;
}

}  // namespace igbt::vision
