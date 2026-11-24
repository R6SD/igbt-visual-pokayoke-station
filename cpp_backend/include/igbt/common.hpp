#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <bcrypt.h>
#ifdef ERROR
#undef ERROR
#endif
#ifdef DELETE
#undef DELETE
#endif
#ifdef OPTIONAL
#undef OPTIONAL
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace igbt {

using json = nlohmann::json;
using Box = std::array<double, 4>;

inline constexpr int kDefaultCameraWidth = 1280;
inline constexpr int kDefaultCameraHeight = 720;
inline constexpr double kDefaultTargetFps = 25.0;
inline constexpr int kDefaultInputSize = 96;
inline constexpr std::size_t kMaxUploadBytes = 8 * 1024 * 1024;

inline const std::array<std::string, 4> kLabelClasses = {
    "complete", "crossed_out", "covered", "absent"
};

struct ROISet {
    Box tube_roi{0.16, 0.42, 0.84, 0.58};
    Box label_roi{0.43, 0.45, 0.57, 0.55};
    Box label_contact_roi{0.40, 0.38, 0.60, 0.62};
    Box left_grip_zone{0.16, 0.34, 0.31, 0.82};
    Box right_grip_zone{0.69, 0.34, 0.84, 0.82};
    bool roi_calibrated{false};
};

struct AutoCaptureConfig {
    bool enabled{false};
    int every_n_frames{10};
    std::string label{"complete"};
    std::string pipe_color{"yellow"};
    std::string physical_tube_id{"UNGROUPED"};
    int max_samples{200};
    int captured_count{0};

    json to_json() const {
        return json{
            {"enabled", enabled},
            {"every_n_frames", every_n_frames},
            {"label", label},
            {"pipe_color", pipe_color},
            {"physical_tube_id", physical_tube_id},
            {"max_samples", max_samples},
            {"captured_count", captured_count},
        };
    }
};

// UTF-8 <-> std::filesystem::path helpers safe for Chinese Windows paths
inline std::filesystem::path path_from_utf8(const std::string& utf8_str) {
    if (utf8_str.empty()) return {};
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), static_cast<int>(utf8_str.size()), nullptr, 0);
    if (wlen <= 0) return std::filesystem::path(utf8_str);
    std::wstring wstr(static_cast<std::size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), static_cast<int>(utf8_str.size()), wstr.data(), wlen);
    return std::filesystem::path(wstr);
}

inline std::string path_to_utf8(const std::filesystem::path& p) {
    const std::wstring wstr = p.wstring();
    if (wstr.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string out(static_cast<std::size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), static_cast<int>(wstr.size()), out.data(), len, nullptr, nullptr);
    return out;
}

inline double monotonic_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline std::string utc_now_iso() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t tt = system_clock::to_time_t(now);
    std::tm gmt{};
    gmtime_s(&gmt, &tt);
    std::ostringstream oss;
    oss << std::put_time(&gmt, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setw(3) << std::setfill('0') << ms.count() << "Z";
    return oss.str();
}

inline std::string sha256_hex(const void* data, std::size_t size) {
    BCRYPT_ALG_HANDLE hAlg = nullptr;
    BCRYPT_HASH_HANDLE hHash = nullptr;
    std::array<unsigned char, 32> digest{};
    if (BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) {
        throw std::runtime_error("BCryptOpenAlgorithmProvider failed");
    }
    DWORD hashObjSize = 0, cbData = 0;
    BCryptGetProperty(hAlg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&hashObjSize), sizeof(DWORD), &cbData, 0);
    std::vector<unsigned char> hashObj(hashObjSize);
    if (BCryptCreateHash(hAlg, &hHash, hashObj.data(), hashObjSize, nullptr, 0, 0) != 0) {
        BCryptCloseAlgorithmProvider(hAlg, 0);
        throw std::runtime_error("BCryptCreateHash failed");
    }
    if (size > 0 && data != nullptr) {
        BCryptHashData(hHash, reinterpret_cast<PUCHAR>(const_cast<void*>(data)), static_cast<ULONG>(size), 0);
    }
    BCryptFinishHash(hHash, digest.data(), static_cast<ULONG>(digest.size()), 0);
    BCryptDestroyHash(hHash);
    BCryptCloseAlgorithmProvider(hAlg, 0);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned char b : digest) oss << std::setw(2) << static_cast<int>(b);
    return oss.str();
}

inline std::string sha256_hex(const std::string& bytes) {
    return sha256_hex(bytes.data(), bytes.size());
}

inline std::string sha256_hex(const std::vector<uint8_t>& bytes) {
    return sha256_hex(bytes.data(), bytes.size());
}

inline std::string random_token_hex(std::size_t num_bytes = 24) {
    std::vector<unsigned char> buf(num_bytes);
    if (BCryptGenRandom(nullptr, buf.data(), static_cast<ULONG>(buf.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        throw std::runtime_error("BCryptGenRandom failed");
    }
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned char b : buf) oss << std::setw(2) << static_cast<int>(b);
    return oss.str();
}

inline bool constant_time_equals(const std::string& a, const std::string& b) {
    const std::string ha = sha256_hex(a);
    const std::string hb = sha256_hex(b);
    unsigned char diff = 0;
    for (std::size_t i = 0; i < ha.size(); ++i) {
        diff |= static_cast<unsigned char>(ha[i] ^ hb[i]);
    }
    return a.size() == b.size() && diff == 0;
}

inline bool read_file_bytes(const std::filesystem::path& p, std::vector<uint8_t>& out) {
    std::ifstream ifs(p, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    if (sz < 0) return false;
    out.resize(static_cast<std::size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (!out.empty()) {
        ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    }
    return static_cast<bool>(ifs);
}

inline bool write_file_bytes_atomic(const std::filesystem::path& p, const void* data, std::size_t size) {
    std::error_code ec;
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    auto tmp = p;
    tmp += L".tmp_" + std::to_wstring(GetCurrentProcessId()) + L"_" + path_from_utf8(random_token_hex(4)).wstring();
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) return false;
        if (size > 0 && data != nullptr) {
            ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        }
        ofs.flush();
        if (!ofs) {
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        std::filesystem::remove(p, ec);
        ec.clear();
        std::filesystem::rename(tmp, p, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    return true;
}

inline cv::Mat imread_unicode(const std::filesystem::path& p, int flags = cv::IMREAD_COLOR) {
    std::vector<uint8_t> buf;
    if (!read_file_bytes(p, buf) || buf.empty()) return {};
    return cv::imdecode(buf, flags);
}

inline bool imwrite_unicode(const std::filesystem::path& p, const cv::Mat& img, int jpeg_quality = 90) {
    if (img.empty()) return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") ext = ".jpg";
    std::vector<uint8_t> encoded;
    std::vector<int> params;
    if (ext == ".jpg" || ext == ".jpeg") {
        params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    }
    if (!cv::imencode(ext, img, encoded, params) || encoded.empty()) return false;
    return write_file_bytes_atomic(p, encoded.data(), encoded.size());
}

inline bool is_path_within(const std::filesystem::path& base, const std::filesystem::path& target) {
    std::error_code ec;
    auto abs_base = std::filesystem::weakly_canonical(base, ec);
    if (ec) return false;
    auto abs_target = std::filesystem::weakly_canonical(target, ec);
    if (ec) return false;
    auto b_it = abs_base.begin();
    auto t_it = abs_target.begin();
    for (; b_it != abs_base.end() && t_it != abs_target.end(); ++b_it, ++t_it) {
        if (*b_it != *t_it) return false;
    }
    return b_it == abs_base.end();
}

// Shared 64-D spatial + HSV + edge feature extractor (< 0.15 ms) for pure C++ cv::ml::ANN_MLP training & 25 FPS inference
inline cv::Mat extract_label_features(const cv::Mat& bgr_input) {
    constexpr int kFeatDim = 64;
    cv::Mat feat = cv::Mat::zeros(1, kFeatDim, CV_32F);
    if (bgr_input.empty()) return feat;

    cv::Mat bgr;
    cv::resize(bgr_input, bgr, cv::Size(96, 64), 0, 0, cv::INTER_AREA);

    cv::Mat hsv, gray, gx, gy, mag;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    int idx = 0;
    constexpr int kCols = 4;
    constexpr int kRows = 3;
    const int cw = bgr.cols / kCols;
    const int ch = bgr.rows / kRows;

    double total_white = 0.0;
    double total_dark = 0.0;
    double center_v_sum = 0.0;
    int center_cnt = 0;
    double border_v_sum = 0.0;
    int border_cnt = 0;
    double diag_energy = 0.0;

    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            cv::Rect cell(c * cw, r * ch, cw, ch);
            cv::Mat cell_hsv = hsv(cell);
            cv::Mat cell_mag = mag(cell);

            int white_px = 0;
            int dark_px = 0;
            double v_sum = 0.0;
            double v_sq = 0.0;
            const int n = cell.width * cell.height;

            for (int y = 0; y < cell.height; ++y) {
                const auto* hp = cell_hsv.ptr<cv::Vec3b>(y);
                for (int x = 0; x < cell.width; ++x) {
                    const int s = hp[x][1];
                    const int v = hp[x][2];
                    v_sum += v;
                    v_sq += static_cast<double>(v) * v;
                    if (v >= 165 && s <= 68) ++white_px;
                    if (v <= 85) ++dark_px;
                }
            }
            const double mean_v = (v_sum / n) / 255.0;
            const double var_v = std::max(0.0, (v_sq / n) - (v_sum / n) * (v_sum / n));
            const double std_v = std::min(1.0, std::sqrt(var_v) / 128.0);
            const double white_r = static_cast<double>(white_px) / n;
            const double dark_r = static_cast<double>(dark_px) / n;
            const double edge_r = std::min(1.0, cv::mean(cell_mag)[0] / 180.0);

            feat.at<float>(0, idx++) = static_cast<float>(mean_v);
            feat.at<float>(0, idx++) = static_cast<float>(std_v);
            feat.at<float>(0, idx++) = static_cast<float>(white_r);
            feat.at<float>(0, idx++) = static_cast<float>(dark_r);
            feat.at<float>(0, idx++) = static_cast<float>(edge_r);

            total_white += white_px;
            total_dark += dark_px;
            if (r == 1 && (c == 1 || c == 2)) {
                center_v_sum += mean_v;
                ++center_cnt;
            } else {
                border_v_sum += mean_v;
                ++border_cnt;
            }
        }
    }

    // Diagonal stroke sensitivity (distinguishes crossed_out strokes from uniform blocks)
    for (int y = 8; y < bgr.rows - 8; ++y) {
        const float* dxp = gx.ptr<float>(y);
        const float* dyp = gy.ptr<float>(y);
        for (int x = 12; x < bgr.cols - 12; ++x) {
            float ax = std::abs(dxp[x]);
            float ay = std::abs(dyp[x]);
            if (ax > 25.0f && ay > 25.0f) {
                diag_energy += std::min(ax, ay);
            }
        }
    }

    const double total_px = static_cast<double>(bgr.rows * bgr.cols);
    feat.at<float>(0, 60) = static_cast<float>(total_white / total_px);
    feat.at<float>(0, 61) = static_cast<float>(total_dark / total_px);
    const double c_mean = center_cnt > 0 ? center_v_sum / center_cnt : 0.0;
    const double b_mean = border_cnt > 0 ? border_v_sum / border_cnt : 0.0;
    feat.at<float>(0, 62) = static_cast<float>(std::clamp((c_mean - b_mean + 1.0) * 0.5, 0.0, 1.0));
    feat.at<float>(0, 63) = static_cast<float>(std::min(1.0, diag_energy / (total_px * 45.0)));

    return feat;
}

}  // namespace igbt
