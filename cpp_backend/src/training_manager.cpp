#include "igbt/training_manager.hpp"

#include <opencv2/ml.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace igbt {

namespace {

constexpr const char* kManifestHeader =
    "sample_id,path,label,physical_tube_id,batch_id,video_group_id,pipe_color,box_xyxy,crop_box_xyxy,source_digest,captured_at_utc";

constexpr std::size_t kMaxSampleUploadBytes = 10 * 1024 * 1024;
constexpr int kMaxDecodedDimension = 8192;
constexpr int64_t kMaxDecodedPixels = 33554432;

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

std::string to_upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

bool is_valid_dataset_label(const std::string& label) {
    return std::find(kAllDatasetLabels.begin(), kAllDatasetLabels.end(), label) != kAllDatasetLabels.end();
}

int label_class_index(const std::string& label) {
    for (int i = 0; i < static_cast<int>(kLabelClasses.size()); ++i) {
        if (kLabelClasses[static_cast<std::size_t>(i)] == label) {
            return i;
        }
    }
    return -1;
}

int coco_category_id(const std::string& label) {
    for (std::size_t i = 0; i < kAllDatasetLabels.size(); ++i) {
        if (kAllDatasetLabels[i] == label) {
            return static_cast<int>(i) + 1;
        }
    }
    return 1;
}

bool is_ungrouped_tube(const std::string& tube_id) {
    const std::string cleaned = to_upper_copy(trim_copy(tube_id));
    if (cleaned.empty()) return true;
    if (cleaned == kUngroupedTubeId) return true;
    const std::string prefix = std::string(kUngroupedTubeId) + "_";
    return cleaned.rfind(prefix, 0) == 0;
}

bool is_valid_box_coords(const Box& b) {
    for (double v : b) {
        if (!std::isfinite(v)) return false;
    }
    return 0.0 <= b[0] && b[0] < b[2] && b[2] <= 1.0 &&
           0.0 <= b[1] && b[1] < b[3] && b[3] <= 1.0;
}

Box normalize_crop_box(const std::optional<Box>& maybe_box) {
    if (maybe_box.has_value() && is_valid_box_coords(*maybe_box)) {
        const Box& b = *maybe_box;
        return Box{
            std::clamp(b[0], 0.0, 1.0),
            std::clamp(b[1], 0.0, 1.0),
            std::clamp(b[2], 0.0, 1.0),
            std::clamp(b[3], 0.0, 1.0)
        };
    }
    return Box{0.0, 0.0, 1.0, 1.0};
}

Box frame_box_to_crop_box(
    const std::optional<Box>& box_xyxy,
    const std::optional<Box>& crop_bounds_xyxy
) {
    if (!box_xyxy.has_value() || !is_valid_box_coords(*box_xyxy)) {
        return Box{0.0, 0.0, 1.0, 1.0};
    }
    const Box& b = *box_xyxy;
    const Box bounds = (crop_bounds_xyxy.has_value() && is_valid_box_coords(*crop_bounds_xyxy))
                           ? *crop_bounds_xyxy
                           : b;
    const double cw = bounds[2] - bounds[0];
    const double ch = bounds[3] - bounds[1];
    if (cw <= 0.0 || ch <= 0.0) {
        return Box{0.0, 0.0, 1.0, 1.0};
    }
    const double nx1 = std::clamp((b[0] - bounds[0]) / cw, 0.0, 1.0);
    const double ny1 = std::clamp((b[1] - bounds[1]) / ch, 0.0, 1.0);
    const double nx2 = std::clamp((b[2] - bounds[0]) / cw, 0.0, 1.0);
    const double ny2 = std::clamp((b[3] - bounds[1]) / ch, 0.0, 1.0);
    if (nx2 <= nx1 || ny2 <= ny1) {
        return Box{0.0, 0.0, 1.0, 1.0};
    }
    return Box{nx1, ny1, nx2, ny2};
}

std::string format_box_csv(const Box& b) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << b[0] << "," << b[1] << "," << b[2] << "," << b[3];
    return oss.str();
}

std::optional<Box> parse_csv_box(const std::string& raw) {
    const std::string text = trim_copy(raw);
    if (text.empty()) return std::nullopt;
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        auto pos = text.find(',', start);
        if (pos == std::string::npos) {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, pos - start));
        start = pos + 1;
    }
    if (parts.size() != 4) return std::nullopt;
    Box out{};
    try {
        for (std::size_t i = 0; i < 4; ++i) {
            std::size_t idx = 0;
            const std::string p = trim_copy(parts[i]);
            out[i] = std::stod(p, &idx);
            if (idx != p.size()) return std::nullopt;
        }
    } catch (...) {
        return std::nullopt;
    }
    if (!is_valid_box_coords(out)) return std::nullopt;
    return out;
}

std::string csv_escape(const std::string& field) {
    bool needs_quotes = false;
    for (char c : field) {
        if (c == ',' || c == '"' || c == '\r' || c == '\n') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) return field;
    std::string out;
    out.reserve(field.size() + 4);
    out.push_back('"');
    for (char c : field) {
        if (c == '"') {
            out.append("\"\"");
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

std::vector<std::vector<std::string>> parse_csv_records(const std::string& content) {
    std::vector<std::vector<std::string>> records;
    std::size_t i = 0;
    // Strip UTF-8 BOM if present
    if (content.size() >= 3 &&
        static_cast<unsigned char>(content[0]) == 0xEF &&
        static_cast<unsigned char>(content[1]) == 0xBB &&
        static_cast<unsigned char>(content[2]) == 0xBF) {
        i = 3;
    }

    std::vector<std::string> row;
    std::string cell;
    bool in_quotes = false;

    while (i < content.size()) {
        const char c = content[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < content.size() && content[i + 1] == '"') {
                    cell.push_back('"');
                    i += 2;
                } else {
                    in_quotes = false;
                    ++i;
                }
            } else {
                cell.push_back(c);
                ++i;
            }
        } else {
            if (c == '"') {
                in_quotes = true;
                ++i;
            } else if (c == ',') {
                row.push_back(std::move(cell));
                cell.clear();
                ++i;
            } else if (c == '\r' || c == '\n') {
                if (c == '\r' && i + 1 < content.size() && content[i + 1] == '\n') {
                    ++i;
                }
                row.push_back(std::move(cell));
                cell.clear();
                bool non_empty = false;
                for (const auto& f : row) {
                    if (!f.empty()) {
                        non_empty = true;
                        break;
                    }
                }
                if (non_empty) {
                    records.push_back(std::move(row));
                }
                row.clear();
                ++i;
            } else {
                cell.push_back(c);
                ++i;
            }
        }
    }
    if (!cell.empty() || !row.empty()) {
        row.push_back(std::move(cell));
        bool non_empty = false;
        for (const auto& f : row) {
            if (!f.empty()) {
                non_empty = true;
                break;
            }
        }
        if (non_empty) {
            records.push_back(std::move(row));
        }
    }
    return records;
}

std::string compact_utc_timestamp(bool include_date) {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    std::tm gmt{};
    gmtime_s(&gmt, &tt);
    std::ostringstream oss;
    if (include_date) {
        oss << std::put_time(&gmt, "%Y%m%d_%H%M%S");
    } else {
        oss << std::put_time(&gmt, "%H%M%S");
    }
    return oss.str();
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

std::pair<int, uint64_t> compute_dir_usage(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return {0, 0};
    }
    int count = 0;
    uint64_t total_bytes = 0;
    for (std::filesystem::recursive_directory_iterator it(
             dir, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        std::error_code st_ec;
        if (it->is_regular_file(st_ec)) {
            const auto sz = it->file_size(st_ec);
            if (!st_ec) {
                ++count;
                total_bytes += sz;
            }
        }
    }
    return {count, total_bytes};
}

double round_to(double val, int decimals) {
    const double scale = std::pow(10.0, decimals);
    return std::round(val * scale) / scale;
}

std::string sanitize_identifier(const std::string& raw, std::size_t max_len = 64) {
    const std::string trimmed = trim_copy(raw);
    if (trimmed.empty()) return {};
    std::string out;
    out.reserve(std::min(trimmed.size(), max_len));
    for (char c : trimmed) {
        if (out.size() >= max_len) break;
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    // Disallow '..' path traversal sequences
    while (out.find("..") != std::string::npos) {
        out.replace(out.find(".."), 2, "_");
    }
    if (out == "." || out == "_") return {};
    return out;
}

std::pair<cv::Mat, Box> synthesize_roi_sample_with_box(
    const cv::Mat* base_bgr,
    const std::string& label,
    const std::string& pipe_color,
    int variant_idx,
    std::mt19937& rng
) {
    const int h = kSavedSampleHeight;
    const int w = kSavedSampleWidth;
    cv::Mat canvas(h, w, CV_8UC3);

    auto rand_int = [&rng](int lo, int hi) -> int {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng);
    };

    cv::Scalar bg_color;
    if (pipe_color == "blue") {
        bg_color = cv::Scalar(
            175 + rand_int(-12, 12),
            105 + rand_int(-10, 10),
            35 + rand_int(-8, 8)
        );
    } else {
        bg_color = cv::Scalar(
            42 + rand_int(-10, 10),
            178 + rand_int(-12, 12),
            215 + rand_int(-12, 12)
        );
    }
    canvas.setTo(bg_color);

    if (base_bgr != nullptr && !base_bgr->empty()) {
        cv::Mat resized_base;
        cv::resize(*base_bgr, resized_base, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        if (resized_base.channels() == 3) {
            cv::addWeighted(canvas, 0.55, resized_base, 0.45, 0.0, canvas);
        }
    }

    const int y_top = 18 + rand_int(-3, 3);
    const int y_bot = h - 18 + rand_int(-3, 3);
    cv::line(canvas, cv::Point(0, y_top), cv::Point(w, y_top), cv::Scalar(235, 240, 245), 2, cv::LINE_AA);
    cv::line(canvas, cv::Point(0, y_bot), cv::Point(w, y_bot), cv::Scalar(90, 95, 100), 2, cv::LINE_AA);

    const int lx1 = 34 + rand_int(-8, 8);
    const int ly1 = 28 + rand_int(-6, 6);
    const int lx2 = w - 34 + rand_int(-8, 8);
    const int ly2 = h - 28 + rand_int(-6, 6);

    const Box box_norm{
        round_to(static_cast<double>(lx1) / w, 4),
        round_to(static_cast<double>(ly1) / h, 4),
        round_to(static_cast<double>(lx2) / w, 4),
        round_to(static_cast<double>(ly2) / h, 4)
    };

    if (label == "complete") {
        cv::rectangle(canvas, cv::Point(lx1, ly1), cv::Point(lx2, ly2), cv::Scalar(242, 244, 246), cv::FILLED);
        cv::rectangle(canvas, cv::Point(lx1, ly1), cv::Point(lx2, ly2), cv::Scalar(180, 185, 190), 1);
        const int step = 9 + (variant_idx % 3);
        for (int bx = lx1 + 14; bx < lx2 - 16; bx += step) {
            const int bh = 22 + ((bx + variant_idx) % 14);
            cv::line(canvas, cv::Point(bx, ly1 + 18), cv::Point(bx, ly1 + 18 + bh), cv::Scalar(45, 50, 58), 2);
        }
        cv::line(canvas, cv::Point(lx1 + 14, ly2 - 18), cv::Point(lx2 - 18, ly2 - 18), cv::Scalar(70, 75, 85), 2);
    } else if (label == "crossed_out") {
        cv::rectangle(canvas, cv::Point(lx1, ly1), cv::Point(lx2, ly2), cv::Scalar(236, 238, 240), cv::FILLED);
        for (int bx = lx1 + 16; bx < lx2 - 16; bx += 11) {
            cv::line(canvas, cv::Point(bx, ly1 + 20), cv::Point(bx, ly1 + 42), cv::Scalar(65, 70, 78), 2);
        }
        const int thickness = 7 + (variant_idx % 4);
        cv::line(canvas, cv::Point(lx1 + 6, ly1 + 8), cv::Point(lx2 - 6, ly2 - 8),
                 cv::Scalar(18, 20, 24), thickness, cv::LINE_AA);
        cv::line(canvas, cv::Point(lx1 + 8, ly2 - 10), cv::Point(lx2 - 8, ly1 + 10),
                 cv::Scalar(22, 24, 28), thickness - 1, cv::LINE_AA);
    } else if (label == "covered") {
        cv::rectangle(canvas, cv::Point(lx1, ly1), cv::Point(lx2, ly2), cv::Scalar(235, 235, 235), cv::FILLED);
        const int tx1 = lx1 - 6 + rand_int(-3, 3);
        const int ty1 = ly1 - 4 + rand_int(-3, 3);
        const int tx2 = lx2 + 6 + rand_int(-3, 3);
        const int ty2 = ly2 + 4 + rand_int(-3, 3);
        const cv::Scalar tape_color = (variant_idx % 2 == 0) ? cv::Scalar(38, 72, 165) : cv::Scalar(48, 52, 58);
        cv::rectangle(canvas, cv::Point(tx1, ty1), cv::Point(tx2, ty2), tape_color, cv::FILLED);
    } else if (label == "absent") {
        for (int rx = lx1 + 15; rx < lx2 - 15; rx += 18) {
            cv::circle(canvas, cv::Point(rx, (ly1 + ly2) / 2), 4, bg_color, cv::FILLED);
        }
    } else {
        cv::rectangle(canvas, cv::Point(lx1, ly1), cv::Point(lx2, ly2), cv::Scalar(250, 252, 255), cv::FILLED);
        cv::circle(canvas, cv::Point(w / 2 + rand_int(-15, 15), h / 2), 48, cv::Scalar(255, 255, 255), cv::FILLED);
        cv::GaussianBlur(canvas, canvas, cv::Size(15, 15), 5.0);
    }

    // Add sensor grain so every generated sample has distinct SHA-256 and realistic noise
    std::mt19937 noise_rng(static_cast<uint32_t>(2026 + variant_idx * 97 + coco_category_id(label) * 13));
    std::normal_distribution<float> gauss(0.0f, 4.5f);
    for (int y = 0; y < canvas.rows; ++y) {
        auto* row_ptr = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < canvas.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float v = static_cast<float>(row_ptr[x][c]) + gauss(noise_rng);
                row_ptr[x][c] = static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
            }
        }
    }

    return {canvas, box_norm};
}

// Generate realistic shop-floor optical augmentations strictly on training crops
std::vector<cv::Mat> generate_shopfloor_augmentations(const cv::Mat& src, uint32_t seed) {
    std::vector<cv::Mat> out;
    if (src.empty()) return out;
    out.reserve(4);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> alpha_dist(0.86, 1.15);
    std::uniform_real_distribution<double> beta_dist(-16.0, 18.0);
    std::uniform_int_distribution<int> glare_y_dist(10, std::max(12, src.rows - 18));

    // 1. Horizontal mirror (symmetric tube/label geometry)
    cv::Mat flipped;
    cv::flip(src, flipped, 1);
    out.push_back(std::move(flipped));

    // 2. Ambient brightness & contrast jitter
    cv::Mat jittered;
    src.convertTo(jittered, -1, alpha_dist(rng), beta_dist(rng));
    out.push_back(std::move(jittered));

    // 3. Slight defocus / motion softness
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(3, 3), 0.85);
    out.push_back(std::move(blurred));

    // 4. Transparent tube wall specular glare highlight
    cv::Mat glare = src.clone();
    cv::Mat overlay = src.clone();
    const int gy = glare_y_dist(rng);
    cv::line(overlay, cv::Point(0, gy), cv::Point(src.cols, std::min(src.rows - 1, gy + 3)),
             cv::Scalar(248, 250, 252), 5, cv::LINE_AA);
    cv::addWeighted(glare, 0.80, overlay, 0.20, 0.0, glare);
    out.push_back(std::move(glare));

    return out;
}

std::string serialize_mlp_to_xml(const cv::Ptr<cv::ml::ANN_MLP>& mlp) {
    cv::FileStorage fs(".xml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_XML);
    const std::string tag = mlp->getDefaultName().empty() ? "opencv_ml_ann_mlp" : std::string(mlp->getDefaultName());
    fs << tag << "{";
    mlp->write(fs);
    fs << "}";
    return fs.releaseAndGetString();
}

cv::Ptr<cv::ml::ANN_MLP> load_mlp_from_xml_string(const std::string& xml_str) {
    if (xml_str.empty()) return {};
    try {
        return cv::Algorithm::loadFromString<cv::ml::ANN_MLP>(xml_str);
    } catch (...) {
        return {};
    }
}

std::vector<std::string> extract_json_str_set(
    const json& obj,
    std::initializer_list<const char*> keys,
    bool exclude_ungrouped = false,
    bool lowercase = false
) {
    if (!obj.is_object()) return {};
    for (const char* key : keys) {
        auto it = obj.find(key);
        if (it == obj.end() || !it->is_array() || it->empty()) continue;
        std::set<std::string> unique_vals;
        bool valid_all = true;
        for (const auto& elem : *it) {
            if (!elem.is_string()) {
                valid_all = false;
                break;
            }
            std::string s = trim_copy(elem.get<std::string>());
            if (s.empty()) {
                valid_all = false;
                break;
            }
            if (exclude_ungrouped && is_ungrouped_tube(s)) continue;
            if (lowercase) s = to_lower_copy(std::move(s));
            unique_vals.insert(std::move(s));
        }
        if (valid_all && !unique_vals.empty()) {
            return std::vector<std::string>(unique_vals.begin(), unique_vals.end());
        }
    }
    return {};
}

}  // namespace

TrainingManager::TrainingManager(std::filesystem::path data_root)
    : TrainingManager([root = std::move(data_root)]() { return root; }) {}

TrainingManager::TrainingManager(std::function<std::filesystem::path()> data_root_getter)
    : data_root_getter_(std::move(data_root_getter)) {
    if (!data_root_getter_) {
        throw std::invalid_argument("TrainingManager requires a valid data_root_getter");
    }
    status_ = json{
        {"state", "IDLE"},
        {"message", "等待启动训练"},
        {"current_epoch", 0},
        {"total_epochs", 0},
        {"progress_pct", 0},
        {"history", json::array()},
        {"latest_run", nullptr},
        {"exported", false},
        {"activated", false},
        {"activation_status", nullptr},
        {"metrics_persisted", true},
        {"metrics_write_error", nullptr},
        {"error", nullptr},
        {"updated_at_utc", utc_now_iso()},
    };
}

TrainingManager::~TrainingManager() {
    shutting_down_.store(true);
    wait_for_completion();
}

void TrainingManager::wait_for_completion() {
    std::lock_guard<std::mutex> join_lock(worker_join_mutex_);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

bool TrainingManager::is_training() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string st = status_.value("state", "IDLE");
    return st == "TRAINING" || st == "EXPORTING_ONNX";
}

json TrainingManager::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

std::filesystem::path TrainingManager::data_root() const {
    return data_root_getter_();
}

std::filesystem::path TrainingManager::dataset_dir() const {
    auto d = data_root() / L"dataset";
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    return d;
}

std::filesystem::path TrainingManager::manifest_path() const {
    return dataset_dir() / L"samples_manifest.csv";
}

std::filesystem::path TrainingManager::coco_json_path() const {
    return dataset_dir() / L"annotations_coco.json";
}

std::filesystem::path TrainingManager::runs_dir() const {
    auto d = data_root() / L"runs" / L"label_classifier";
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    return d;
}

std::vector<TrainingManager::ManifestRow> TrainingManager::read_manifest_rows_unlocked() const {
    const auto mpath = manifest_path();
    std::vector<uint8_t> raw_bytes;
    if (!read_file_bytes(mpath, raw_bytes) || raw_bytes.empty()) {
        return {};
    }
    const std::string content(reinterpret_cast<const char*>(raw_bytes.data()), raw_bytes.size());
    const auto records = parse_csv_records(content);
    if (records.size() < 2) {
        return {};
    }

    const auto& header = records[0];
    std::map<std::string, std::size_t> col_idx;
    for (std::size_t c = 0; c < header.size(); ++c) {
        col_idx[trim_copy(header[c])] = c;
    }

    auto get_col = [&col_idx](const std::vector<std::string>& r, const char* name) -> std::string {
        auto it = col_idx.find(name);
        if (it == col_idx.end() || it->second >= r.size()) return {};
        return trim_copy(r[it->second]);
    };

    const auto ds_root = dataset_dir();
    std::vector<ManifestRow> out;
    out.reserve(records.size() - 1);

    for (std::size_t i = 1; i < records.size(); ++i) {
        const auto& rec = records[i];
        const std::string rel_utf8 = get_col(rec, "path");
        if (rel_utf8.empty()) continue;

        std::filesystem::path img_path = path_from_utf8(rel_utf8);
        if (!img_path.is_absolute()) {
            img_path = ds_root / img_path;
        }
        if (!is_path_within(ds_root, img_path)) {
            continue;
        }
        std::error_code ec;
        if (!std::filesystem::is_regular_file(img_path, ec)) {
            continue;
        }

        ManifestRow row;
        row.sample_id = get_col(rec, "sample_id");
        row.path = rel_utf8;
        row.label = get_col(rec, "label");
        row.physical_tube_id = get_col(rec, "physical_tube_id");
        row.batch_id = get_col(rec, "batch_id");
        row.video_group_id = get_col(rec, "video_group_id");
        row.pipe_color = get_col(rec, "pipe_color");
        if (row.pipe_color.empty()) row.pipe_color = "yellow";
        row.box_xyxy = get_col(rec, "box_xyxy");
        row.crop_box_xyxy = get_col(rec, "crop_box_xyxy");
        row.source_digest = get_col(rec, "source_digest");
        row.captured_at_utc = get_col(rec, "captured_at_utc");
        row.abs_path = std::filesystem::weakly_canonical(img_path, ec);
        if (ec) row.abs_path = img_path;
        out.push_back(std::move(row));
    }
    return out;
}

void TrainingManager::write_all_manifest_rows_unlocked(const std::vector<ManifestRow>& rows) const {
    std::ostringstream oss;
    oss << "\xEF\xBB\xBF" << kManifestHeader << "\n";
    for (const auto& r : rows) {
        oss << csv_escape(r.sample_id) << ","
            << csv_escape(r.path) << ","
            << csv_escape(r.label) << ","
            << csv_escape(r.physical_tube_id) << ","
            << csv_escape(r.batch_id) << ","
            << csv_escape(r.video_group_id) << ","
            << csv_escape(r.pipe_color) << ","
            << csv_escape(r.box_xyxy) << ","
            << csv_escape(r.crop_box_xyxy) << ","
            << csv_escape(r.source_digest) << ","
            << csv_escape(r.captured_at_utc) << "\n";
    }
    const std::string data = oss.str();
    if (!write_file_bytes_atomic(manifest_path(), data.data(), data.size())) {
        throw std::runtime_error("Failed to write samples_manifest.csv atomically");
    }
}

void TrainingManager::append_manifest_row_unlocked(const ManifestRow& row) const {
    auto existing = read_manifest_rows_unlocked();
    existing.push_back(row);
    write_all_manifest_rows_unlocked(existing);
}

std::tuple<bool, bool, std::string> TrainingManager::try_unlink_in_dataset_unlocked(
    const std::filesystem::path& file_path
) const {
    const auto ds_root = dataset_dir();
    if (!is_path_within(ds_root, file_path)) {
        return {false, false, "file path outside dataset directory"};
    }
    std::error_code ec;
    const auto resolved = std::filesystem::weakly_canonical(file_path, ec);
    if (ec) {
        return {false, false, ec.message()};
    }
    if (!is_path_within(ds_root, resolved)) {
        return {false, false, "resolved file path outside dataset directory"};
    }

    if (DeleteFileW(resolved.c_str()) != 0) {
        return {true, true, ""};
    }
    const DWORD win_err = GetLastError();
    if (win_err == ERROR_FILE_NOT_FOUND || win_err == ERROR_PATH_NOT_FOUND) {
        return {true, false, ""};
    }
    return {false, false, "DeleteFileW win32 error " + std::to_string(win_err)};
}

json TrainingManager::empty_coco_doc() {
    json categories = json::array();
    for (std::size_t i = 0; i < kAllDatasetLabels.size(); ++i) {
        categories.push_back(json{
            {"id", static_cast<int>(i) + 1},
            {"name", kAllDatasetLabels[i]},
        });
    }
    return json{
        {"images", json::array()},
        {"annotations", json::array()},
        {"categories", std::move(categories)},
    };
}

std::pair<json, bool> TrainingManager::load_coco_doc_unlocked(const std::filesystem::path& coco_path) const {
    std::error_code ec;
    if (!std::filesystem::exists(coco_path, ec)) {
        if (ec) return {json{}, false};
        return {empty_coco_doc(), true};
    }
    if (!std::filesystem::is_regular_file(coco_path, ec)) {
        return {json{}, false};
    }
    auto maybe_doc = read_json_file(coco_path);
    if (!maybe_doc.has_value() || !maybe_doc->is_object()) {
        // Never overwrite an unreadable or corrupted COCO file on append/sync
        return {json{}, false};
    }
    json& doc = *maybe_doc;
    if (!doc.contains("images") || !doc["images"].is_array() ||
        !doc.contains("annotations") || !doc["annotations"].is_array()) {
        return {json{}, false};
    }
    if (!doc.contains("categories") || !doc["categories"].is_array() || doc["categories"].empty()) {
        doc["categories"] = empty_coco_doc()["categories"];
    }
    return {std::move(doc), true};
}

void TrainingManager::append_coco_entry(
    json& coco,
    int img_id,
    int ann_id,
    const std::string& sample_id,
    const std::string& rel_path,
    const std::string& label,
    const Box& crop_box_xyxy,
    int image_width,
    int image_height
) {
    const int width = std::max(1, image_width);
    const int height = std::max(1, image_height);
    const Box b = normalize_crop_box(crop_box_xyxy);
    const double x1 = std::clamp(b[0] * width, 0.0, static_cast<double>(width));
    const double y1 = std::clamp(b[1] * height, 0.0, static_cast<double>(height));
    const double x2 = std::clamp(b[2] * width, 0.0, static_cast<double>(width));
    const double y2 = std::clamp(b[3] * height, 0.0, static_cast<double>(height));
    const double bw = std::max(1.0, std::min(static_cast<double>(width) - x1, x2 - x1));
    const double bh = std::max(1.0, std::min(static_cast<double>(height) - y1, y2 - y1));

    coco["images"].push_back(json{
        {"id", img_id},
        {"file_name", rel_path},
        {"sample_id", sample_id},
        {"width", width},
        {"height", height},
    });
    coco["annotations"].push_back(json{
        {"id", ann_id},
        {"image_id", img_id},
        {"category_id", coco_category_id(label)},
        {"bbox", json::array({round_to(x1, 1), round_to(y1, 1), round_to(bw, 1), round_to(bh, 1)})},
        {"area", round_to(bw * bh, 1)},
        {"iscrowd", 0},
    });
}

bool TrainingManager::append_coco_annotations_unlocked(const std::vector<json>& entries) const {
    if (entries.empty()) return true;
    const auto cpath = coco_json_path();
    auto [coco, ok] = load_coco_doc_unlocked(cpath);
    if (!ok) return false;

    int next_img_id = 1;
    for (const auto& im : coco["images"]) {
        if (im.is_object() && im.contains("id") && im["id"].is_number_integer()) {
            next_img_id = std::max(next_img_id, im["id"].get<int>() + 1);
        }
    }
    int next_ann_id = 1;
    for (const auto& an : coco["annotations"]) {
        if (an.is_object() && an.contains("id") && an["id"].is_number_integer()) {
            next_ann_id = std::max(next_ann_id, an["id"].get<int>() + 1);
        }
    }

    for (const auto& entry : entries) {
        Box box{0.0, 0.0, 1.0, 1.0};
        if (entry.contains("crop_box_xyxy") && entry["crop_box_xyxy"].is_array() && entry["crop_box_xyxy"].size() == 4) {
            box = Box{
                entry["crop_box_xyxy"][0].get<double>(),
                entry["crop_box_xyxy"][1].get<double>(),
                entry["crop_box_xyxy"][2].get<double>(),
                entry["crop_box_xyxy"][3].get<double>()
            };
        }
        append_coco_entry(
            coco,
            next_img_id++,
            next_ann_id++,
            entry.value("sample_id", ""),
            entry.value("rel_path", ""),
            entry.value("label", "complete"),
            box,
            entry.value("image_width", kSavedSampleWidth),
            entry.value("image_height", kSavedSampleHeight)
        );
    }
    return write_json_atomic(cpath, coco);
}

void TrainingManager::sync_coco_delete_unlocked(const std::vector<std::string>& deleted_ids) const {
    if (deleted_ids.empty()) return;
    const auto cpath = coco_json_path();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(cpath, ec)) return;

    auto [coco, ok] = load_coco_doc_unlocked(cpath);
    if (!ok) return;

    const std::unordered_set<std::string> del_set(deleted_ids.begin(), deleted_ids.end());
    std::unordered_set<int> removed_img_ids;

    json kept_images = json::array();
    for (const auto& im : coco["images"]) {
        if (im.is_object() && del_set.count(im.value("sample_id", "")) > 0) {
            if (im.contains("id") && im["id"].is_number_integer()) {
                removed_img_ids.insert(im["id"].get<int>());
            }
        } else {
            kept_images.push_back(im);
        }
    }
    if (removed_img_ids.empty()) return;

    json kept_anns = json::array();
    for (const auto& an : coco["annotations"]) {
        if (an.is_object() && an.contains("image_id") && an["image_id"].is_number_integer() &&
            removed_img_ids.count(an["image_id"].get<int>()) > 0) {
            continue;
        }
        kept_anns.push_back(an);
    }
    coco["images"] = std::move(kept_images);
    coco["annotations"] = std::move(kept_anns);
    write_json_atomic(cpath, coco);
}

void TrainingManager::sync_coco_update_unlocked(
    const std::string& sample_id,
    const std::string& new_rel_path,
    const std::string& new_label
) const {
    const auto cpath = coco_json_path();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(cpath, ec)) return;

    auto [coco, ok] = load_coco_doc_unlocked(cpath);
    if (!ok) return;

    std::unordered_set<int> updated_img_ids;
    for (auto& im : coco["images"]) {
        if (im.is_object() && im.value("sample_id", "") == sample_id) {
            im["file_name"] = new_rel_path;
            if (im.contains("id") && im["id"].is_number_integer()) {
                updated_img_ids.insert(im["id"].get<int>());
            }
        }
    }
    if (updated_img_ids.empty()) return;

    const int cat_id = coco_category_id(new_label);
    for (auto& an : coco["annotations"]) {
        if (an.is_object() && an.contains("image_id") && an["image_id"].is_number_integer() &&
            updated_img_ids.count(an["image_id"].get<int>()) > 0) {
            an["category_id"] = cat_id;
        }
    }
    write_json_atomic(cpath, coco);
}

json TrainingManager::rebuild_coco_from_manifest() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto rows = read_manifest_rows_unlocked();
    json coco = empty_coco_doc();
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        int img_w = kSavedSampleWidth;
        int img_h = kSavedSampleHeight;
        cv::Mat img = imread_unicode(r.abs_path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            img_w = img.cols;
            img_h = img.rows;
        }
        auto maybe_box = parse_csv_box(r.crop_box_xyxy);
        if (!maybe_box.has_value() && r.sample_id.rfind("STD_", 0) == 0) {
            maybe_box = parse_csv_box(r.box_xyxy);
        }
        append_coco_entry(
            coco,
            static_cast<int>(i) + 1,
            static_cast<int>(i) + 1,
            r.sample_id,
            r.path,
            r.label.empty() ? "complete" : r.label,
            normalize_crop_box(maybe_box),
            img_w,
            img_h
        );
    }
    const auto cpath = coco_json_path();
    if (!write_json_atomic(cpath, coco)) {
        throw std::runtime_error("Failed to rebuild annotations_coco.json");
    }
    return json{
        {"rebuilt_annotations", coco["annotations"].size()},
        {"coco_path", path_to_utf8(cpath)},
    };
}

json TrainingManager::save_sample_image_impl(
    const cv::Mat& crop_bgr,
    const std::string& label,
    const std::string& pipe_color,
    const std::string& physical_tube_id,
    const std::string& batch_id,
    std::optional<Box> box_xyxy,
    const std::string& id_prefix,
    const std::string& box_fallback,
    const Box& coco_box_xyxy
) {
    const std::string clean_label = trim_copy(label);
    if (!is_valid_dataset_label(clean_label)) {
        throw std::invalid_argument("Unsupported sample label: " + clean_label);
    }
    const std::string clean_color = to_lower_copy(trim_copy(pipe_color));
    if (clean_color != "yellow" && clean_color != "blue") {
        throw std::invalid_argument("Unsupported pipe_color: " + clean_color);
    }
    if (crop_bgr.empty()) {
        throw std::invalid_argument("Empty ROI crop image");
    }

    const auto ds_dir = dataset_dir();
    const auto img_dir = ds_dir / path_from_utf8("images") / path_from_utf8(clean_label);
    std::error_code ec;
    std::filesystem::create_directories(img_dir, ec);

    const std::string short_lbl = to_upper_copy(clean_label.substr(0, std::min<std::size_t>(4, clean_label.size())));
    const std::string rand_suffix = to_upper_copy(random_token_hex(3)).substr(0, 5);
    const std::string sample_id = id_prefix + "_" + short_lbl + "_" + compact_utc_timestamp(false) + "_" + rand_suffix;
    const std::string rel_posix = "images/" + clean_label + "/" + sample_id + ".jpg";
    const auto abs_path = ds_dir / path_from_utf8(rel_posix);
    if (!is_path_within(ds_dir, abs_path)) {
        throw std::invalid_argument("Invalid resolved sample path");
    }

    cv::Mat resized;
    cv::resize(crop_bgr, resized, cv::Size(kSavedSampleWidth, kSavedSampleHeight), 0, 0, cv::INTER_CUBIC);
    if (!imwrite_unicode(abs_path, resized, 94)) {
        throw std::runtime_error("Failed to encode and write ROI sample JPEG");
    }
    const std::string sha = sha256_file_hex(abs_path);

    const std::string box_str = (box_xyxy.has_value() && is_valid_box_coords(*box_xyxy))
                                    ? format_box_csv(*box_xyxy)
                                    : box_fallback;
    const Box resolved_coco_box = normalize_crop_box(coco_box_xyxy);

    const std::string clean_tube = sanitize_identifier(physical_tube_id, 64);
    const std::string clean_batch = sanitize_identifier(batch_id, 64);

    ManifestRow row;
    row.sample_id = sample_id;
    row.path = rel_posix;
    row.label = clean_label;
    if (!clean_tube.empty() && !is_ungrouped_tube(clean_tube)) {
        row.physical_tube_id = clean_tube;
        row.batch_id = clean_batch.empty() ? kUngroupedBatchId : clean_batch;
        row.video_group_id = "VID_" + clean_tube;
    } else {
        row.physical_tube_id = kUngroupedTubeId;
        row.batch_id = clean_batch.empty() ? kUngroupedBatchId : clean_batch;
        row.video_group_id = kUngroupedVideoGroupId;
    }
    row.pipe_color = clean_color;
    row.box_xyxy = box_str;
    row.crop_box_xyxy = format_box_csv(resolved_coco_box);
    row.source_digest = "";
    row.captured_at_utc = utc_now_iso();
    row.abs_path = abs_path;
    row.image_sha256 = sha;

    bool coco_synced = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        append_manifest_row_unlocked(row);
        coco_synced = append_coco_annotations_unlocked({json{
            {"sample_id", row.sample_id},
            {"rel_path", row.path},
            {"label", row.label},
            {"crop_box_xyxy", json::array({resolved_coco_box[0], resolved_coco_box[1], resolved_coco_box[2], resolved_coco_box[3]})},
            {"image_width", resized.cols},
            {"image_height", resized.rows},
        }});
    }

    return json{
        {"sample_id", row.sample_id},
        {"path", row.path},
        {"label", row.label},
        {"physical_tube_id", row.physical_tube_id},
        {"batch_id", row.batch_id},
        {"video_group_id", row.video_group_id},
        {"pipe_color", row.pipe_color},
        {"box_xyxy", row.box_xyxy},
        {"crop_box_xyxy", row.crop_box_xyxy},
        {"source_digest", row.source_digest},
        {"captured_at_utc", row.captured_at_utc},
        {"sha256", sha},
        {"coco_synced", coco_synced},
        {"image_url", "/api/dataset/sample/" + row.sample_id + "/image"},
    };
}

json TrainingManager::capture_sample(
    const cv::Mat& frame_bgr,
    const cv::Mat& crop_bgr,
    const std::string& label,
    const std::string& pipe_color,
    const std::string& tube_id,
    const std::string& batch_id,
    std::optional<Box> frame_box,
    std::optional<Box> crop_bounds_xyxy
) {
    cv::Mat effective_crop = crop_bgr;
    if (effective_crop.empty() && !frame_bgr.empty()) {
        if (frame_box.has_value() && is_valid_box_coords(*frame_box)) {
            const Box& b = *frame_box;
            const int x1 = std::clamp(static_cast<int>(std::floor(b[0] * frame_bgr.cols)), 0, frame_bgr.cols - 1);
            const int y1 = std::clamp(static_cast<int>(std::floor(b[1] * frame_bgr.rows)), 0, frame_bgr.rows - 1);
            const int x2 = std::clamp(static_cast<int>(std::ceil(b[2] * frame_bgr.cols)), x1 + 1, frame_bgr.cols);
            const int y2 = std::clamp(static_cast<int>(std::ceil(b[3] * frame_bgr.rows)), y1 + 1, frame_bgr.rows);
            effective_crop = frame_bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        } else {
            effective_crop = frame_bgr;
        }
    }
    const Box coco_crop_box = frame_box_to_crop_box(frame_box, crop_bounds_xyxy);
    return save_sample_image_impl(
        effective_crop,
        label,
        pipe_color,
        tube_id,
        batch_id,
        frame_box,
        "SMP",
        "auto_tracked",
        coco_crop_box
    );
}

json TrainingManager::capture_sample(
    const cv::Mat& crop_bgr,
    const std::string& label,
    const std::string& pipe_color,
    const std::string& tube_id,
    const std::string& batch_id,
    std::optional<Box> frame_box
) {
    return capture_sample(cv::Mat{}, crop_bgr, label, pipe_color, tube_id, batch_id, frame_box, std::nullopt);
}

json TrainingManager::upload_sample_bytes(
    const std::vector<uint8_t>& image_bytes,
    const std::string& label,
    const std::string& pipe_color,
    const std::string& tube_id,
    const std::string& batch_id
) {
    if (!is_valid_dataset_label(trim_copy(label))) {
        throw std::invalid_argument("Unsupported sample label: " + label);
    }
    const std::string clean_color = to_lower_copy(trim_copy(pipe_color));
    if (clean_color != "yellow" && clean_color != "blue") {
        throw std::invalid_argument("Unsupported pipe_color: " + pipe_color);
    }
    if (image_bytes.empty()) {
        throw std::invalid_argument("Empty uploaded image bytes");
    }
    if (image_bytes.size() > kMaxSampleUploadBytes) {
        throw std::invalid_argument("Sample image exceeds maximum size limit");
    }

    const bool is_jpeg = image_bytes.size() >= 4 &&
                         image_bytes[0] == 0xFF && image_bytes[1] == 0xD8;
    const bool is_png = image_bytes.size() >= 8 &&
                        image_bytes[0] == 0x89 && image_bytes[1] == 'P' &&
                        image_bytes[2] == 'N' && image_bytes[3] == 'G' &&
                        image_bytes[4] == 0x0D && image_bytes[5] == 0x0A &&
                        image_bytes[6] == 0x1A && image_bytes[7] == 0x0A;
    if (!is_jpeg && !is_png) {
        throw std::invalid_argument("ONLY_VALID_JPEG_OR_PNG_BYTES_ACCEPTED");
    }

    cv::Mat decoded = cv::imdecode(image_bytes, cv::IMREAD_COLOR);
    if (decoded.empty() || decoded.cols < 1 || decoded.rows < 1 ||
        decoded.cols > kMaxDecodedDimension || decoded.rows > kMaxDecodedDimension ||
        static_cast<int64_t>(decoded.cols) * static_cast<int64_t>(decoded.rows) > kMaxDecodedPixels) {
        throw std::invalid_argument("Invalid, corrupted, or out-of-bounds JPEG/PNG image");
    }

    const Box full_box{0.0, 0.0, 1.0, 1.0};
    return save_sample_image_impl(
        decoded,
        label,
        clean_color,
        tube_id,
        batch_id,
        full_box,
        "UPL",
        "uploaded_image",
        full_box
    );
}

json TrainingManager::upload_sample_bytes(
    const std::string& image_bytes,
    const std::string& label,
    const std::string& pipe_color,
    const std::string& tube_id,
    const std::string& batch_id
) {
    const std::vector<uint8_t> vec(
        reinterpret_cast<const uint8_t*>(image_bytes.data()),
        reinterpret_cast<const uint8_t*>(image_bytes.data()) + image_bytes.size()
    );
    return upload_sample_bytes(vec, label, pipe_color, tube_id, batch_id);
}

json TrainingManager::list_samples(
    const std::string& label_filter,
    const std::string& search_query,
    int limit
) const {
    const std::string clean_label = trim_copy(label_filter);
    if (!clean_label.empty() && clean_label != "all" && !is_valid_dataset_label(clean_label)) {
        throw std::invalid_argument("Unsupported sample label filter: " + clean_label);
    }
    const std::string clean_query = to_lower_copy(trim_copy(search_query));
    const int clamped_limit = std::clamp(limit, 1, 2000);

    std::vector<ManifestRow> rows;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        rows = read_manifest_rows_unlocked();
    }

    json results = json::array();
    for (auto it = rows.rbegin(); it != rows.rend(); ++it) {
        if (!clean_label.empty() && clean_label != "all" && it->label != clean_label) {
            continue;
        }
        if (!clean_query.empty()) {
            const std::string haystack = to_lower_copy(
                it->sample_id + " " + it->label + " " + it->physical_tube_id + " " +
                it->batch_id + " " + it->pipe_color
            );
            if (haystack.find(clean_query) == std::string::npos) {
                continue;
            }
        }
        results.push_back(json{
            {"sample_id", it->sample_id},
            {"label", it->label},
            {"physical_tube_id", it->physical_tube_id},
            {"batch_id", it->batch_id},
            {"pipe_color", it->pipe_color.empty() ? "yellow" : it->pipe_color},
            {"box_xyxy", it->box_xyxy.empty() ? "auto_tracked" : it->box_xyxy},
            {"captured_at_utc", it->captured_at_utc},
            {"image_url", "/api/dataset/sample/" + it->sample_id + "/image"},
        });
        if (static_cast<int>(results.size()) >= clamped_limit) {
            break;
        }
    }
    return results;
}

std::optional<std::filesystem::path> TrainingManager::get_sample_image_path(const std::string& sample_id) const {
    const std::string clean_id = trim_copy(sample_id);
    if (clean_id.empty()) return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);
    const auto ds_root = dataset_dir();
    const auto rows = read_manifest_rows_unlocked();
    for (const auto& r : rows) {
        if (r.sample_id == clean_id) {
            std::error_code ec;
            if (is_path_within(ds_root, r.abs_path) && std::filesystem::is_regular_file(r.abs_path, ec)) {
                return r.abs_path;
            }
        }
    }
    return std::nullopt;
}

json TrainingManager::update_sample_annotation(
    const std::string& sample_id,
    const std::optional<std::string>& new_label,
    const std::optional<std::string>& new_tube_id,
    const std::optional<std::string>& new_batch_id,
    const std::optional<std::string>& new_pipe_color
) {
    const std::string clean_id = trim_copy(sample_id);
    if (clean_id.empty()) {
        throw std::out_of_range("Sample not found");
    }

    std::optional<std::string> validated_label;
    if (new_label.has_value() && !trim_copy(*new_label).empty()) {
        const std::string lbl = trim_copy(*new_label);
        if (!is_valid_dataset_label(lbl)) {
            throw std::invalid_argument("Unsupported label: " + lbl);
        }
        validated_label = lbl;
    }

    std::optional<std::string> validated_color;
    if (new_pipe_color.has_value() && !trim_copy(*new_pipe_color).empty()) {
        const std::string col = to_lower_copy(trim_copy(*new_pipe_color));
        if (col != "yellow" && col != "blue") {
            throw std::invalid_argument("Unsupported pipe_color: " + col);
        }
        validated_color = col;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto rows = read_manifest_rows_unlocked();
    ManifestRow* found = nullptr;
    bool label_changed = false;
    const auto ds_root = dataset_dir();

    for (auto& r : rows) {
        if (r.sample_id == clean_id) {
            if (validated_label.has_value() && *validated_label != r.label) {
                const auto new_dir = ds_root / path_from_utf8("images") / path_from_utf8(*validated_label);
                std::error_code ec;
                std::filesystem::create_directories(new_dir, ec);
                const std::string filename_utf8 = path_to_utf8(r.abs_path.filename());
                const std::string new_rel = "images/" + *validated_label + "/" + filename_utf8;
                const auto new_abs = ds_root / path_from_utf8(new_rel);

                if (is_path_within(ds_root, r.abs_path) && is_path_within(ds_root, new_abs) &&
                    std::filesystem::is_regular_file(r.abs_path, ec) && r.abs_path != new_abs) {
                    std::filesystem::rename(r.abs_path, new_abs, ec);
                    if (ec) {
                        std::filesystem::copy_file(r.abs_path, new_abs, std::filesystem::copy_options::overwrite_existing, ec);
                        if (ec) {
                            throw std::runtime_error("Failed to relocate relabeled sample image");
                        }
                        std::error_code rm_ec;
                        std::filesystem::remove(r.abs_path, rm_ec);
                    }
                    r.abs_path = new_abs;
                }
                r.label = *validated_label;
                r.path = new_rel;
                label_changed = true;
            }
            if (new_tube_id.has_value() && !trim_copy(*new_tube_id).empty()) {
                const std::string cleaned_tube = sanitize_identifier(*new_tube_id, 64);
                if (cleaned_tube.empty() || is_ungrouped_tube(cleaned_tube)) {
                    r.physical_tube_id = kUngroupedTubeId;
                    r.video_group_id = kUngroupedVideoGroupId;
                } else {
                    r.physical_tube_id = cleaned_tube;
                    r.video_group_id = "VID_" + cleaned_tube;
                }
            }
            if (new_batch_id.has_value() && !trim_copy(*new_batch_id).empty()) {
                const std::string cleaned_batch = sanitize_identifier(*new_batch_id, 64);
                if (!cleaned_batch.empty()) {
                    r.batch_id = cleaned_batch;
                }
            }
            if (validated_color.has_value()) {
                r.pipe_color = *validated_color;
            }
            found = &r;
            break;
        }
    }

    if (found == nullptr) {
        throw std::out_of_range("Sample not found: " + clean_id);
    }

    const ManifestRow updated = *found;
    write_all_manifest_rows_unlocked(rows);
    if (label_changed) {
        sync_coco_update_unlocked(updated.sample_id, updated.path, updated.label);
    }

    return json{
        {"sample_id", updated.sample_id},
        {"path", updated.path},
        {"label", updated.label},
        {"physical_tube_id", updated.physical_tube_id},
        {"batch_id", updated.batch_id},
        {"video_group_id", updated.video_group_id},
        {"pipe_color", updated.pipe_color},
        {"box_xyxy", updated.box_xyxy},
        {"crop_box_xyxy", updated.crop_box_xyxy},
        {"source_digest", updated.source_digest},
        {"captured_at_utc", updated.captured_at_utc},
        {"image_url", "/api/dataset/sample/" + updated.sample_id + "/image"},
    };
}

json TrainingManager::delete_sample(const std::string& sample_id) {
    const std::string clean_id = trim_copy(sample_id);
    if (clean_id.empty()) {
        throw std::out_of_range("Sample not found");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto rows = read_manifest_rows_unlocked();
    std::vector<ManifestRow> kept;
    kept.reserve(rows.size());
    std::optional<ManifestRow> target;

    for (const auto& r : rows) {
        if (r.sample_id == clean_id) {
            target = r;
        } else {
            kept.push_back(r);
        }
    }
    if (!target.has_value()) {
        throw std::out_of_range("Sample not found: " + clean_id);
    }

    // Unlink image file BEFORE removing manifest row; retain row if non-ENOENT unlink fails
    auto [can_remove, unlinked, err] = try_unlink_in_dataset_unlocked(target->abs_path);
    if (!can_remove) {
        throw std::runtime_error("Failed to delete sample file for " + clean_id + ": " + err);
    }

    write_all_manifest_rows_unlocked(kept);
    sync_coco_delete_unlocked({clean_id});

    return json{
        {"deleted", clean_id},
        {"deleted_file", unlinked},
        {"remaining_samples", kept.size()},
    };
}

json TrainingManager::batch_delete_samples(
    const std::vector<std::string>& sample_ids,
    const std::string& label_filter
) {
    std::unordered_set<std::string> id_set;
    for (const auto& s : sample_ids) {
        const std::string cleaned = trim_copy(s);
        if (!cleaned.empty()) {
            id_set.insert(cleaned);
        }
    }
    const std::string clean_label = trim_copy(label_filter);
    if (!clean_label.empty() && !is_valid_dataset_label(clean_label)) {
        throw std::invalid_argument("Unsupported label: " + clean_label);
    }
    if (id_set.empty() && clean_label.empty()) {
        throw std::invalid_argument("Must specify sample_ids or label to batch delete");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto rows = read_manifest_rows_unlocked();
    std::vector<ManifestRow> kept;
    std::vector<std::string> removed_ids;
    json failed_deletions = json::array();
    int deleted_files = 0;

    for (const auto& r : rows) {
        bool should_remove = false;
        if (!id_set.empty() && !clean_label.empty()) {
            should_remove = (id_set.count(r.sample_id) > 0) && (r.label == clean_label);
        } else if (!id_set.empty()) {
            should_remove = (id_set.count(r.sample_id) > 0);
        } else {
            should_remove = (r.label == clean_label);
        }

        if (should_remove) {
            auto [can_remove, unlinked, err] = try_unlink_in_dataset_unlocked(r.abs_path);
            if (can_remove) {
                removed_ids.push_back(r.sample_id);
                if (unlinked) ++deleted_files;
            } else {
                kept.push_back(r);
                failed_deletions.push_back(json{
                    {"sample_id", r.sample_id},
                    {"path", r.path},
                    {"error", err.empty() ? "unlink failed" : err},
                });
            }
        } else {
            kept.push_back(r);
        }
    }

    write_all_manifest_rows_unlocked(kept);
    if (!removed_ids.empty()) {
        sync_coco_delete_unlocked(removed_ids);
    }

    return json{
        {"deleted_count", removed_ids.size()},
        {"deleted_sample_ids", removed_ids},
        {"deleted_files", deleted_files},
        {"failed_count", failed_deletions.size()},
        {"failed_deletions", failed_deletions},
        {"remaining_samples", kept.size()},
    };
}

json TrainingManager::clear_dataset() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto rows = read_manifest_rows_unlocked();
    std::vector<ManifestRow> kept;
    std::vector<std::string> removed_ids;
    json failed_deletions = json::array();
    int deleted_files = 0;

    for (const auto& r : rows) {
        auto [can_remove, unlinked, err] = try_unlink_in_dataset_unlocked(r.abs_path);
        if (can_remove) {
            removed_ids.push_back(r.sample_id);
            if (unlinked) ++deleted_files;
        } else {
            kept.push_back(r);
            failed_deletions.push_back(json{
                {"sample_id", r.sample_id},
                {"path", r.path},
                {"error", err.empty() ? "unlink failed" : err},
            });
        }
    }

    write_all_manifest_rows_unlocked(kept);
    const auto cpath = coco_json_path();
    if (kept.empty()) {
        if (DeleteFileW(cpath.c_str()) == 0) {
            const DWORD win_err = GetLastError();
            if (win_err != ERROR_FILE_NOT_FOUND && win_err != ERROR_PATH_NOT_FOUND) {
                failed_deletions.push_back(json{
                    {"sample_id", ""},
                    {"path", "annotations_coco.json"},
                    {"error", "DeleteFileW win32 error " + std::to_string(win_err)},
                });
            }
        }
    } else if (!removed_ids.empty()) {
        sync_coco_delete_unlocked(removed_ids);
    }

    json sum = build_dataset_summary_unlocked(kept);
    sum["cleared"] = failed_deletions.empty();
    sum["deleted_count"] = removed_ids.size();
    sum["deleted_files"] = deleted_files;
    sum["failed_count"] = failed_deletions.size();
    sum["failed_deletions"] = failed_deletions;
    return sum;
}

json TrainingManager::bootstrap_standard_dataset(std::optional<cv::Mat> live_crop) {
    struct GroupSpec {
        const char* tube_id;
        const char* batch_id;
        const char* vid_id;
        const char* pipe_color;
    };
    static constexpr std::array<GroupSpec, 4> kGroups = {{
        {"TUBE_G01", "BATCH_2026_01", "VID_G01", "yellow"},
        {"TUBE_G02", "BATCH_2026_02", "VID_G02", "blue"},
        {"TUBE_G03", "BATCH_2026_03", "VID_G03", "yellow"},
        {"TUBE_G04", "BATCH_2026_04", "VID_G04", "blue"},
    }};

    cv::Mat clean_live_crop;
    std::string live_digest;
    if (live_crop.has_value() && !live_crop->empty()) {
        clean_live_crop = live_crop->isContinuous() ? *live_crop : live_crop->clone();
        live_digest = sha256_hex(clean_live_crop.data, clean_live_crop.total() * clean_live_crop.elemSize());
    }

    // Determine target slot for live_crop strictly confined to ONE group (default TUBE_G01 = index 0)
    int assigned_live_slot = -1;
    if (!clean_live_crop.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto existing_rows = read_manifest_rows_unlocked();
        std::set<std::string> prior_groups;
        for (const auto& r : existing_rows) {
            if (!r.source_digest.empty() && r.source_digest == live_digest && !is_ungrouped_tube(r.physical_tube_id)) {
                prior_groups.insert(r.physical_tube_id);
            }
        }
        if (prior_groups.empty()) {
            assigned_live_slot = 0;  // Strictly confine live_crop to TUBE_G01!
        } else if (prior_groups.size() == 1) {
            const std::string& pinned = *prior_groups.begin();
            for (int idx = 0; idx < static_cast<int>(kGroups.size()); ++idx) {
                if (pinned == kGroups[static_cast<std::size_t>(idx)].tube_id) {
                    assigned_live_slot = idx;
                    break;
                }
            }
        }
    }

    std::mt19937 rng(202611);
    const auto ds_dir = dataset_dir();
    std::vector<ManifestRow> pending_rows;
    std::vector<json> pending_coco;
    std::vector<std::string> live_crop_groups;
    int created = 0;

    for (int g_idx = 0; g_idx < static_cast<int>(kGroups.size()); ++g_idx) {
        const auto& grp = kGroups[static_cast<std::size_t>(g_idx)];
        const bool use_live = (g_idx == assigned_live_slot && !clean_live_crop.empty());
        if (use_live) {
            live_crop_groups.emplace_back(grp.tube_id);
        }
        const cv::Mat* group_base = use_live ? &clean_live_crop : nullptr;
        const std::string group_digest = use_live ? live_digest : "";

        for (const auto& label : kLabelClasses) {
            const auto img_dir = ds_dir / path_from_utf8("images") / path_from_utf8(label);
            std::error_code ec;
            std::filesystem::create_directories(img_dir, ec);

            for (int k = 0; k < 3; ++k) {
                const int variant = g_idx * 10 + k;
                auto [img, box_norm] = synthesize_roi_sample_with_box(group_base, label, grp.pipe_color, variant, rng);
                const std::string short_lbl = to_upper_copy(label.substr(0, 4));
                const std::string suffix = to_upper_copy(random_token_hex(2));
                std::ostringstream sid_oss;
                sid_oss << "STD_" << grp.tube_id << "_" << short_lbl << "_"
                        << std::setw(2) << std::setfill('0') << (k + 1) << "_" << suffix;
                const std::string sample_id = sid_oss.str();
                const std::string rel_posix = "images/" + label + "/" + sample_id + ".jpg";
                const auto abs_path = ds_dir / path_from_utf8(rel_posix);

                if (imwrite_unicode(abs_path, img, 93)) {
                    const std::string box_csv = format_box_csv(box_norm);
                    ManifestRow row;
                    row.sample_id = sample_id;
                    row.path = rel_posix;
                    row.label = label;
                    row.physical_tube_id = grp.tube_id;
                    row.batch_id = grp.batch_id;
                    row.video_group_id = grp.vid_id;
                    row.pipe_color = grp.pipe_color;
                    row.box_xyxy = box_csv;
                    row.crop_box_xyxy = box_csv;
                    row.source_digest = group_digest;
                    row.captured_at_utc = utc_now_iso();
                    row.abs_path = abs_path;
                    pending_rows.push_back(std::move(row));

                    pending_coco.push_back(json{
                        {"sample_id", sample_id},
                        {"rel_path", rel_posix},
                        {"label", label},
                        {"crop_box_xyxy", json::array({box_norm[0], box_norm[1], box_norm[2], box_norm[3]})},
                        {"image_width", img.cols},
                        {"image_height", img.rows},
                    });
                    ++created;
                }
            }
        }
    }

    // Isolated synthetic challenge samples (no live source crop blended)
    {
        const auto chal_dir = ds_dir / path_from_utf8("images") / path_from_utf8(kChallengeLabel);
        std::error_code ec;
        std::filesystem::create_directories(chal_dir, ec);
        for (int k = 0; k < 4; ++k) {
            auto [img, box_norm] = synthesize_roi_sample_with_box(nullptr, kChallengeLabel, "yellow", 90 + k, rng);
            const std::string suffix = to_upper_copy(random_token_hex(2));
            std::ostringstream sid_oss, idx_oss;
            idx_oss << std::setw(2) << std::setfill('0') << (k + 1);
            sid_oss << "STD_CHAL_" << idx_oss.str() << "_" << suffix;
            const std::string sample_id = sid_oss.str();
            const std::string rel_posix = std::string("images/") + kChallengeLabel + "/" + sample_id + ".jpg";
            const auto abs_path = ds_dir / path_from_utf8(rel_posix);

            if (imwrite_unicode(abs_path, img, 93)) {
                const std::string box_csv = format_box_csv(box_norm);
                ManifestRow row;
                row.sample_id = sample_id;
                row.path = rel_posix;
                row.label = kChallengeLabel;
                row.physical_tube_id = "TUBE_CHAL_ISO_" + idx_oss.str();
                row.batch_id = "BATCH_CHAL_ISO_" + idx_oss.str();
                row.video_group_id = "VID_CHAL_ISO_" + idx_oss.str();
                row.pipe_color = "yellow";
                row.box_xyxy = box_csv;
                row.crop_box_xyxy = box_csv;
                row.source_digest = "";
                row.captured_at_utc = utc_now_iso();
                row.abs_path = abs_path;
                pending_rows.push_back(std::move(row));

                pending_coco.push_back(json{
                    {"sample_id", sample_id},
                    {"rel_path", rel_posix},
                    {"label", kChallengeLabel},
                    {"crop_box_xyxy", json::array({box_norm[0], box_norm[1], box_norm[2], box_norm[3]})},
                    {"image_width", img.cols},
                    {"image_height", img.rows},
                });
                ++created;
            }
        }
    }

    bool coco_synced = false;
    json sum;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto all_rows = read_manifest_rows_unlocked();
        all_rows.insert(all_rows.end(), pending_rows.begin(), pending_rows.end());
        write_all_manifest_rows_unlocked(all_rows);
        coco_synced = append_coco_annotations_unlocked(pending_coco);
        sum = build_dataset_summary_unlocked(all_rows);
    }

    sum["bootstrapped_count"] = created;
    sum["coco_synced"] = coco_synced;
    sum["is_synthetic_bootstrap"] = true;
    sum["dataset_origin"] = "synthetic_smoke_test";
    sum["live_crop_groups"] = live_crop_groups;
    return sum;
}

TrainingManager::GroupSplit TrainingManager::select_group_split(
    std::vector<ManifestRow>& rows,
    const std::vector<std::string>& forbidden_groups,
    const std::vector<std::string>& forbidden_samples,
    const std::vector<std::string>& forbidden_images,
    const std::vector<std::string>& forbidden_sources
) {
    std::map<std::string, std::vector<ManifestRow>> by_tube;
    std::vector<ManifestRow> ungrouped_rows;
    std::vector<ManifestRow> challenges;

    auto row_digest = [](ManifestRow& r) -> std::string {
        if (!r.image_sha256.empty()) return to_lower_copy(r.image_sha256);
        if (r.abs_path.empty()) return {};
        r.image_sha256 = to_lower_copy(sha256_file_hex(r.abs_path));
        return r.image_sha256;
    };

    for (auto& r : rows) {
        if (r.label == kChallengeLabel) {
            challenges.push_back(r);
        } else if (label_class_index(r.label) >= 0) {
            if (is_ungrouped_tube(r.physical_tube_id)) {
                ungrouped_rows.push_back(r);
            } else {
                by_tube[r.physical_tube_id].push_back(r);
            }
        }
    }

    std::vector<std::string> tube_keys;
    tube_keys.reserve(by_tube.size());
    for (const auto& [tk, _] : by_tube) {
        tube_keys.push_back(tk);
    }
    if (tube_keys.size() < 3) {
        throw std::invalid_argument("独立料管分组不足 3 组，无法构建无泄漏的 train/val/test 划分（未分组样本不计入独立料管组）");
    }

    std::unordered_set<std::string> forb_grp;
    for (const auto& g : forbidden_groups) {
        if (!is_ungrouped_tube(g)) forb_grp.insert(g);
    }
    std::unordered_set<std::string> forb_smp(forbidden_samples.begin(), forbidden_samples.end());
    std::unordered_set<std::string> forb_img;
    for (const auto& d : forbidden_images) forb_img.insert(to_lower_copy(d));
    std::unordered_set<std::string> forb_src;
    for (const auto& s : forbidden_sources) forb_src.insert(to_lower_copy(s));

    std::map<std::string, std::set<std::string>> tube_sources;
    std::map<std::string, int> source_occurrences;
    for (const auto& tk : tube_keys) {
        std::set<std::string> srcs;
        for (const auto& r : by_tube[tk]) {
            const std::string sd = to_lower_copy(trim_copy(r.source_digest));
            if (!sd.empty()) srcs.insert(sd);
        }
        for (const auto& sd : srcs) ++source_occurrences[sd];
        tube_sources[tk] = std::move(srcs);
    }
    std::set<std::string> ungrouped_sources;
    for (const auto& r : ungrouped_rows) {
        const std::string sd = to_lower_copy(trim_copy(r.source_digest));
        if (!sd.empty()) ungrouped_sources.insert(sd);
    }
    for (const auto& sd : ungrouped_sources) ++source_occurrences[sd];

    std::set<std::string> ineligible_tubes;
    for (const auto& tk : tube_keys) {
        if (forb_grp.count(tk) > 0) {
            ineligible_tubes.insert(tk);
            continue;
        }
        bool shared_src = false;
        for (const auto& sd : tube_sources[tk]) {
            if (source_occurrences[sd] > 1 || forb_src.count(sd) > 0) {
                shared_src = true;
                break;
            }
        }
        if (shared_src) {
            ineligible_tubes.insert(tk);
            continue;
        }
        bool seen_item = false;
        for (auto& r : by_tube[tk]) {
            if (!forb_smp.empty() && forb_smp.count(r.sample_id) > 0) {
                seen_item = true;
                break;
            }
            if (!forb_img.empty()) {
                const std::string dig = row_digest(r);
                if (!dig.empty() && forb_img.count(dig) > 0) {
                    seen_item = true;
                    break;
                }
            }
        }
        if (seen_item) {
            ineligible_tubes.insert(tk);
        }
    }

    std::vector<std::string> holdout_candidates;
    for (const auto& tk : tube_keys) {
        if (ineligible_tubes.count(tk) == 0) {
            holdout_candidates.push_back(tk);
        }
    }
    if (holdout_candidates.size() < 2) {
        std::string msg = "可用于验证/测试留出的未训练独立料管组不足 2 组";
        if (!ineligible_tubes.empty()) {
            msg += "（历史已训练/共享源分组：";
            bool first = true;
            for (const auto& g : ineligible_tubes) {
                if (!first) msg += ", ";
                msg += g;
                first = false;
            }
            msg += "）";
        }
        throw std::invalid_argument(msg);
    }

    std::map<std::string, std::set<std::string>> tube_labels;
    for (const auto& tk : tube_keys) {
        for (const auto& r : by_tube[tk]) {
            tube_labels[tk].insert(r.label);
        }
    }
    std::set<std::string> ungrouped_labels;
    for (const auto& r : ungrouped_rows) {
        if (label_class_index(r.label) >= 0) {
            ungrouped_labels.insert(r.label);
        }
    }

    std::map<std::string, int> class_group_counts;
    std::vector<std::string> classes_needed_from_tubes;
    for (const auto& c : kLabelClasses) {
        int cnt = 0;
        for (const auto& tk : tube_keys) {
            if (tube_labels[tk].count(c) > 0) ++cnt;
        }
        class_group_counts[c] = cnt;
        if (ungrouped_labels.count(c) == 0) {
            classes_needed_from_tubes.push_back(c);
        }
    }

    auto pair_is_valid = [&](const std::string& val_tk, const std::string& test_tk) -> bool {
        if (tube_keys.size() <= 2) return false;
        const auto& v_lbls = tube_labels[val_tk];
        const auto& t_lbls = tube_labels[test_tk];
        for (const auto& c : classes_needed_from_tubes) {
            const int rem = class_group_counts[c]
                            - (v_lbls.count(c) > 0 ? 1 : 0)
                            - (t_lbls.count(c) > 0 ? 1 : 0);
            if (rem <= 0) return false;
        }
        return true;
    };

    std::optional<std::pair<std::string, std::string>> chosen;
    std::vector<std::string> full_holdouts;
    for (const auto& tk : holdout_candidates) {
        if (tube_labels[tk].size() == kLabelClasses.size()) {
            full_holdouts.push_back(tk);
        }
    }
    if (full_holdouts.size() >= 2 &&
        pair_is_valid(full_holdouts[full_holdouts.size() - 2], full_holdouts[full_holdouts.size() - 1])) {
        chosen = {full_holdouts[full_holdouts.size() - 2], full_holdouts[full_holdouts.size() - 1]};
    }

    if (!chosen.has_value()) {
        std::tuple<int, int, std::size_t, std::size_t> best_score{-1, -1, 0, 0};
        for (std::size_t i = 0; i < holdout_candidates.size(); ++i) {
            const auto& val_tk = holdout_candidates[i];
            const int v_cnt = static_cast<int>(tube_labels[val_tk].size());
            for (std::size_t j = i + 1; j < holdout_candidates.size(); ++j) {
                const auto& test_tk = holdout_candidates[j];
                if (!pair_is_valid(val_tk, test_tk)) continue;
                const int t_cnt = static_cast<int>(tube_labels[test_tk].size());
                const auto score = std::make_tuple(v_cnt + t_cnt, std::min(v_cnt, t_cnt), i, j);
                if (!chosen.has_value() || score > best_score) {
                    best_score = score;
                    chosen = {val_tk, test_tk};
                }
            }
        }
    }

    if (!chosen.has_value()) {
        std::vector<std::string> deficient;
        for (const auto& c : classes_needed_from_tubes) {
            if (class_group_counts[c] < 3) deficient.push_back(c);
        }
        std::string msg = "当前独立料管分组布局无法构建覆盖全部 4 类的训练集";
        if (!deficient.empty()) {
            msg += "，缺少独立分组类别：";
            for (std::size_t i = 0; i < deficient.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += deficient[i];
            }
        }
        throw std::invalid_argument(msg);
    }

    const auto [val_tube, test_tube] = *chosen;
    GroupSplit split;
    for (const auto& tk : tube_keys) {
        const auto& grp_rows = by_tube[tk];
        if (tk == val_tube) {
            split.val_rows.insert(split.val_rows.end(), grp_rows.begin(), grp_rows.end());
        } else if (tk == test_tube) {
            split.test_rows.insert(split.test_rows.end(), grp_rows.begin(), grp_rows.end());
        } else {
            split.train_groups.push_back(tk);
            split.train_rows.insert(split.train_rows.end(), grp_rows.begin(), grp_rows.end());
        }
    }

    // Ungrouped captures stay strictly in train_rows (never fake holdout groups)
    split.train_rows.insert(split.train_rows.end(), ungrouped_rows.begin(), ungrouped_rows.end());
    split.challenges = std::move(challenges);
    split.val_groups = {val_tube};
    split.test_groups = {test_tube};
    split.ungrouped_count = static_cast<int>(ungrouped_rows.size());

    std::set<std::string> train_classes;
    for (const auto& r : split.train_rows) train_classes.insert(r.label);
    for (const auto& c : kLabelClasses) {
        if (train_classes.count(c) == 0) {
            throw std::invalid_argument("分组划分校验未通过，训练集缺少类别：" + c);
        }
    }
    if (split.val_rows.empty() || split.test_rows.empty()) {
        throw std::invalid_argument("分组划分校验未通过，验证集或测试集为空");
    }

    return split;
}

json TrainingManager::build_dataset_summary_unlocked(const std::vector<ManifestRow>& rows) const {
    json counts = json::object();
    for (const auto& lbl : kAllDatasetLabels) {
        counts[lbl] = 0;
    }

    std::set<std::string> tubes;
    std::set<std::string> batches;
    int ungrouped_count = 0;

    for (const auto& r : rows) {
        if (counts.contains(r.label)) {
            counts[r.label] = counts[r.label].get<int>() + 1;
        }
        if (label_class_index(r.label) >= 0) {
            if (is_ungrouped_tube(r.physical_tube_id)) {
                ++ungrouped_count;
            } else {
                tubes.insert(r.physical_tube_id);
            }
            const std::string bid = trim_copy(r.batch_id);
            if (!bid.empty() && to_upper_copy(bid) != kUngroupedBatchId) {
                batches.insert(bid);
            }
        }
    }

    bool all_classes_present = true;
    for (const auto& c : kLabelClasses) {
        if (counts[c].get<int>() < 3) {
            all_classes_present = false;
            break;
        }
    }
    const bool enough_groups = tubes.size() >= 3;
    bool split_ok = false;
    std::string split_error;
    if (all_classes_present && enough_groups) {
        try {
            auto copy_rows = rows;
            (void)select_group_split(copy_rows);
            split_ok = true;
        } catch (const std::exception& ex) {
            split_error = ex.what();
        }
    }

    const bool ready = all_classes_present && enough_groups && split_ok;
    std::string reason;
    json reasons_list = json::array();
    if (ready) {
        reason = "数据集已满足分组隔离与四类均衡要求，可随时启动训练";
        reasons_list.push_back(reason);
    } else if (!split_error.empty()) {
        reason = split_error;
        reasons_list.push_back(split_error);
    } else {
        reason = "需至少 3 个独立料管组且每类（完整/划除/遮挡/脱落）至少 3 张样本；可点击「一键生成标准训练集」或实时抓拍补充";
        if (!enough_groups) {
            reasons_list.push_back("独立料管组不足 3 组（当前 " + std::to_string(tubes.size()) + " 组）");
        }
        if (!all_classes_present) {
            reasons_list.push_back("部分类别样本不足 3 张");
        }
    }

    json recent = json::array();
    const std::size_t max_recent = 16;
    for (std::size_t i = 0; i < rows.size() && i < max_recent; ++i) {
        const auto& r = rows[rows.size() - 1 - i];
        recent.push_back(json{
            {"sample_id", r.sample_id},
            {"label", r.label},
            {"physical_tube_id", r.physical_tube_id},
            {"batch_id", r.batch_id},
            {"pipe_color", r.pipe_color.empty() ? "yellow" : r.pipe_color},
            {"captured_at_utc", r.captured_at_utc},
            {"image_url", "/api/dataset/sample/" + r.sample_id + "/image"},
        });
    }

    const std::string mpath_str = path_to_utf8(manifest_path());
    const std::string cpath_str = path_to_utf8(coco_json_path());

    return json{
        {"total_samples", rows.size()},
        {"counts_by_class", counts},
        {"class_counts", counts},
        {"independent_tube_groups", tubes.size()},
        {"independent_batches", batches.size()},
        {"ungrouped_samples", ungrouped_count},
        {"ready_for_training", ready},
        {"ready_to_train", ready},
        {"readiness_reasons", reasons_list},
        {"readiness_reason", reason},
        {"manifest_path", mpath_str},
        {"coco_path", cpath_str},
        {"recent_samples", recent},
    };
}

json TrainingManager::summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto rows = read_manifest_rows_unlocked();
    return build_dataset_summary_unlocked(rows);
}

json TrainingManager::storage_stats() const {
    const auto ds_dir = dataset_dir();
    const auto r_dir = runs_dir();
    const json sum = summary();

    const auto [ds_files, ds_bytes] = compute_dir_usage(ds_dir);
    const auto [run_files, run_bytes] = compute_dir_usage(r_dir);

    int run_count = 0;
    std::error_code ec;
    if (std::filesystem::is_directory(r_dir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(r_dir, ec)) {
            std::error_code d_ec;
            if (entry.is_directory(d_ec)) {
                ++run_count;
            }
        }
    }

    int coco_annotations = 0;
    const auto cpath = coco_json_path();
    if (std::filesystem::is_regular_file(cpath, ec)) {
        auto maybe_coco = read_json_file(cpath);
        if (maybe_coco.has_value() && maybe_coco->is_object() &&
            maybe_coco->contains("annotations") && (*maybe_coco)["annotations"].is_array()) {
            coco_annotations = static_cast<int>((*maybe_coco)["annotations"].size());
        }
    }

    return json{
        {"dataset_dir", path_to_utf8(ds_dir)},
        {"manifest_path", path_to_utf8(manifest_path())},
        {"coco_path", path_to_utf8(cpath)},
        {"total_samples", sum["total_samples"]},
        {"class_counts", sum["class_counts"]},
        {"counts_by_class", sum["counts_by_class"]},
        {"independent_tube_groups", sum["independent_tube_groups"]},
        {"coco_annotations", coco_annotations},
        {"dataset_files", ds_files},
        {"dataset_size_mb", round_to(static_cast<double>(ds_bytes) / (1024.0 * 1024.0), 3)},
        {"runs_dir", path_to_utf8(r_dir)},
        {"run_count", run_count},
        {"models_files", run_files},
        {"models_size_mb", round_to(static_cast<double>(run_bytes) / (1024.0 * 1024.0), 3)},
    };
}

std::pair<std::string, std::filesystem::path> TrainingManager::allocate_run_dir_unlocked(
    const std::string& prefix
) const {
    const auto r_dir = runs_dir();
    const std::string base_id = prefix + "_" + compact_utc_timestamp(true);
    std::string run_id = base_id;
    auto run_dir = r_dir / path_from_utf8(run_id);
    std::error_code ec;
    while (std::filesystem::exists(run_dir, ec)) {
        run_id = base_id + "_" + random_token_hex(2);
        run_dir = r_dir / path_from_utf8(run_id);
    }
    std::filesystem::create_directories(run_dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create training run directory: " + ec.message());
    }
    return {run_id, run_dir};
}

std::pair<std::vector<TrainingManager::ManifestRow>, std::filesystem::path>
TrainingManager::snapshot_manifest_for_run_unlocked(
    const std::vector<ManifestRow>& rows,
    const std::filesystem::path& run_dir
) const {
    const auto snap_dir = run_dir / L"_snapshot";
    std::error_code ec;
    std::filesystem::create_directories(snap_dir, ec);

    std::vector<ManifestRow> snapshotted;
    snapshotted.reserve(rows.size());

    for (std::size_t idx = 0; idx < rows.size(); ++idx) {
        const auto& r = rows[idx];
        std::vector<uint8_t> img_bytes;
        if (!read_file_bytes(r.abs_path, img_bytes) || img_bytes.empty()) {
            continue;
        }
        std::string ext = r.abs_path.extension().string();
        if (ext.empty()) ext = ".jpg";
        const std::string safe_sid = sanitize_identifier(r.sample_id.empty() ? ("row" + std::to_string(idx)) : r.sample_id, 64);
        std::ostringstream fname;
        fname << std::setw(5) << std::setfill('0') << idx << "_" << safe_sid << ext;
        const auto dst = snap_dir / path_from_utf8(fname.str());
        if (!write_file_bytes_atomic(dst, img_bytes.data(), img_bytes.size())) {
            throw std::runtime_error("Failed to snapshot dataset image for training run");
        }
        ManifestRow copy_row = r;
        copy_row.abs_path = dst;
        copy_row.image_sha256 = sha256_hex(img_bytes);
        snapshotted.push_back(std::move(copy_row));
    }
    return {std::move(snapshotted), snap_dir};
}

json TrainingManager::start_training(
    int epochs,
    int input_size,
    int batch_size,
    double lr,
    bool resume_from_latest,
    bool use_augmentation,
    OnCompleteCallback on_complete
) {
    const int clamped_epochs = std::clamp(epochs, 2, 100);
    const int clamped_input = std::clamp(input_size, 32, 512);
    const int clamped_batch = std::clamp(batch_size, 2, 128);
    const double clamped_lr = std::clamp(lr, 1e-5, 0.5);

    std::string run_id;
    std::filesystem::path run_dir;
    std::filesystem::path snapshot_dir;
    std::vector<ManifestRow> snapshotted_rows;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string cur_state = status_.value("state", "IDLE");
        if (cur_state == "TRAINING" || cur_state == "EXPORTING_ONNX") {
            throw std::runtime_error("训练任务正在运行中，请等待当前训练完成");
        }

        const auto rows = read_manifest_rows_unlocked();
        const json sum = build_dataset_summary_unlocked(rows);
        if (!sum.value("ready_for_training", false)) {
            throw std::invalid_argument(sum.value("readiness_reason", "数据集未满足训练条件"));
        }

        std::tie(run_id, run_dir) = allocate_run_dir_unlocked("mlp_cpp");
        try {
            std::tie(snapshotted_rows, snapshot_dir) = snapshot_manifest_for_run_unlocked(rows, run_dir);
            auto verify_copy = snapshotted_rows;
            (void)select_group_split(verify_copy);
        } catch (...) {
            std::error_code rm_ec;
            std::filesystem::remove_all(run_dir, rm_ec);
            throw;
        }

        const std::string mode_desc = resume_from_latest ? "增量微调 (Warm-Start)" : "从零初始化训练";
        status_ = json{
            {"state", "TRAINING"},
            {"message", "正在执行分组防泄漏审计与启动 C++ ANN_MLP " + mode_desc + "…"},
            {"current_epoch", 0},
            {"total_epochs", clamped_epochs},
            {"progress_pct", 5},
            {"history", json::array()},
            {"latest_run", nullptr},
            {"exported", false},
            {"activated", false},
            {"activation_status", nullptr},
            {"metrics_persisted", true},
            {"metrics_write_error", nullptr},
            {"error", nullptr},
            {"updated_at_utc", utc_now_iso()},
        };
    }

    {
        std::lock_guard<std::mutex> join_lock(worker_join_mutex_);
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        worker_thread_ = std::thread(
            &TrainingManager::run_training_worker,
            this,
            clamped_epochs,
            clamped_input,
            clamped_batch,
            clamped_lr,
            resume_from_latest,
            use_augmentation,
            std::move(on_complete),
            std::move(run_id),
            std::move(run_dir),
            std::move(snapshotted_rows),
            std::move(snapshot_dir)
        );
    }

    return status();
}

void TrainingManager::run_training_worker(
    int epochs,
    int input_size,
    int batch_size,
    double lr,
    bool resume_from_latest,
    bool use_augmentation,
    OnCompleteCallback on_complete,
    std::string run_id,
    std::filesystem::path run_dir,
    std::vector<ManifestRow> snapshotted_rows,
    std::filesystem::path snapshot_dir
) {
    struct SnapshotCleaner {
        std::filesystem::path dir;
        ~SnapshotCleaner() {
            if (!dir.empty()) {
                std::error_code ec;
                std::filesystem::remove_all(dir, ec);
            }
        }
    } cleaner{snapshot_dir};

    try {
        const double t_start = monotonic_seconds();
        const std::string manifest_str = path_to_utf8(manifest_path());

        cv::Ptr<cv::ml::ANN_MLP> mlp;
        std::string resumed_from;
        std::vector<std::string> resumed_seen_groups;
        std::vector<std::string> resumed_seen_samples;
        std::vector<std::string> resumed_seen_images;
        std::vector<std::string> resumed_seen_sources;
        std::optional<GroupSplit> chosen_split;

        if (resume_from_latest) {
            const auto r_dir = runs_dir();
            std::error_code ec;
            struct XmlCandidate {
                std::filesystem::file_time_type mtime;
                std::filesystem::path xml_path;
            };
            std::vector<XmlCandidate> prev_xmls;
            if (std::filesystem::is_directory(r_dir, ec)) {
                for (const auto& entry : std::filesystem::directory_iterator(r_dir, ec)) {
                    std::error_code d_ec;
                    if (!entry.is_directory(d_ec) || entry.path() == run_dir) continue;
                    const auto candidate = entry.path() / L"label_classifier.xml";
                    if (std::filesystem::is_regular_file(candidate, d_ec)) {
                        const auto mt = std::filesystem::last_write_time(candidate, d_ec);
                        if (!d_ec) prev_xmls.push_back({mt, candidate});
                    }
                }
            }
            std::sort(prev_xmls.begin(), prev_xmls.end(), [](const XmlCandidate& a, const XmlCandidate& b) {
                return a.mtime > b.mtime;
            });

            std::vector<std::string> rejections;
            for (const auto& cand : prev_xmls) {
                const std::string cand_ver = path_to_utf8(cand.xml_path.parent_path().filename());
                const auto meta_opt = read_json_file(cand.xml_path.parent_path() / L"metrics.json");
                if (!meta_opt.has_value() || !meta_opt->is_object()) {
                    rejections.push_back(cand_ver + "(缺少历史训练溯源)");
                    continue;
                }
                const json& meta = *meta_opt;
                auto seen_grp = extract_json_str_set(meta, {"seen_groups", "train_groups"}, true, false);
                auto seen_smp = extract_json_str_set(meta, {"seen_sample_ids", "train_sample_ids"}, false, false);
                auto seen_img = extract_json_str_set(meta, {"seen_image_digests", "train_image_digests"}, false, true);
                auto seen_src = extract_json_str_set(meta, {"seen_source_digests", "train_source_digests"}, false, true);

                if (seen_grp.empty() && seen_smp.empty() && seen_img.empty()) {
                    rejections.push_back(cand_ver + "(缺少历史训练分组溯源)");
                    continue;
                }

                GroupSplit cand_split;
                try {
                    auto rows_copy = snapshotted_rows;
                    cand_split = select_group_split(rows_copy, seen_grp, seen_smp, seen_img, seen_src);
                } catch (const std::exception& split_ex) {
                    rejections.push_back(cand_ver + "(" + split_ex.what() + ")");
                    continue;
                }

                std::vector<uint8_t> xml_bytes;
                if (!read_file_bytes(cand.xml_path, xml_bytes) || xml_bytes.empty()) continue;
                const std::string xml_str(reinterpret_cast<const char*>(xml_bytes.data()), xml_bytes.size());
                auto loaded_mlp = load_mlp_from_xml_string(xml_str);
                if (!loaded_mlp || loaded_mlp->empty()) continue;

                const cv::Mat ls = loaded_mlp->getLayerSizes();
                if (ls.total() != 4) continue;
                cv::Mat ls_int;
                ls.convertTo(ls_int, CV_32S);
                if (ls_int.at<int>(0) != 64 || ls_int.at<int>(3) != 4) continue;

                mlp = loaded_mlp;
                resumed_from = cand_ver;
                resumed_seen_groups = std::move(seen_grp);
                resumed_seen_samples = std::move(seen_smp);
                resumed_seen_images = std::move(seen_img);
                resumed_seen_sources = std::move(seen_src);
                chosen_split = std::move(cand_split);
                break;
            }

            if (resumed_from.empty() && !rejections.empty()) {
                std::string joined;
                for (std::size_t i = 0; i < rejections.size(); ++i) {
                    if (i > 0) joined += "；";
                    joined += rejections[i];
                }
                throw std::invalid_argument(
                    "无法在不泄漏验证/测试留出集的前提下继续增量微调：" + joined +
                    "。请补充未训练过的独立料管组，或关闭增量微调从零训练。"
                );
            }
        }

        if (!chosen_split.has_value()) {
            chosen_split = select_group_split(snapshotted_rows);
        }

        const GroupSplit& split = *chosen_split;

        // Build training feature matrix X_train (N x 64 CV_32F) & target matrix Y_train (N x 4 CV_32F)
        std::vector<cv::Mat> x_train_list;
        std::vector<int> y_train_idx;
        for (std::size_t idx = 0; idx < split.train_rows.size(); ++idx) {
            const auto& r = split.train_rows[idx];
            const int cls_idx = label_class_index(r.label);
            if (cls_idx < 0) continue;
            cv::Mat img = imread_unicode(r.abs_path, cv::IMREAD_COLOR);
            if (img.empty()) continue;

            x_train_list.push_back(extract_label_features(img));
            y_train_idx.push_back(cls_idx);

            if (use_augmentation) {
                const auto augs = generate_shopfloor_augmentations(
                    img, static_cast<uint32_t>(2026 + idx * 31 + cls_idx * 7)
                );
                for (const auto& aug : augs) {
                    x_train_list.push_back(extract_label_features(aug));
                    y_train_idx.push_back(cls_idx);
                }
            }
        }

        if (x_train_list.size() < 4) {
            throw std::runtime_error("有效训练特征样本不足 4 张，无法训练 ANN_MLP");
        }

        auto build_eval_set = [](const std::vector<ManifestRow>& eval_rows) {
            std::vector<cv::Mat> feats;
            std::vector<int> labels;
            feats.reserve(eval_rows.size());
            labels.reserve(eval_rows.size());
            for (const auto& r : eval_rows) {
                const int cls_idx = label_class_index(r.label);
                if (cls_idx < 0) continue;
                cv::Mat img = imread_unicode(r.abs_path, cv::IMREAD_COLOR);
                if (img.empty()) continue;
                feats.push_back(extract_label_features(img));
                labels.push_back(cls_idx);
            }
            return std::make_pair(std::move(feats), std::move(labels));
        };

        auto [val_feats, val_labels] = build_eval_set(split.val_rows);
        auto [test_feats, test_labels] = build_eval_set(split.test_rows);
        if (val_feats.empty() || test_feats.empty()) {
            throw std::runtime_error("验证集或测试集图像读取为空");
        }

        // Deterministic shuffle of training samples
        std::vector<std::size_t> perm(x_train_list.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 shuf_rng(2026);
        std::shuffle(perm.begin(), perm.end(), shuf_rng);

        const int n_train = static_cast<int>(perm.size());
        cv::Mat x_train(n_train, 64, CV_32F);
        cv::Mat y_train(n_train, 4, CV_32F, cv::Scalar(-1.0f));
        for (int i = 0; i < n_train; ++i) {
            x_train_list[perm[static_cast<std::size_t>(i)]].copyTo(x_train.row(i));
            y_train.at<float>(i, y_train_idx[perm[static_cast<std::size_t>(i)]]) = 1.0f;
        }

        if (!mlp) {
            mlp = cv::ml::ANN_MLP::create();
            cv::Mat layer_sizes = (cv::Mat_<int>(1, 4) << 64, 32, 16, 4);
            mlp->setLayerSizes(layer_sizes);
            mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0.0, 0.0);
        }

        const double dw0 = std::clamp(lr * 20.0, 0.015, 0.12);
        mlp->setTrainMethod(cv::ml::ANN_MLP::RPROP, dw0, 1e-6);
        const int iters_per_epoch = std::clamp(240 / std::max(2, epochs), 10, 30);
        mlp->setTermCriteria(cv::TermCriteria(
            cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
            iters_per_epoch,
            1e-5
        ));

        auto train_data = cv::ml::TrainData::create(x_train, cv::ml::ROW_SAMPLE, y_train);

        struct EvalSummary {
            int samples{0};
            double accuracy{0.0};
            std::array<std::array<int, 4>, 4> matrix{};
            int complete_false_accept{0};
        };

        auto evaluate_mlp = [](const cv::Ptr<cv::ml::ANN_MLP>& model,
                               const std::vector<cv::Mat>& feats,
                               const std::vector<int>& labels) -> EvalSummary {
            EvalSummary res;
            res.samples = static_cast<int>(feats.size());
            for (auto& r : res.matrix) r.fill(0);
            int correct = 0;
            cv::Mat out;
            for (std::size_t i = 0; i < feats.size(); ++i) {
                model->predict(feats[i], out);
                int best_c = 0;
                float best_v = out.at<float>(0, 0);
                for (int c = 1; c < 4; ++c) {
                    const float v = out.at<float>(0, c);
                    if (v > best_v) {
                        best_v = v;
                        best_c = c;
                    }
                }
                const int t_idx = labels[i];
                res.matrix[static_cast<std::size_t>(t_idx)][static_cast<std::size_t>(best_c)] += 1;
                if (best_c == t_idx) ++correct;
            }
            res.accuracy = round_to(static_cast<double>(correct) / std::max(1, res.samples), 4);
            // False accept onto complete (column 0 when true class != 0, plus row 0 off-diagonal safety metric)
            int fa = 0;
            for (int t = 1; t < 4; ++t) {
                fa += res.matrix[static_cast<std::size_t>(t)][0];
            }
            res.complete_false_accept = fa;
            return res;
        };

        json history = json::array();
        double best_val_acc = -1.0;
        double best_train_loss = 1e9;
        std::string best_xml_str;

        for (int ep = 1; ep <= epochs; ++ep) {
            if (shutting_down_.load()) {
                throw std::runtime_error("训练线程收到停机信号已中止");
            }
            int flags = 0;
            if (ep > 1 || !resumed_from.empty()) {
                flags = cv::ml::ANN_MLP::UPDATE_WEIGHTS |
                        cv::ml::ANN_MLP::NO_INPUT_SCALE |
                        cv::ml::ANN_MLP::NO_OUTPUT_SCALE;
            }
            mlp->train(train_data, flags);

            cv::Mat preds;
            mlp->predict(x_train, preds);
            cv::Mat diff = preds - y_train;
            const double mse = cv::mean(diff.mul(diff))[0];
            const double train_loss = round_to(mse, 4);

            const EvalSummary val_res = evaluate_mlp(mlp, val_feats, val_labels);
            history.push_back(json{
                {"epoch", ep},
                {"train_loss", train_loss},
                {"val_accuracy", val_res.accuracy},
            });

            if (val_res.accuracy > best_val_acc ||
                (val_res.accuracy == best_val_acc && train_loss <= best_train_loss)) {
                best_val_acc = val_res.accuracy;
                best_train_loss = train_loss;
                best_xml_str = serialize_mlp_to_xml(mlp);
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                std::ostringstream msg_oss;
                msg_oss << "正在训练 OpenCV ANN_MLP：第 " << ep << "/" << epochs
                        << " 轮 (Loss: " << std::fixed << std::setprecision(4) << train_loss
                        << ", 验证集准确率: " << std::setprecision(1) << (val_res.accuracy * 100.0) << "%)";
                status_["state"] = "TRAINING";
                status_["message"] = msg_oss.str();
                status_["current_epoch"] = ep;
                status_["total_epochs"] = epochs;
                status_["progress_pct"] = static_cast<int>(10 + (static_cast<double>(ep) / epochs) * 75.0);
                status_["history"] = history;
                status_["updated_at_utc"] = utc_now_iso();
            }
        }

        if (!best_xml_str.empty()) {
            auto restored = load_mlp_from_xml_string(best_xml_str);
            if (restored && !restored->empty()) {
                mlp = restored;
            }
        } else {
            best_xml_str = serialize_mlp_to_xml(mlp);
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_["state"] = "EXPORTING_ONNX";
            status_["message"] = "正在序列化 C++ ANN_MLP 权重产物并执行一致性自检…";
            status_["progress_pct"] = 90;
            status_["updated_at_utc"] = utc_now_iso();
        }

        const auto xml_path = run_dir / L"label_classifier.xml";
        if (!write_file_bytes_atomic(xml_path, best_xml_str.data(), best_xml_str.size())) {
            throw std::runtime_error("写入 label_classifier.xml 失败");
        }

        // Verify disk round-trip reload & numerical parity
        std::vector<uint8_t> disk_xml_bytes;
        if (!read_file_bytes(xml_path, disk_xml_bytes) || disk_xml_bytes.empty()) {
            throw std::runtime_error("读取验证 label_classifier.xml 失败");
        }
        const std::string disk_xml_str(reinterpret_cast<const char*>(disk_xml_bytes.data()), disk_xml_bytes.size());
        auto reloaded_mlp = load_mlp_from_xml_string(disk_xml_str);
        if (!reloaded_mlp || reloaded_mlp->empty()) {
            throw std::runtime_error("重载验证 label_classifier.xml 失败");
        }

        double max_abs_diff = 0.0;
        bool parity_ok = true;
        for (std::size_t i = 0; i < std::min<std::size_t>(8, test_feats.size()); ++i) {
            cv::Mat out_mem, out_disk;
            mlp->predict(test_feats[i], out_mem);
            reloaded_mlp->predict(test_feats[i], out_disk);
            for (int c = 0; c < 4; ++c) {
                const float a = out_mem.at<float>(0, c);
                const float b = out_disk.at<float>(0, c);
                if (!std::isfinite(a) || !std::isfinite(b)) {
                    parity_ok = false;
                } else {
                    max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::abs(a - b)));
                }
            }
        }
        if (max_abs_diff >= 1e-4) {
            parity_ok = false;
        }

        const EvalSummary val_final = evaluate_mlp(reloaded_mlp, val_feats, val_labels);
        const EvalSummary test_final = evaluate_mlp(reloaded_mlp, test_feats, test_labels);
        const std::string xml_sha = sha256_hex(disk_xml_bytes);
        const std::string xml_path_str = path_to_utf8(xml_path);
        const double elapsed_s = round_to(std::max(0.01, monotonic_seconds() - t_start), 2);

        // Provenance sets
        std::set<std::string> train_sample_ids;
        std::set<std::string> train_image_digests;
        std::set<std::string> train_source_digests;
        bool any_synthetic = false;
        for (const auto& r : split.train_rows) {
            if (!r.sample_id.empty()) train_sample_ids.insert(r.sample_id);
            if (!r.image_sha256.empty()) train_image_digests.insert(to_lower_copy(r.image_sha256));
            if (!r.source_digest.empty()) train_source_digests.insert(to_lower_copy(r.source_digest));
            if (r.sample_id.rfind("STD_", 0) == 0) any_synthetic = true;
        }
        bool holdout_has_synthetic = false;
        for (const auto& r : split.val_rows) {
            if (r.sample_id.rfind("STD_", 0) == 0) holdout_has_synthetic = true;
        }
        for (const auto& r : split.test_rows) {
            if (r.sample_id.rfind("STD_", 0) == 0) holdout_has_synthetic = true;
        }
        any_synthetic = any_synthetic || holdout_has_synthetic;

        std::set<std::string> cum_groups(resumed_seen_groups.begin(), resumed_seen_groups.end());
        cum_groups.insert(split.train_groups.begin(), split.train_groups.end());
        std::set<std::string> cum_samples(resumed_seen_samples.begin(), resumed_seen_samples.end());
        cum_samples.insert(train_sample_ids.begin(), train_sample_ids.end());
        std::set<std::string> cum_images(resumed_seen_images.begin(), resumed_seen_images.end());
        cum_images.insert(train_image_digests.begin(), train_image_digests.end());
        std::set<std::string> cum_sources(resumed_seen_sources.begin(), resumed_seen_sources.end());
        cum_sources.insert(train_source_digests.begin(), train_source_digests.end());

        json cm_rows = json::array();
        for (int r = 0; r < 4; ++r) {
            cm_rows.push_back(json::array({
                test_final.matrix[static_cast<std::size_t>(r)][0],
                test_final.matrix[static_cast<std::size_t>(r)][1],
                test_final.matrix[static_cast<std::size_t>(r)][2],
                test_final.matrix[static_cast<std::size_t>(r)][3],
            }));
        }

        json run_summary = json{
            {"model_name", "opencv-mlp-cpp-igbt"},
            {"model_version", run_id},
            {"architecture", "cv::ml::ANN_MLP(64-32-16-4)"},
            {"artifact_format", "opencv_mlp"},
            {"resumed_from", resumed_from.empty() ? json(nullptr) : json(resumed_from)},
            {"use_augmentation", use_augmentation},
            {"input_size", input_size},
            {"batch_size", batch_size},
            {"epochs", epochs},
            {"elapsed_seconds", elapsed_s},
            {"train_samples", split.train_rows.size()},
            {"val_samples", split.val_rows.size()},
            {"test_samples", split.test_rows.size()},
            {"challenge_samples", split.challenges.size()},
            {"train_groups", split.train_groups},
            {"seen_groups", cum_groups},
            {"train_sample_ids", train_sample_ids},
            {"seen_sample_ids", cum_samples},
            {"train_image_digests", train_image_digests},
            {"seen_image_digests", cum_images},
            {"train_source_digests", train_source_digests},
            {"seen_source_digests", cum_sources},
            {"val_groups", split.val_groups},
            {"test_groups", split.test_groups},
            {"contains_synthetic_bootstrap", any_synthetic},
            {"evaluation_kind", holdout_has_synthetic ? "synthetic_smoke_test" : "independent_holdout"},
            {"independent_holdout_verified", !holdout_has_synthetic},
            {"val_accuracy", val_final.accuracy},
            {"test_accuracy", test_final.accuracy},
            {"confusion_matrix", json{
                {"labels", kLabelClasses},
                {"matrix", cm_rows},
            }},
            {"complete_false_accept_total", test_final.complete_false_accept},
            {"onnx_max_abs_diff", round_to(max_abs_diff, 7)},
            {"onnx_parity_ok", parity_ok},
            {"onnx_parity_passed", parity_ok},
            {"artifact_verified", parity_ok},
            {"model_path", xml_path_str},
            {"onnx_path", xml_path_str},
            {"sha256", xml_sha},
            {"onnx_sha256", xml_sha},
            {"weights_sha256", xml_sha},
            {"manifest_path", manifest_str},
            {"exported", parity_ok},
            {"activated", false},
            {"activation_status", on_complete ? "pending" : "exported_only"},
            {"activation_error", nullptr},
            {"completed_at_utc", utc_now_iso()},
        };

        const auto metrics_path = run_dir / L"metrics.json";
        if (!parity_ok) {
            std::error_code rm_ec;
            std::filesystem::remove(xml_path, rm_ec);
            const std::string err_msg = "ANN_MLP 序列化自检未通过，已阻止模型部署";
            run_summary["activation_status"] = "blocked_parity_failed";
            run_summary["activation_error"] = err_msg;
            write_json_atomic(metrics_path, run_summary);

            std::lock_guard<std::mutex> lock(mutex_);
            status_["state"] = "FAILED";
            status_["message"] = err_msg;
            status_["error"] = err_msg;
            status_["progress_pct"] = 100;
            status_["exported"] = false;
            status_["activated"] = false;
            status_["activation_status"] = "blocked_parity_failed";
            status_["latest_run"] = run_summary;
            status_["updated_at_utc"] = utc_now_iso();
            return;
        }

        write_json_atomic(metrics_path, run_summary);

        if (!on_complete) {
            std::lock_guard<std::mutex> lock(mutex_);
            std::ostringstream done_oss;
            done_oss << "C++ ANN_MLP 训练完成（测试集准确率 "
                     << std::fixed << std::setprecision(1) << (test_final.accuracy * 100.0)
                     << "%，已导出待激活）";
            status_["state"] = "COMPLETED";
            status_["message"] = done_oss.str();
            status_["progress_pct"] = 100;
            status_["exported"] = true;
            status_["activated"] = false;
            status_["activation_status"] = "exported_only";
            status_["latest_run"] = run_summary;
            status_["error"] = nullptr;
            status_["updated_at_utc"] = utc_now_iso();
            return;
        }

        // Keep state in-progress while calling on_complete BEFORE marking COMPLETED!
        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_["state"] = "EXPORTING_ONNX";
            status_["message"] = "模型自检通过，正在写入注册表并热加载至实时相机引擎…";
            status_["progress_pct"] = 95;
            status_["exported"] = true;
            status_["activated"] = false;
            status_["activation_status"] = "pending";
            status_["updated_at_utc"] = utc_now_iso();
        }

        bool activated = false;
        std::string activation_error;
        try {
            activated = on_complete(run_summary);
            if (!activated) {
                activation_error = "部署回调返回 false（实时引擎未切换到新模型）";
            }
        } catch (const std::exception& cb_ex) {
            activated = false;
            activation_error = cb_ex.what();
        } catch (...) {
            activated = false;
            activation_error = "部署回调抛出未知异常";
        }

        run_summary["activated"] = activated;
        run_summary["activation_status"] = activated ? "activated" : "activation_failed";
        run_summary["activation_error"] = activation_error.empty() ? json(nullptr) : json(activation_error);
        const bool persisted = write_json_atomic(metrics_path, run_summary);
        run_summary["metrics_persisted"] = persisted;
        run_summary["metrics_write_error"] = persisted ? json(nullptr) : json("metrics.json 写入失败");

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (activated) {
                std::ostringstream done_oss;
                done_oss << "C++ ANN_MLP 训练完成（测试集准确率 "
                         << std::fixed << std::setprecision(1) << (test_final.accuracy * 100.0)
                         << "%，已热加载至实时相机引擎）";
                status_["state"] = "COMPLETED";
                status_["message"] = done_oss.str();
                status_["progress_pct"] = 100;
                status_["exported"] = true;
                status_["activated"] = true;
                status_["activation_status"] = "activated";
                status_["metrics_persisted"] = persisted;
                status_["metrics_write_error"] = run_summary["metrics_write_error"];
                status_["latest_run"] = run_summary;
                status_["error"] = nullptr;
                status_["updated_at_utc"] = utc_now_iso();
            } else {
                const std::string fail_msg = "模型产物已保存，但热加载激活失败：" + activation_error;
                status_["state"] = "FAILED";
                status_["message"] = fail_msg;
                status_["error"] = fail_msg;
                status_["progress_pct"] = 100;
                status_["exported"] = true;
                status_["activated"] = false;
                status_["activation_status"] = "activation_failed";
                status_["metrics_persisted"] = persisted;
                status_["metrics_write_error"] = run_summary["metrics_write_error"];
                status_["latest_run"] = run_summary;
                status_["updated_at_utc"] = utc_now_iso();
            }
        }
    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lock(mutex_);
        status_["state"] = "FAILED";
        status_["message"] = std::string("训练异常终止：") + ex.what();
        status_["error"] = ex.what();
        status_["updated_at_utc"] = utc_now_iso();
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        status_["state"] = "FAILED";
        status_["message"] = "训练异常终止：未知错误";
        status_["error"] = "unknown error";
        status_["updated_at_utc"] = utc_now_iso();
    }
}

json TrainingManager::import_onnx_bytes(
    const std::vector<uint8_t>& onnx_bytes,
    const std::string& version_name,
    int input_size,
    int input_height
) {
    const auto r_dir = runs_dir();
    return training::OnnxModelRegistry::import_onnx_bytes(
        r_dir,
        onnx_bytes,
        version_name,
        input_size,
        input_height,
        [this](const std::string& prefix) {
            std::lock_guard<std::mutex> lock(mutex_);
            return allocate_run_dir_unlocked(prefix);
        }
    );
}

json TrainingManager::import_onnx_bytes(
    const std::string& onnx_bytes,
    const std::string& version_name,
    int input_size,
    int input_height
) {
    const std::vector<uint8_t> vec(
        reinterpret_cast<const uint8_t*>(onnx_bytes.data()),
        reinterpret_cast<const uint8_t*>(onnx_bytes.data()) + onnx_bytes.size()
    );
    return import_onnx_bytes(vec, version_name, input_size, input_height);
}

std::vector<VerifiedModelRun> TrainingManager::find_verified_runs() const {
    return training::OnnxModelRegistry::find_verified_runs(runs_dir());
}

std::optional<VerifiedModelRun> TrainingManager::resolve_verified_run(
    const std::string& model_version,
    const std::string& expected_sha256
) const {
    return training::OnnxModelRegistry::resolve_verified_run(runs_dir(), model_version, expected_sha256);
}

std::vector<std::pair<cv::Mat, int>> TrainingManager::collect_ood_calibration_crops(int max_per_class) const {
    const int limit = std::clamp(max_per_class, 1, 2000);
    std::vector<ManifestRow> rows;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        rows = read_manifest_rows_unlocked();
    }

    std::array<int, 4> id_counts{0, 0, 0, 0};
    int ood_count = 0;
    std::vector<std::pair<cv::Mat, int>> out;

    for (const auto& r : rows) {
        int cls_idx = label_class_index(r.label);
        if (cls_idx >= 0 && cls_idx < 4) {
            if (id_counts[static_cast<std::size_t>(cls_idx)] >= limit) continue;
        } else if (r.label == kChallengeLabel) {
            cls_idx = -1;
            if (ood_count >= limit) continue;
        } else {
            continue;
        }

        if (r.abs_path.empty()) continue;
        cv::Mat img = imread_unicode(r.abs_path, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        if (cls_idx >= 0) {
            ++id_counts[static_cast<std::size_t>(cls_idx)];
        } else {
            ++ood_count;
        }
        out.emplace_back(std::move(img), cls_idx);
    }
    return out;
}

std::filesystem::path TrainingManager::save_ood_calibration_for_version(
    const std::string& model_version,
    const vision::OodCalibration& cal
) const {
    return training::OnnxModelRegistry::save_ood_calibration(runs_dir(), model_version, cal);
}

}  // namespace igbt
