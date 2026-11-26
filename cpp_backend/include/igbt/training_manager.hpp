#pragma once

#include "igbt/common.hpp"
#include "igbt/training/onnx_model_registry.hpp"

#include <atomic>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace igbt {

inline constexpr int kSavedSampleWidth = 224;
inline constexpr int kSavedSampleHeight = 144;
inline constexpr const char* kUngroupedTubeId = "UNGROUPED";
inline constexpr const char* kUngroupedBatchId = "BATCH_UNGROUPED";
inline constexpr const char* kUngroupedVideoGroupId = "VID_UNGROUPED";
inline constexpr const char* kChallengeLabel = "unknown_challenge";

inline const std::array<std::string, 5> kAllDatasetLabels = {
    "complete", "crossed_out", "covered", "absent", "unknown_challenge"
};

class TrainingManager {
public:
    using OnCompleteCallback = std::function<bool(const json& metrics_json)>;

    explicit TrainingManager(std::filesystem::path data_root);
    explicit TrainingManager(std::function<std::filesystem::path()> data_root_getter);
    ~TrainingManager();

    TrainingManager(const TrainingManager&) = delete;
    TrainingManager& operator=(const TrainingManager&) = delete;

    // Paths
    std::filesystem::path data_root() const;
    std::filesystem::path dataset_dir() const;
    std::filesystem::path manifest_path() const;
    std::filesystem::path coco_json_path() const;
    std::filesystem::path runs_dir() const;

    // Dataset CRUD & COCO JSON
    json summary() const;
    json get_dataset_summary() const { return summary(); }

    json list_samples(
        const std::string& label_filter = "",
        const std::string& search_query = "",
        int limit = 200
    ) const;

    json capture_sample(
        const cv::Mat& frame_bgr,
        const cv::Mat& crop_bgr,
        const std::string& label,
        const std::string& pipe_color = "yellow",
        const std::string& tube_id = "",
        const std::string& batch_id = "",
        std::optional<Box> frame_box = std::nullopt,
        std::optional<Box> crop_bounds_xyxy = std::nullopt
    );

    json capture_sample(
        const cv::Mat& crop_bgr,
        const std::string& label,
        const std::string& pipe_color = "yellow",
        const std::string& tube_id = "",
        const std::string& batch_id = "",
        std::optional<Box> frame_box = std::nullopt
    );

    json upload_sample_bytes(
        const std::vector<uint8_t>& image_bytes,
        const std::string& label,
        const std::string& pipe_color = "yellow",
        const std::string& tube_id = "",
        const std::string& batch_id = ""
    );

    json upload_sample_bytes(
        const std::string& image_bytes,
        const std::string& label,
        const std::string& pipe_color = "yellow",
        const std::string& tube_id = "",
        const std::string& batch_id = ""
    );

    json update_sample_annotation(
        const std::string& sample_id,
        const std::optional<std::string>& new_label = std::nullopt,
        const std::optional<std::string>& new_tube_id = std::nullopt,
        const std::optional<std::string>& new_batch_id = std::nullopt,
        const std::optional<std::string>& new_pipe_color = std::nullopt
    );

    json delete_sample(const std::string& sample_id);

    json batch_delete_samples(
        const std::vector<std::string>& sample_ids,
        const std::string& label_filter = ""
    );

    json clear_dataset();

    json bootstrap_standard_dataset(std::optional<cv::Mat> live_crop = std::nullopt);

    json rebuild_coco_from_manifest();

    json storage_stats() const;

    std::optional<std::filesystem::path> get_sample_image_path(const std::string& sample_id) const;

    // Pure C++ OpenCV ANN_MLP Training & ONNX Import
    json status() const;
    json get_training_status() const { return status(); }
    bool is_training() const;
    void wait_for_completion();

    json start_training(
        int epochs = 12,
        int input_size = kDefaultInputSize,
        int batch_size = 8,
        double lr = 0.0025,
        bool resume_from_latest = true,
        bool use_augmentation = true,
        OnCompleteCallback on_complete = nullptr
    );

    json import_onnx_bytes(
        const std::vector<uint8_t>& onnx_bytes,
        const std::string& version_name = "",
        int input_size = kDefaultInputSize,
        int input_height = 0
    );

    json import_onnx_bytes(
        const std::string& onnx_bytes,
        const std::string& version_name = "",
        int input_size = kDefaultInputSize,
        int input_height = 0
    );

    std::vector<VerifiedModelRun> find_verified_runs() const;

    std::optional<VerifiedModelRun> resolve_verified_run(
        const std::string& model_version,
        const std::string& expected_sha256 = ""
    ) const;

    std::vector<std::pair<cv::Mat, int>> collect_ood_calibration_crops(int max_per_class = 150) const;

    std::filesystem::path save_ood_calibration_for_version(
        const std::string& model_version,
        const vision::OodCalibration& cal
    ) const;

private:
    struct ManifestRow {
        std::string sample_id;
        std::string path;
        std::string label;
        std::string physical_tube_id;
        std::string batch_id;
        std::string video_group_id;
        std::string pipe_color;
        std::string box_xyxy;
        std::string crop_box_xyxy;
        std::string source_digest;
        std::string captured_at_utc;
        std::filesystem::path abs_path;
        std::string image_sha256;
    };

    struct GroupSplit {
        std::vector<ManifestRow> train_rows;
        std::vector<ManifestRow> val_rows;
        std::vector<ManifestRow> test_rows;
        std::vector<ManifestRow> challenges;
        std::vector<std::string> train_groups;
        std::vector<std::string> val_groups;
        std::vector<std::string> test_groups;
        int ungrouped_count{0};
    };

    std::vector<ManifestRow> read_manifest_rows_unlocked() const;
    void write_all_manifest_rows_unlocked(const std::vector<ManifestRow>& rows) const;
    void append_manifest_row_unlocked(const ManifestRow& row) const;
    json build_dataset_summary_unlocked(const std::vector<ManifestRow>& rows) const;

    std::tuple<bool, bool, std::string> try_unlink_in_dataset_unlocked(const std::filesystem::path& file_path) const;

    static json empty_coco_doc();
    std::pair<json, bool> load_coco_doc_unlocked(const std::filesystem::path& coco_path) const;
    static void append_coco_entry(
        json& coco,
        int img_id,
        int ann_id,
        const std::string& sample_id,
        const std::string& rel_path,
        const std::string& label,
        const Box& crop_box_xyxy,
        int image_width = kSavedSampleWidth,
        int image_height = kSavedSampleHeight
    );
    bool append_coco_annotations_unlocked(const std::vector<json>& entries) const;
    void sync_coco_delete_unlocked(const std::vector<std::string>& deleted_ids) const;
    void sync_coco_update_unlocked(
        const std::string& sample_id,
        const std::string& new_rel_path,
        const std::string& new_label
    ) const;

    json save_sample_image_impl(
        const cv::Mat& crop_bgr,
        const std::string& label,
        const std::string& pipe_color,
        const std::string& physical_tube_id,
        const std::string& batch_id,
        std::optional<Box> box_xyxy,
        const std::string& id_prefix,
        const std::string& box_fallback,
        const Box& coco_box_xyxy
    );

    std::pair<std::string, std::filesystem::path> allocate_run_dir_unlocked(const std::string& prefix) const;
    std::pair<std::vector<ManifestRow>, std::filesystem::path> snapshot_manifest_for_run_unlocked(
        const std::vector<ManifestRow>& rows,
        const std::filesystem::path& run_dir
    ) const;

    static GroupSplit select_group_split(
        std::vector<ManifestRow>& rows,
        const std::vector<std::string>& forbidden_groups = {},
        const std::vector<std::string>& forbidden_samples = {},
        const std::vector<std::string>& forbidden_images = {},
        const std::vector<std::string>& forbidden_sources = {}
    );

    void run_training_worker(
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
    );

    std::function<std::filesystem::path()> data_root_getter_;
    mutable std::mutex mutex_;
    mutable std::mutex worker_join_mutex_;
    std::thread worker_thread_;
    std::atomic<bool> shutting_down_{false};
    json status_;
};

}  // namespace igbt
