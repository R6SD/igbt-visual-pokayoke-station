#pragma once

#include "igbt/common.hpp"
#include "igbt/vision_engine.hpp"

#include <sqlite3.h>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace igbt {

class DatabaseHttpError : public std::runtime_error {
public:
    DatabaseHttpError(int status, const char* detail_msg)
        : std::runtime_error(detail_msg ? detail_msg : "Database error"),
          status_code_(status),
          detail_(json(detail_msg ? detail_msg : "Database error")) {}

    DatabaseHttpError(int status, std::string detail_msg)
        : std::runtime_error(detail_msg),
          status_code_(status),
          detail_(json(std::move(detail_msg))) {}

    DatabaseHttpError(int status, json detail_obj, const std::string& fallback_msg)
        : std::runtime_error(fallback_msg),
          status_code_(status),
          detail_(std::move(detail_obj)) {}

    int status_code() const noexcept { return status_code_; }
    const json& detail() const noexcept { return detail_; }

private:
    int status_code_{500};
    json detail_;
};

struct EvidenceFileResult {
    std::filesystem::path file_path;
    std::string mime_type;
};

class StationDatabase {
public:
    explicit StationDatabase(
        std::filesystem::path data_root,
        std::optional<std::filesystem::path> explicit_db_path = std::nullopt
    );
    ~StationDatabase();

    StationDatabase(const StationDatabase&) = delete;
    StationDatabase& operator=(const StationDatabase&) = delete;

    const std::filesystem::path& data_root() const noexcept { return data_root_; }
    const std::filesystem::path& db_path() const noexcept { return db_path_; }
    const std::filesystem::path& evidence_root() const noexcept { return evidence_root_; }

    static json default_config();
    static json default_station_state();
    static std::string validate_cycle_id(const std::string& raw_cycle_id);
    static json validate_config_update(const json& payload);

    // Station state & summary
    json get_persisted_station_state() const;
    void set_persisted_station_state(const json& state_data);
    json cycle_summary_all() const;
    json attach_latest_evidence_and_summary(json state) const;

    // Configuration CRUD (config_current + configuration_history)
    json get_config() const;
    json update_config(const json& payload, const std::string& actor);
    json list_config_history(int limit = 50) const;

    // Hardware Camera Optics CRUD (camera_optics singleton)
    json get_camera_optics() const;
    json save_camera_optics(const json& payload, const std::string& actor);

    // Cycles & Events CRUD
    json list_cycles(int limit = 100) const;
    json list_cycle_events(const std::string& cycle_id, int limit = 200) const;
    json delete_cycle(const std::string& cycle_id);
    json purge_cycles(const std::string& scope);

    // Live / Simulated / Station Runtime Persistence
    bool record_live_transition(
        const json& state_data,
        const std::optional<std::string>& state_before,
        const std::vector<uint8_t>& evidence_jpeg = {}
    );

    json save_manual_capture(
        const std::vector<uint8_t>& jpeg_bytes,
        const json& live_state,
        const std::string& fallback_cycle_id
    );

    json record_simulated_run(
        const SimulationRunResult& run,
        const std::string& requested_scenario,
        const json& validated_state
    );

    json record_runtime_snapshot(const json& validated_snapshot);

    // Evidence CRUD
    json upload_evidence_bytes(
        const std::string& cycle_id,
        const std::string& kind,
        const std::vector<uint8_t>& body,
        const std::string& mime_type,
        int64_t frame_id,
        std::optional<double> capture_monotonic_s,
        const std::optional<std::string>& pipe_color,
        const std::optional<std::string>& operation_event,
        const std::optional<std::string>& operator_anon_id,
        const json& camera_metadata,
        const std::optional<std::string>& label_status,
        const std::string& captured_utc_iso
    );

    json archive_dataset_sample_evidence(
        const std::vector<uint8_t>& frame_jpeg,
        const json& sample_info,
        const std::string& label,
        const std::string& pipe_color,
        const json& live_state,
        const std::string& fallback_cycle_id
    );

    json list_evidence(int limit = 100) const;
    EvidenceFileResult get_evidence_file(const std::string& evidence_id) const;
    json delete_evidence(const std::string& evidence_id);

    // Model Registry CRUD
    json register_model(
        const std::string& model_name,
        const std::string& model_version,
        const std::string& sha256_hex_str,
        const std::string& license_status,
        const std::string& notes,
        const std::string& registered_by
    );
    json list_models(int limit = 50) const;
    std::optional<json> get_model_by_id(const std::string& model_id) const;
    json delete_model(const std::string& model_id, const std::string& active_model_version);

    // Storage Overview & WAL Maintenance
    json storage_overview(const json& training_stats, const std::string& active_model_version) const;
    json enforce_evidence_quota(
        std::uint64_t max_bytes = 50ULL * 1024 * 1024 * 1024,
        double min_free_ratio = 0.15
    );
    void run_wal_checkpoint_truncate();
    json pragma_diagnostics() const;

    // Red Rabbit Golden-Sample Challenge & Hash-Chain Audit Repository (v4.3 AEC-Q101)
    json rr_list_samples(bool active_only = false) const;
    std::optional<json> rr_get_sample(const std::string& sample_uid) const;
    json rr_upsert_sample(const json& payload, const std::string& actor);

    json rr_create_session(const json& session_obj);
    std::optional<json> rr_get_session(const std::string& session_id) const;
    json rr_list_sessions(int limit = 30) const;
    void rr_update_session_status(
        const std::string& session_id,
        const std::string& status,
        std::optional<int64_t> completed_at_ms = std::nullopt
    );

    json rr_record_attempt(const json& attempt_obj);
    json rr_list_attempts(const std::string& session_id) const;
    std::optional<json> rr_last_passed_session() const;

    json rr_append_lock_audit(
        const std::string& event_type,
        const std::string& actor_id,
        const std::string& actor_role,
        const std::string& reason_code,
        const std::string& comment,
        const std::optional<std::string>& related_session,
        int64_t created_at_ms
    );
    json rr_list_lock_audit(int limit = 50) const;
    json rr_verify_lock_audit_chain() const;

private:
    void open_and_initialize();
    void exec_sql_unlocked(const std::string& sql) const;
    bool column_exists_unlocked(const std::string& table_name, const std::string& col_name) const;
    void backfill_legacy_cycle_sources_unlocked();

    json latest_config_unlocked() const;
    json cycle_summary_all_unlocked() const;
    json attach_latest_evidence_and_summary_unlocked(json state) const;
    bool is_stale_live_transition_unlocked(const json& state_data) const;

    std::string insert_event_unlocked(
        const std::optional<std::string>& cycle_id,
        const std::string& event_type,
        const json& payload,
        const std::optional<std::string>& state_before = std::nullopt,
        const std::optional<std::string>& state_after = std::nullopt,
        const std::optional<std::string>& outcome = std::nullopt,
        const std::optional<std::string>& reason = std::nullopt,
        std::optional<double> capture_monotonic_s = std::nullopt,
        std::optional<int64_t> frame_id = std::nullopt,
        const std::optional<std::string>& explicit_event_id = std::nullopt,
        const std::optional<std::string>& explicit_created_utc = std::nullopt
    );

    bool safe_unlink_evidence_file(const std::string& file_relpath) const;
    std::vector<std::filesystem::path> snapshot_evidence_files() const;
    int safe_clean_evidence_dir(const std::vector<std::filesystem::path>& allowed_files) const;
    std::optional<std::filesystem::path> safe_model_run_dir(const std::string& version) const;

    std::filesystem::path data_root_;
    std::filesystem::path db_path_;
    std::filesystem::path evidence_root_;

    mutable std::mutex mutex_;
    sqlite3* db_{nullptr};

    std::optional<double> purge_cutoff_monotonic_;
    std::unordered_set<std::string> deleted_cycles_;
    static constexpr std::size_t kMaxTrackedDeletedCycles = 2048;
    int write_ops_since_checkpoint_{0};
};

}  // namespace igbt
